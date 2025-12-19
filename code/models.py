import torch
import torch.nn as nn


class SigmaEmbedding(nn.Module):
    """
    Embed scalar sigma (B,), (B,1) or (B,1,1,1) -> (B, emb_dim)
    via log-sigma + MLP.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        # Accept (B,), (B,1), (B,1,1,1)
        if sigma.dim() == 4:          # (B,1,1,1)
            sigma = sigma.view(sigma.size(0), 1)
        elif sigma.dim() == 1:        # (B,)
            sigma = sigma.unsqueeze(1)
        elif sigma.dim() == 2:        # (B,1)
            pass
        else:
            raise ValueError(f"Unexpected sigma shape: {sigma.shape}")

        if sigma.size(1) != 1:
            raise ValueError(f"Sigma should have shape (B,1), got {sigma.shape}")

        log_sigma = torch.log(sigma)
        return self.net(log_sigma)    # (B, emb_dim)


class ResBlock(nn.Module):
    """
    Conv-ResNet block modulated by sigma embedding via per-channel bias.
    """
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_ch), num_channels=in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # Projections de l'embedding -> biais par canal (out_ch)
        self.emb_proj1 = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))
        self.emb_proj2 = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, in_ch, H, W)
        emb: (B, emb_dim)
        """
        h = self.conv1(self.act1(self.norm1(x)))
        # injection biais après conv1
        h = h + self.emb_proj1(emb).unsqueeze(-1).unsqueeze(-1)

        h = self.conv2(self.act2(self.norm2(h)))
        # injection biais après conv2
        h = h + self.emb_proj2(emb).unsqueeze(-1).unsqueeze(-1)

        return h + self.skip(x)



class SmallUNetSigma(nn.Module):
    """
    Small U-Net for images, conditioned on scalar sigma.

    - in_ch:       1 (MNIST), 3 (CIFAR), etc.
    - base_ch:     base number of channels (e.g. 64)
    - channel_mults: per-level multipliers, e.g. (1,2,4)
    - emb_dim:     sigma embedding dim
    """
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 64,
        channel_mults=(1, 2, 4),
        emb_dim: int = 128,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.base_ch = base_ch
        self.channel_mults = channel_mults

        self.sigma_emb = SigmaEmbedding(emb_dim)

        # Initial projection
        self.init_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # ----- ENCODER -----
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        curr_ch = base_ch
        num_levels = len(channel_mults)

        for level in range(num_levels):
            out_ch = base_ch * channel_mults[level]
            self.down_blocks.append(ResBlock(curr_ch, out_ch, emb_dim))
            curr_ch = out_ch

            # Downsample after this block, except for the deepest level
            if level < num_levels - 1:
                self.down_samples.append(
                    nn.Conv2d(curr_ch, curr_ch, kernel_size=3, stride=2, padding=1)
                )

        self.bottleneck_ch = curr_ch

        # ----- BOTTLENECK -----
        self.mid_block1 = ResBlock(curr_ch, curr_ch, emb_dim)
        self.mid_block2 = ResBlock(curr_ch,curr_ch,emb_dim)

        # ----- DECODER -----
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # On remonte les niveaux (sauf le bottleneck) en sens inverse
        # Les skips seront pris sur les sorties des blocks encoder avant downsample
        for level in reversed(range(num_levels - 1)):
            skip_ch = base_ch * channel_mults[level]

            # upsample: garder curr_ch
            self.up_samples.append(
                nn.ConvTranspose2d(
                    curr_ch, curr_ch,
                    kernel_size=4, stride=2, padding=1
                )
            )
            # après concat: curr_ch (up) + skip_ch
            self.up_blocks.append(
                ResBlock(curr_ch + skip_ch, skip_ch, emb_dim)
            )

            curr_ch = skip_ch

        # Final projection to input channels
        self.final_ch = curr_ch
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=min(8, self.final_ch), num_channels=self.final_ch),
            nn.SiLU(),
            nn.Conv2d(self.final_ch, in_ch, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        x:     (B, C, H, W)
        sigma: (B,), (B,1), or (B,1,1,1)
        """
        B, C, H, W = x.shape
        emb = self.sigma_emb(sigma)  # (B, emb_dim)

        # ----- ENCODE -----
        h = self.init_conv(x)
        skips = []
        ds_idx = 0

        for level, block in enumerate(self.down_blocks):
            h = block(h, emb)

            # si on n'est pas au niveau le plus profond,
            # on stocke le skip AVANT le downsample
            if level < len(self.down_samples):
                skips.append(h)
                h = self.down_samples[ds_idx](h)
                ds_idx += 1

        # ----- BOTTLENECK -----
        h = self.mid_block1(h, emb)
        h = self.mid_block2(h,emb)
        # ----- DECODE -----
        # on consomme les skips du plus profond au plus proche de l'entrée
        for up, block in zip(self.up_samples, self.up_blocks):
            h = up(h)

            skip = skips.pop()  # shape compatible en canaux

            # align spatial dimensions si besoin (off-by-one)
            if h.shape[-2:] != skip.shape[-2:]:
                H = min(h.shape[-2], skip.shape[-2])
                W = min(h.shape[-1], skip.shape[-1])
                h = h[:, :, :H, :W]
                skip = skip[:, :, :H, :W]

            h = torch.cat([h, skip], dim=1)
            h = block(h, emb)

        out = self.out_conv(h)
        return out

