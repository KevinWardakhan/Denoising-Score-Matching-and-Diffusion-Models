import math
import functools
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # au cas où
from copy import deepcopy
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from tqdm.auto import tqdm, trange

warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


########################
# SDE UTILS + LOSS
########################

def marginal_prob_std(t, sigma):
    """
    Compute the std of p_{0t}(x(t) | x(0)).

    Args:
      t: tensor of shape (B,)
      sigma: scalar
    """
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2.0 * t) - 1.0) / (2.0 * np.log(sigma)))


def diffusion_coeff(t, sigma):
    """
    Diffusion coefficient g(t).
    """
    return torch.tensor(sigma ** t, device=device)


def differentiate_diffusion_coeff(t, sigma):
    """
    Derivative g'(t) of the diffusion coefficient.
    """
    return torch.tensor(t * sigma ** (t - 1.0), device=device)


def loss_sn(model, x, marginal_prob_std, eps=1e-6):
    """
    Denoising Score Matching loss from Song et al.
    """
    # t ~ U(eps, 1)
    random_t = torch.rand(x.shape[0], device=x.device) * (1.0 - eps) + eps
    z = torch.randn_like(x)  # N(0, I)

    std = marginal_prob_std(random_t)  # (B,)
    std_x = std[:, None, None, None]

    # x(t) = x(0) + std(t) * z
    perturbed_x = x + z * std_x

    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std_x + z) ** 2, dim=(1, 2, 3)))
    return loss


########################
# PC SAMPLER
########################

signal_to_noise_ratio = 0.16
num_steps = 500


def pc_sampler(
    score_model,
    marginal_prob_std,
    diffusion_coeff,
    batch_size=64,
    num_steps=num_steps,
    snr=signal_to_noise_ratio,
    device=device,
    eps=1e-3,
    channels=3,
    image_size=32,
):
    """
    Predictor Corrector sampler (VE-SDE).
    """
    t = torch.ones(batch_size, device=device)
    std = marginal_prob_std(t)[:, None, None, None]

    # pour CIFAR10: 3 x 32 x 32
    x = torch.randn(batch_size, channels, image_size, image_size, device=device) * std

    time_steps = np.linspace(1.0, eps, num_steps)
    step_size = time_steps[0] - time_steps[1]

    score_model.eval()
    with torch.no_grad():
        for time_step in tqdm(time_steps, desc="Sampling"):
            batch_time_step = torch.ones(batch_size, device=device) * float(time_step)

            # Corrector: Langevin MCMC
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2.0 * (snr * noise_norm / grad_norm) ** 2

            x = x + langevin_step_size * grad
            x = x + torch.sqrt(2.0 * langevin_step_size) * torch.randn_like(x)

            # Predictor: Euler Maruyama
            g = diffusion_coeff(batch_time_step)
            g2 = (g ** 2)[:, None, None, None]

            x_mean = x + g2 * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g2 * step_size) * torch.randn_like(x)

    return x_mean


########################
# MODEL
########################

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False
        )

    def forward(self, t):
        """
        t: (B,) or (B, 1)
        returns: (B, embed_dim)
        """
        if t.dim() == 1:
            t = t[:, None]
        x_proj = t * self.W[None, :] * 2.0 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResBlock(nn.Module):
    """Bloc résiduel avec GroupNorm et injection de temps."""
    def __init__(self, in_ch, out_ch, embed_dim, groups=32, dropout=0.0):
        super().__init__()
        self.act = lambda x: x * torch.sigmoid(x)

        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.embed = nn.Linear(embed_dim, out_ch)

        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        h = h + self.embed(emb)[:, :, None, None]

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.skip(x) + h


class AttentionBlock(nn.Module):
    """Self-attention sur les feature maps."""
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        h_ = self.norm(x)

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        head_dim = c // self.num_heads

        # (B, heads, L, head_dim) avec L = H*W
        q = q.view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        k = k.view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)
        v = v.view(b, self.num_heads, head_dim, h * w).permute(0, 1, 3, 2)

        scale = 1.0 / math.sqrt(head_dim)
        attn = torch.softmax(
            torch.matmul(q * scale, k.transpose(-1, -2)),
            dim=-1
        )  # (B, heads, L, L)

        out = torch.matmul(attn, v)  # (B, heads, L, head_dim)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)

        out = self.proj(out)
        return x + out


class ScoreNetBig(nn.Module):
    """
    U-Net résiduel + attention pour CIFAR10.

    Entrée:  (B, 3, 32, 32)
    Sortie:  (B, 3, 32, 32)
    """
    def __init__(
        self,
        marginal_prob_std,
        in_channels=3,
        out_channels=3,
        channels=(128, 256, 256),
        embed_dim=256,
        attn_resolutions=(16,),
        num_heads=4,
        dropout=0.0,
    ):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        self.act = lambda x: x * torch.sigmoid(x)

        # Time embedding
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        C0, C1, C2 = channels

        # Encoder
        self.in_conv = nn.Conv2d(in_channels, C0, 3, padding=1)

        # 32x32
        self.down1_block1 = ResBlock(C0, C0, embed_dim, dropout=dropout)
        self.down1_block2 = ResBlock(C0, C0, embed_dim, dropout=dropout)
        self.down1_attn = AttentionBlock(C0, num_heads) if 32 in attn_resolutions else nn.Identity()
        self.down1_downsample = nn.Conv2d(C0, C1, 3, stride=2, padding=1)  # 32 -> 16

        # 16x16
        self.down2_block1 = ResBlock(C1, C1, embed_dim, dropout=dropout)
        self.down2_block2 = ResBlock(C1, C1, embed_dim, dropout=dropout)
        self.down2_attn = AttentionBlock(C1, num_heads) if 16 in attn_resolutions else nn.Identity()
        self.down2_downsample = nn.Conv2d(C1, C2, 3, stride=2, padding=1)  # 16 -> 8

        # bottleneck 8x8
        self.mid_block1 = ResBlock(C2, C2, embed_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(C2, num_heads) if 8 in attn_resolutions else nn.Identity()
        self.mid_block2 = ResBlock(C2, C2, embed_dim, dropout=dropout)

        # Decoder
        # 8 -> 16
        self.up2_upsample = nn.ConvTranspose2d(C2, C1, 4, stride=2, padding=1)
        self.up2_block1 = ResBlock(C1 + C1, C1, embed_dim, dropout=dropout)
        self.up2_block2 = ResBlock(C1, C1, embed_dim, dropout=dropout)
        self.up2_attn = AttentionBlock(C1, num_heads) if 16 in attn_resolutions else nn.Identity()

        # 16 -> 32
        self.up1_upsample = nn.ConvTranspose2d(C1, C0, 4, stride=2, padding=1)
        self.up1_block1 = ResBlock(C0 + C0, C0, embed_dim, dropout=dropout)
        self.up1_block2 = ResBlock(C0, C0, embed_dim, dropout=dropout)
        self.up1_attn = AttentionBlock(C0, num_heads) if 32 in attn_resolutions else nn.Identity()

        self.out_conv = nn.Conv2d(C0, out_channels, 3, padding=1)

    def forward(self, x, t):
        """
        x: (B, 3, 32, 32)
        t: (B,)
        """
        emb = self.act(self.embed(t))  # (B, embed_dim)

        # Encoder
        h = self.in_conv(x)

        # 32x32
        h = self.down1_block1(h, emb)
        h = self.down1_block2(h, emb)
        h = self.down1_attn(h)
        h1 = h
        h = self.down1_downsample(h)  # 16x16

        # 16x16
        h = self.down2_block1(h, emb)
        h = self.down2_block2(h, emb)
        h = self.down2_attn(h)
        h2 = h
        h = self.down2_downsample(h)  # 8x8

        # bottleneck 8x8
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        # Decoder
        # 8 -> 16
        h = self.up2_upsample(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up2_block1(h, emb)
        h = self.up2_block2(h, emb)
        h = self.up2_attn(h)

        # 16 -> 32
        h = self.up1_upsample(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up1_block1(h, emb)
        h = self.up1_block2(h, emb)
        h = self.up1_attn(h)

        h = self.out_conv(h)

        # normalisation par la std marginale
        std = self.marginal_prob_std(t)[:, None, None, None]
        return h / std


########################
# EMA UTILS
########################

def update_ema(ema_model, model, decay):
    """
    ema_model.params = decay * ema_model.params + (1-decay) * model.params
    """
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for name, p_ema in ema_params.items():
            p_model = model_params[name]
            p_ema.data.mul_(decay).add_(p_model.data, alpha=1.0 - decay)


########################
# MAIN TRAINING
########################

def main():
    # Hyperparams
    n_epochs = 120
    batch_size = 32
    lr = 1e-4
    ema_decay = 0.999
    sigma = 50.0

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
    differentiate_diffusion_fn = functools.partial(differentiate_diffusion_coeff, sigma=sigma)

    # Model + EMA
    score_model = ScoreNetBig(
        marginal_prob_std=marginal_prob_std_fn,
        in_channels=3,
        out_channels=3,
    )
    score_model = torch.nn.DataParallel(score_model).to(device)

    ema_score_model = deepcopy(score_model)
    ema_score_model.eval()
    for p in ema_score_model.parameters():
        p.requires_grad_(False)

    total_params = sum(p.numel() for p in score_model.parameters())
    trainable_params = sum(p.numel() for p in score_model.parameters() if p.requires_grad)
    print("Total params:", total_params)
    print("Trainable params:", trainable_params)

    # Data
    transform = transforms.Compose([
        transforms.Resize(32),  # <== 32x32
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])

    dataset = CIFAR10(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    optimizer = Adam(score_model.parameters(), lr=lr, amsgrad=True)

    # Training loop
    for epoch in trange(n_epochs, desc="Epochs"):
        avg_loss = 0.0
        num_items = 0

        score_model.train()
        for x, _ in data_loader:
            x = x.to(device)

            loss = loss_sn(score_model, x, marginal_prob_std_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema(ema_score_model, score_model, ema_decay)

            avg_loss += loss.item() * x.size(0)
            num_items += x.size(0)

        epoch_loss = avg_loss / num_items
        print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {epoch_loss:.5f}")

        # Sauvegarde
        torch.save(score_model.state_dict(), "ckpt_last.pth")
        torch.save(ema_score_model.state_dict(), "ckpt_last_ema.pth")

        # Echantillonnage rapide tous les 20 epochs
        if (epoch + 1) % 20 == 0:
            samples = pc_sampler(
                ema_score_model,
                marginal_prob_std_fn,
                diffusion_coeff_fn,
                batch_size=64,
                device=device,
                channels=3,
                image_size=32,
            )
            # revenir dans [0,1] avant sauvegarde
            samples = samples.clamp(-1.0, 1.0)
            samples = (samples + 1.0) / 2.0
            save_image(samples, f"samples_epoch_{epoch+1}.png", nrow=8)
            print(f"Saved samples to samples_epoch_{epoch+1}.png")


if __name__ == "__main__":
    main()
