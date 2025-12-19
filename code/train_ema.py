#!/usr/bin/env python

import argparse
import os
import json
import shutil
import copy  # EMA

import matplotlib
matplotlib.use("Agg")  # pour le headless HPC
import matplotlib.pyplot as plt

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import models as m  

CELEBA_ROOT = "data/celeba"
device = torch.device("cpu")


def LossVE_SDE(model, x, sigma_min, sigma_max):
    B = x.size(0)
    device_x = x.device

    t_uniform = torch.rand(B, device=device_x)  # (B,)

    log_sigma_min = torch.log(torch.tensor(sigma_min, device=device_x))
    log_sigma_max = torch.log(torch.tensor(sigma_max, device=device_x))
    log_sigma_t = log_sigma_min + t_uniform * (log_sigma_max - log_sigma_min)
    sigma_t = torch.exp(log_sigma_t)  # (B,)

    sigma_t_broadcast = sigma_t.view(B, 1, 1, 1)

    eps = torch.randn_like(x)
    x_t = x + sigma_t_broadcast * eps

    sigma_input = sigma_t.view(B, 1)  # (B,1)
    s_hat = model(x_t, sigma_input)

    target = -eps / (sigma_t_broadcast + 1e-8)

    w = (sigma_t ** 2).view(B, 1, 1, 1)

    residual = (s_hat - target) ** 2
    loss_per_sample = (w * residual).view(B, -1).mean(dim=1)
    loss = loss_per_sample.mean()
    return loss


def parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def get_args():
    parser = argparse.ArgumentParser(
        description="Score-based VE-SDE model on CelebA (train only)"
    )

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--exp-name", type=str, default="debugging")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="max_norm for clip_grad_norm_. Set 0 to disable."
    )

    parser.add_argument("--sigma-min", type=float, default=1e-1)
    parser.add_argument("--sigma-max", type=float, default=0.5)

    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--channel-mults", type=str, default="1,2,4")
    parser.add_argument("--sigma-emb-dim", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=128)

    # EMA
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay (<=0 pour désactiver l'EMA)."
    )

    # Sauvegarde epoch : tous les k epochs à partir d'une fraction du training
    parser.add_argument(
        "--save-epoch-interval",
        type=int,
        default=5,
        help="Sauvegarder les poids tous les k epochs (après le démarrage)."
    )
    parser.add_argument(
        "--save-epoch-start-frac",
        type=float,
        default=0.5,
        help="Fraction du nombre total d'epochs à partir de laquelle on commence à sauvegarder."
    )

    return parser.parse_args()


def main(args):
    global device

    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA runtime:", torch.version.cuda)
        print("cuDNN:", torch.backends.cudnn.version())
        print("GPU:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))

    torch.set_default_dtype(torch.float32)

    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    SAVE = not args.no_save

    def get_next_run_id(exp_dir: str) -> str:
        existing = [
            d for d in os.listdir(exp_dir)
            if d.isdigit() and len(d) == 3
        ]
        if not existing:
            return "001"
        return f"{int(max(existing)) + 1:03d}"

    BASE_PATH = os.getcwd()
    RUNS_ROOT = os.path.join(BASE_PATH, "runs")
    os.makedirs(RUNS_ROOT, exist_ok=True)

    EXP_NAME = args.exp_name
    EXP_DIR = os.path.join(RUNS_ROOT, EXP_NAME)
    os.makedirs(EXP_DIR, exist_ok=True)

    RUN_ID = get_next_run_id(EXP_DIR)
    RUN_DIR = os.path.join(EXP_DIR, RUN_ID)
    print("RUN_ID:", RUN_ID)

    if SAVE:
        os.makedirs(RUN_DIR, exist_ok=True)
        FIG_DIR = os.path.join(RUN_DIR, "figures")
        WEIGHTS_DIR = os.path.join(RUN_DIR, "weights")
        LOGS_DIR = os.path.join(RUN_DIR, "logs")

        os.makedirs(FIG_DIR, exist_ok=True)
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
    else:
        FIG_DIR = WEIGHTS_DIR = LOGS_DIR = None

    job_name = os.environ.get("SLURM_JOB_NAME")
    job_id = os.environ.get("SLURM_JOB_ID")
    if SAVE and job_name and job_id:
        out_src = os.path.join(RUNS_ROOT, f"{job_name}_{job_id}.out")
        err_src = os.path.join(RUNS_ROOT, f"{job_name}_{job_id}.err")
        out_dst = os.path.join(RUN_DIR, f"{job_name}_{job_id}.out")
        err_dst = os.path.join(RUN_DIR, f"{job_name}_{job_id}.err")

        for src, dst in [(out_src, out_dst), (err_src, err_dst)]:
            if os.path.exists(src):
                try:
                    shutil.move(src, dst)
                    print(f"Moved {src} -> {dst}")
                except Exception as e:
                    print(f"Could not move {src} to {dst}: {e}")

    img_size = args.img_size
    print(f"Image size: {img_size}")

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])

    full_data = datasets.ImageFolder(
        root=CELEBA_ROOT,
        transform=transform,
    )
    print("CelebA root:", CELEBA_ROOT)
    print("Nb d'images :", len(full_data))

    dataloader = DataLoader(
        full_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    batch_size = args.batch_size
    N_EPOCH = args.epochs
    EVAL_EVERY = args.eval_every
    lr = args.lr
    grad_clip = args.grad_clip

    sigma_min = args.sigma_min
    sigma_max = args.sigma_max

    print(f"Sigma hyperparams: min={sigma_min}, max={sigma_max}")

    in_ch = 3
    base_ch = args.base_ch
    channel_mults = tuple(parse_int_list(args.channel_mults))
    sigma_emb_dim = args.sigma_emb_dim

    print(f"Model: base_ch={base_ch}, channel_mults={channel_mults}")
    print(f"Grad clip max_norm: {grad_clip}")

    model = m.SmallUNetSigma(
        in_ch=in_ch,
        base_ch=base_ch,
        channel_mults=channel_mults,
        emb_dim=sigma_emb_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

    ema_decay = args.ema_decay
    if ema_decay is not None and ema_decay > 0.0:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"EMA activée avec decay={ema_decay}")
    else:
        ema_state = None
        print("EMA désactivée (ema_decay <= 0).")

    # Config de sauvegarde par epoch
    save_interval = max(1, args.save_epoch_interval)
    save_start_frac = args.save_epoch_start_frac
    save_start_epoch = max(1, int(np.ceil(save_start_frac * N_EPOCH)))
    print(
        f"Sauvegarde epoch: tous les {save_interval} epochs "
        f"à partir de l'epoch {save_start_epoch} (sur {N_EPOCH})"
    )

    if SAVE:
        hparams = {
            "batch_size": batch_size,
            "N_epoch": N_EPOCH,
            "EVAL_EVERY": EVAL_EVERY,
            "lr": lr,
            "grad_clip": grad_clip,
            "sigma": {
                "min": sigma_min,
                "max": sigma_max, 
            },
            "ema_decay": ema_decay,
            "save_epoch_interval": save_interval,
            "save_epoch_start_frac": save_start_frac,
            "device": str(device),
            "model": {
                "in_channel": in_ch,
                "base_ch": base_ch,
                "channel_mults": list(channel_mults),
                "sigma_emb_dim": sigma_emb_dim,
            },
            "celeba_root": CELEBA_ROOT,
            "img_size": img_size,
            "slurm": {
                "job_name": job_name,
                "job_id": job_id,
            },
        }
        with open(os.path.join(LOGS_DIR, "hparams.json"), "w") as f:
            json.dump(hparams, f, indent=4)

    total = sum(p.numel() for p in model.parameters())
    print(f"{total/1e6:.2f} M params")

    model.train()
    L = []
    eval_steps = []
    step = 0

    for epoch in range(N_EPOCH):
        for u, (x, _) in enumerate(dataloader):
            step += 1
            x = x.to(device)

            optimizer.zero_grad()
            loss = LossVE_SDE(model, x, sigma_min, sigma_max)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            if ema_state is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        ema_param = ema_state[name]
                        ema_param.mul_(ema_decay).add_(param.data, alpha=1.0 - ema_decay)

            L.append(loss.item())

            if (step) % EVAL_EVERY == 0:
                model.eval()
                eval_steps.append(step + 1)
                print(
                    f"[step : {step} | epoch : {epoch}] "
                    f"train loss (last {EVAL_EVERY}) = "
                    f"{np.mean(L[-EVAL_EVERY:]):.4f}"
                )
                if SAVE:
                    np.save(os.path.join(LOGS_DIR, "train_loss.npy"), np.array(L))
                    np.save(os.path.join(LOGS_DIR, "steps_eval.npy"), np.array(eval_steps))
                model.train()

        # ---- fin d'epoch ----
        epoch_idx = epoch + 1

        # 1) Toujours sauvegarder model.pt (overwrite)
        if SAVE:
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "model.pt"))
            if ema_state is not None:
                torch.save(ema_state, os.path.join(WEIGHTS_DIR, "model_ema.pt"))

        # 2) Sauvegarde conditionnelle de checkpoints supplémentaires
        if SAVE and epoch_idx >= save_start_epoch:
            if (epoch_idx - save_start_epoch) % save_interval == 0:
                ckpt_name = f"model_epoch{epoch_idx:03d}.pt"
                torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, ckpt_name))
                if ema_state is not None:
                    ckpt_name_ema = f"model_ema_epoch{epoch_idx:03d}.pt"
                    torch.save(ema_state, os.path.join(WEIGHTS_DIR, ckpt_name_ema))

    # Final save
    if SAVE:
        np.save(os.path.join(LOGS_DIR, "train_loss.npy"), np.array(L))
        np.save(os.path.join(LOGS_DIR, "steps_eval.npy"),
                np.array(eval_steps))
        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "model_final.pt"))
        if ema_state is not None:
            torch.save(ema_state, os.path.join(WEIGHTS_DIR, "model_ema_final.pt"))


if __name__ == "__main__":
    args = get_args()
    main(args)
