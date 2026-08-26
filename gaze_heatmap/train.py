#!/usr/bin/env python3
"""Train ResNet18-UNet tip/hole heatmaps on DexJoCo gaze_spiral_ego dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gaze_heatmap.dataset import GazeSpiralDataset, discover_episodes, load_index
from gaze_heatmap.unet import ResNetUNet


def split_episodes(episode_dirs: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    eps = list(episode_dirs)
    rng = random.Random(seed)
    rng.shuffle(eps)
    n_val = max(1, int(round(len(eps) * val_ratio)))
    return eps[n_val:], eps[:n_val]


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    num = (diff * mask).sum()
    den = mask.sum().clamp(min=1.0)
    return num / den


def train_one_epoch(model, loader, opt, scheduler, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["heatmap"].to(device)
        m = batch["mask"].to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = masked_mse(pred, y, m)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["heatmap"].to(device)
        m = batch["mask"].to(device)
        pred = model(x)
        loss = masked_mse(pred, y, m)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/datasets/gaze_spiral_ego_100"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/outputs/gaze_heatmap_train"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np_seed = args.seed
    torch.manual_seed(np_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np_seed)

    episode_dirs = discover_episodes(args.data_root)
    train_eps, val_eps = split_episodes(episode_dirs, args.val_ratio, args.seed)
    train_records = load_index(args.data_root, train_eps)
    val_records = load_index(args.data_root, val_eps)

    train_ds = GazeSpiralDataset(
        train_records, image_size=args.image_size, sigma=args.sigma, train=True
    )
    val_ds = GazeSpiralDataset(val_records, image_size=args.image_size, sigma=args.sigma, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ResNetUNet(n_class=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    warmup = 100

    def lr_lambda(step: int) -> float:
        return min(step / warmup, 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    global_step = 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "data_root": str(args.data_root),
        "train_episodes": [p.name for p in train_eps],
        "val_episodes": [p.name for p in val_eps],
        "train_frames": len(train_records),
        "val_frames": len(val_records),
        "image_size": args.image_size,
        "sigma": args.sigma,
        "channels": "0=hole,1=tip",
    }
    (args.out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    best_val = float("inf")
    log_path = args.out_dir / "train.log"
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["heatmap"].to(device)
            m = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = masked_mse(pred, y, m)
            loss.backward()
            opt.step()
            scheduler.step()
            global_step += 1
            train_loss_sum += float(loss.item())
            train_batches += 1
        train_loss = train_loss_sum / max(train_batches, 1)
        val_loss = evaluate(model, val_loader, device)
        line = (
            f"epoch={epoch+1}/{args.epochs} train={train_loss:.5f} val={val_loss:.5f} "
            f"eps={len(episode_dirs)} frames={len(train_records)+len(val_records)} "
            f"elapsed={time.time()-t0:.1f}s"
        )
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "meta": meta,
        }
        torch.save(ckpt, args.out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, args.out_dir / "best.pt")
            print(f"  saved best val={best_val:.5f}", flush=True)

    print(json.dumps({"best_val": best_val, "out_dir": str(args.out_dir)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
