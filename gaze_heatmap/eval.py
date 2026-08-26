#!/usr/bin/env python3
"""Eval heatmap checkpoint: peak pixel error + overlay PNGs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gaze_heatmap.dataset import GazeSpiralDataset, discover_episodes, load_index
from gaze_heatmap.train import split_episodes
from gaze_heatmap.unet import ResNetUNet
from gaze_heatmap.utils import draw_points, heatmap_peak


def preprocess_rgb(rgb: np.ndarray, image_size: int) -> torch.Tensor:
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return torch.from_numpy(x.transpose(2, 0, 1))


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/datasets/gaze_spiral_ego_100"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    image_size = int(meta.get("image_size", 224))
    sigma = float(meta.get("sigma", 5.0))
    seed = 0

    episode_dirs = discover_episodes(args.data_root)
    _, val_eps = split_episodes(episode_dirs, val_ratio=0.1, seed=seed)
    val_records = load_index(args.data_root, val_eps)
    val_ds = GazeSpiralDataset(val_records, image_size=image_size, sigma=sigma, train=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ResNetUNet(n_class=2).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hole_errs: list[float] = []
    tip_errs: list[float] = []

    n = min(len(val_ds), args.max_samples)
    for i in range(n):
        rec = val_records[i]
        bgr = cv2.imread(str(rec.episode_dir / rec.image_rel))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = rgb.shape[:2]
        sx = image_size / w0
        sy = image_size / h0

        x = preprocess_rgb(rgb, image_size).unsqueeze(0).to(device)
        pred = model(x)[0].cpu().numpy()
        pred = np.clip(pred, 0.0, 1.0)
        hole_xy = heatmap_peak(pred[0])
        tip_xy = heatmap_peak(pred[1])

        gt_hole = (rec.hole_u * sx, rec.hole_v * sy)
        gt_tip = (rec.tip_u * sx, rec.tip_v * sy)
        if rec.hole_in_frame and rec.hole_visible:
            hole_errs.append(float(np.hypot(hole_xy[0] - gt_hole[0], hole_xy[1] - gt_hole[1])))
        if rec.tip_in_frame and rec.tip_visible:
            tip_errs.append(float(np.hypot(tip_xy[0] - gt_tip[0], tip_xy[1] - gt_tip[1])))

        vis = cv2.resize(rgb, (image_size, image_size))
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        draw_points(vis_bgr, [gt_hole], color=(0, 255, 0))
        draw_points(vis_bgr, [gt_tip], color=(0, 0, 255))
        draw_points(vis_bgr, [hole_xy], color=(0, 180, 180))
        draw_points(vis_bgr, [tip_xy], color=(180, 0, 180))
        out_name = f"{rec.episode_dir.name}_f{rec.frame:05d}.png"
        cv2.imwrite(str(args.out_dir / out_name), vis_bgr)

    summary = {
        "ckpt": str(args.ckpt),
        "val_episodes": [p.name for p in val_eps],
        "samples": n,
        "hole_px_mean": float(np.mean(hole_errs)) if hole_errs else None,
        "tip_px_mean": float(np.mean(tip_errs)) if tip_errs else None,
        "hole_px_median": float(np.median(hole_errs)) if hole_errs else None,
        "tip_px_median": float(np.median(tip_errs)) if tip_errs else None,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
