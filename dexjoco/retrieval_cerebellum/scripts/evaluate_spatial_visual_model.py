"""Evaluate a frozen RGB spatial visual checkpoint on an untouched split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from retrieval_cerebellum.scripts.train_spatial_visual_model import (
    _run_split,
    _sidecar_paths,
)
from retrieval_cerebellum.spatial_visual_learning import build_spatial_visual_model


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--supervision-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--heatmap-sigma-px", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_spatial_visual_model(
        base_channels=int(checkpoint["base_channels"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    paths = _sidecar_paths(args.supervision_dir, args.split, None)
    if not paths:
        raise ValueError(f"no {args.split} sidecars found")
    with torch.no_grad():
        losses, metrics = _run_split(
            model,
            paths,
            dataset_root=args.dataset_root,
            input_size=int(checkpoint["input_size"]),
            batch_size=args.batch_size,
            device=device,
            sigma_px=args.heatmap_sigma_px,
        )
    payload = {
        "stage": "V2 frozen RGB spatial visual evaluation",
        "split": args.split,
        "episodes": [int(path.stem.rsplit("_", 1)[-1]) for path in paths],
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "losses": losses,
        "metrics": metrics,
        "mask_downsample": int(checkpoint["mask_downsample"]),
        "uses_rgb_only_at_inference": True,
        "uses_teacher_geometry_at_inference": False,
        "approved_for_multiview_fusion": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
