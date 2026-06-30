#!/usr/bin/env python3
"""Backfill action44.npy for exported PoseInsert train demos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEXJOJO_ROOT = _REPO_ROOT / "dexjoco"
if str(_DEXJOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXJOJO_ROOT))

from interaction_retarget.io.zarr_io import load_zarr_episode
from pose_insert.paths import default_poseinsert_data_dir
from pose_insert.wrist_actions import zarr_flat_to_action44
from interaction_retarget.constants import TASK_ID


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--split", type=str, default="train")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    data_root = args.data_root if args.data_root is not None else default_poseinsert_data_dir(TASK_ID)
    split_dir = data_root / args.split
    if not split_dir.is_dir():
        print(f"missing {split_dir}", file=sys.stderr)
        return 1

    n_ok = 0
    for demo_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        meta_path = demo_dir / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        seg = meta.get("segment", {})
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
        actions, _, _ = load_zarr_episode(Path(meta["zarr_path"]))
        sl = slice(start, end + 1)
        zarr_seg = np.asarray(actions[sl], dtype=np.float64)
        action44 = np.stack([zarr_flat_to_action44(a) for a in zarr_seg], axis=0)
        np.save(demo_dir / "action44.npy", action44)
        n_ok += 1
        print(f"ok {demo_dir.name} frames={action44.shape[0]}", flush=True)

    print(f"backfilled {n_ok} demos -> {split_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
