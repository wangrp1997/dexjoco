#!/usr/bin/env python3
"""Smoke one regrasp video.

Modes (short names):
  drop        — mid-lift slow release, then SE(3) regrasp
  grasp_fail  — underclose grasp/lift (object stays), then SE(3) regrasp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO), str(_REPO / "dexjoco")]

from dexjoco_datagen.paths import DEFAULT_MANIFEST, video_dir
from dexjoco_datagen.regrasp_pipeline import (
    load_manifest,
    normalize_mode,
    pick_entry,
    run_perturb_regrasp_one,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("drop", "grasp_fail", "drop_mid_lift", "lift_no_follow"),
        default="drop",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/mnt/ssd/datasets/dexjoco_gendata"),
    )
    args = parser.parse_args()
    mode = normalize_mode(args.mode)

    os.environ.setdefault("MUJOCO_GL", "egl")

    manifest = load_manifest(args.manifest)
    entry = pick_entry(manifest, args.episode)
    ep = int(entry["episode_index"])
    out_dir = video_dir("bimanual_assembly", args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"ep{ep:03d}_{mode}.mp4"

    print(f"episode={ep} mode={mode} zarr={entry['zarr_path']}", flush=True)
    print(f"timing={entry['timing']}", flush=True)
    print(f"video -> {video_path}", flush=True)

    result = run_perturb_regrasp_one(
        entry, video_path=video_path, seed=args.seed, mode=mode
    )
    meta_path = out_dir / f"ep{ep:03d}_{mode}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "episode_index": result.episode_index,
                "success": result.success,
                "mode": result.mode,
                "peg_z_after_fail": result.peg_z_after_fail,
                "peg_z_final": result.peg_z_final,
                "video_path": result.video_path,
                "message": result.message,
                "diagnostics": result.diagnostics,
                "zarr_path": entry["zarr_path"],
                "timing": entry["timing"],
            },
            f,
            indent=2,
        )
    print(result.message, flush=True)
    print(f"meta -> {meta_path}", flush=True)
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
