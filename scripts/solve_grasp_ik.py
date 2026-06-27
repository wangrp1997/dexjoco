#!/usr/bin/env python3
"""Solve Laplacian grasp IK from canonical δ* (phase-1 step 3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from interaction_retarget.constants import TASK_ID, default_sidecar_dir
from interaction_retarget.io.zarr_io import discover_zarr_demos
from interaction_retarget.grasp.ik import solve_from_canonical_npz, warm_start_from_demo
from interaction_retarget.grasp.pre_grasp import derive_pre_grasp_from_grasp
from interaction_retarget.sim.replay import make_assembly_env


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sidecar-dir",
        type=Path,
        default=None,
        help=f"Dir with canonical_*_grasp.npz (default: {default_sidecar_dir(TASK_ID)})",
    )
    p.add_argument("--object", choices=("tray", "peg", "both"), default="both")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--randomize", action="store_true", help="Randomize object poses on reset")
    p.add_argument("--maxiter", type=int, default=60)
    p.add_argument("--warm-start-ep", type=int, default=None, help="Replay demo ep to grasp frame for IK init")
    p.add_argument(
        "--zarr-root",
        type=Path,
        default=Path("/mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly"),
    )
    p.add_argument("--pre-grasp", action="store_true", help="Print GenHand-style pre-grasp offset")
    return p.parse_args()


def _warm_start(
    env,
    *,
    sidecar_dir: Path,
    zarr_root: Path,
    episode_index: int,
    object_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    demos = discover_zarr_demos(zarr_root)
    meta_path = sidecar_dir / f"episode_{episode_index:03d}" / "meta.json"
    side = "left" if object_name == "tray" else "right"
    grasp_key = "left_grasp_frame" if side == "left" else "right_grasp_frame"
    grasp_frame = 0
    if meta_path.is_file():
        timing = json.loads(meta_path.read_text(encoding="utf-8")).get("timing", {})
        grasp_frame = int(timing.get(grasp_key, 0))
    warm_right, warm_left = warm_start_from_demo(
        env,
        zarr_path=demos[episode_index],
        grasp_frame=grasp_frame,
    )
    init_active = warm_left if side == "left" else warm_right
    return warm_right, warm_left, init_active


def main() -> None:
    args = _parse_args()
    sidecar_dir = args.sidecar_dir if args.sidecar_dir is not None else default_sidecar_dir(TASK_ID)
    objects = ("tray", "peg") if args.object == "both" else (args.object,)

    env = make_assembly_env(seed=args.seed, randomize=args.randomize)
    raw = env.unwrapped
    try:
        if args.warm_start_ep is None:
            env.reset()
        for object_name in objects:
            npz = sidecar_dir / f"canonical_{object_name}_grasp.npz"
            if not npz.is_file():
                print(f"Missing {npz}")
                continue
            warm_right = warm_left = init_active = None
            if args.warm_start_ep is not None:
                warm_right, warm_left, init_active = _warm_start(
                    env,
                    sidecar_dir=sidecar_dir,
                    zarr_root=args.zarr_root,
                    episode_index=int(args.warm_start_ep),
                    object_name=object_name,
                )
            result = solve_from_canonical_npz(
                raw,
                npz,
                object_name=object_name,  # type: ignore[arg-type]
                maxiter=args.maxiter,
                hold_right=warm_right,
                hold_left=warm_left,
                initial_active=init_active,
            )
            print(
                f"[{object_name}] side={result.active_side} success={result.success} "
                f"lap_rmse={result.laplacian_rmse_m*1e3:.2f}mm "
                f"hand_rmse={result.hand_rmse_m*1e3:.2f}mm cost={result.cost:.4f}"
            )
            if args.pre_grasp:
                active = result.action_left if result.active_side == "left" else result.action_right
                pre = derive_pre_grasp_from_grasp(active, side=result.active_side)  # type: ignore[arg-type]
                print(f"  pre-grasp offset (GenHand Allegro frame): {pre.offset_m}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
