#!/usr/bin/env python3
"""Open-loop bimanual grasp validation: IK → contact repair → hold check (phase-1 step 4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEXJOJO_ROOT = _REPO_ROOT / "dexjoco"
if str(_DEXJOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXJOJO_ROOT))

import numpy as np

from interaction_retarget.constants import TASK_ID, default_sidecar_dir
from interaction_retarget.grasp.distill import load_canonical_grasp
from interaction_retarget.grasp.ik import solve_from_canonical_npz, warm_start_from_demo
from interaction_retarget.grasp.repair import (
    laplacian_rmse,
    merge_ik_with_warm_start,
    ramp_grasp_sequential,
    repair_and_verify,
)
from interaction_retarget.io.zarr_io import discover_zarr_demos, load_zarr_episode
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.replay import make_assembly_env, raw_flat_to_dict
from interaction_retarget.sim.settle import read_arm_action


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sidecar-dir",
        type=Path,
        default=None,
        help=f"Dir with canonical_*_grasp.npz (default: {default_sidecar_dir(TASK_ID)})",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--warm-start-ep", type=int, default=79, help="Demo ep for grasp replay (default: 79)")
    p.add_argument(
        "--zarr-root",
        type=Path,
        default=Path("/mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly"),
    )
    p.add_argument("--maxiter", type=int, default=60)
    p.add_argument("--hold-steps", type=int, default=12)
    p.add_argument("--max-repair-iters", type=int, default=24)
    p.add_argument(
        "--apply-ik",
        action="store_true",
        help="Execute blended IK grasp (default: demo grasp pose at grasp frame)",
    )
    p.add_argument("--finger-alpha", type=float, default=0.25, help="IK finger mix when --apply-ik")
    p.add_argument("--arm-alpha", type=float, default=0.0, help="IK mocap mix when --apply-ik")
    p.add_argument("--ramp-steps", type=int, default=3)
    return p.parse_args()


def _grasp_frame_index(sidecar_dir: Path, episode_index: int) -> int:
    meta_path = sidecar_dir / f"episode_{episode_index:03d}" / "meta.json"
    timing = {}
    if meta_path.is_file():
        timing = json.loads(meta_path.read_text(encoding="utf-8")).get("timing", {})
    left_frame = int(timing.get("left_grasp_frame", 0))
    right_frame = int(timing.get("right_grasp_frame", 0))
    return max(left_frame, right_frame)


def _replay_to_grasp_frame(
    env,
    *,
    sidecar_dir: Path,
    zarr_root: Path,
    episode_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay demo to grasp frame; return stored zarr actions (not mocap readback)."""
    demos = discover_zarr_demos(zarr_root)
    zarr_path = demos[episode_index]
    grasp_frame = _grasp_frame_index(sidecar_dir, episode_index)
    actions, _, _ = load_zarr_episode(zarr_path)
    warm_start_from_demo(env, zarr_path=zarr_path, grasp_frame=grasp_frame)
    stored = raw_flat_to_dict(actions[grasp_frame])
    return np.asarray(stored["right"], dtype=np.float64), np.asarray(stored["left"], dtype=np.float64)


def _warm_start_for_ik(
    env,
    *,
    sidecar_dir: Path,
    zarr_root: Path,
    episode_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    demos = discover_zarr_demos(zarr_root)
    grasp_frame = _grasp_frame_index(sidecar_dir, episode_index)
    return warm_start_from_demo(
        env,
        zarr_path=demos[episode_index],
        grasp_frame=grasp_frame,
    )


def main() -> None:
    args = _parse_args()
    sidecar_dir = args.sidecar_dir if args.sidecar_dir is not None else default_sidecar_dir(TASK_ID)
    tray_npz = sidecar_dir / "canonical_tray_grasp.npz"
    peg_npz = sidecar_dir / "canonical_peg_grasp.npz"
    for path in (tray_npz, peg_npz):
        if not path.is_file():
            raise FileNotFoundError(path)

    canonical_tray = load_canonical_grasp(tray_npz)
    canonical_peg = load_canonical_grasp(peg_npz)

    env = make_assembly_env(seed=args.seed, randomize=args.randomize)
    raw = env.unwrapped
    detector = AssemblyContactDetector(raw)
    try:
        if args.warm_start_ep is None:
            raise ValueError("--warm-start-ep is required for open-loop grasp validation")

        warm_right, warm_left = _warm_start_for_ik(
            env,
            sidecar_dir=sidecar_dir,
            zarr_root=args.zarr_root,
            episode_index=int(args.warm_start_ep),
        )
        detector.reset_reference(raw)

        tray_ik = solve_from_canonical_npz(
            raw,
            tray_npz,
            object_name="tray",
            maxiter=args.maxiter,
            hold_right=warm_right,
            hold_left=warm_left,
            initial_active=warm_left,
        )
        peg_ik = solve_from_canonical_npz(
            raw,
            peg_npz,
            object_name="peg",
            maxiter=args.maxiter,
            hold_right=warm_right,
            hold_left=tray_ik.action_left,
            initial_active=warm_right,
        )

        grasp_right, grasp_left = _replay_to_grasp_frame(
            env,
            sidecar_dir=sidecar_dir,
            zarr_root=args.zarr_root,
            episode_index=int(args.warm_start_ep),
        )
        detector.reset_reference(raw)

        if args.apply_ik:
            grasp_right, grasp_left = merge_ik_with_warm_start(
                tray_ik,
                peg_ik,
                warm_right=grasp_right,
                warm_left=grasp_left,
                finger_alpha=args.finger_alpha,
                arm_alpha=args.arm_alpha,
            )
            ramp_grasp_sequential(
                raw,
                home_right=read_arm_action(raw, "right"),
                home_left=read_arm_action(raw, "left"),
                action_right=grasp_right,
                action_left=grasp_left,
                steps=args.ramp_steps,
            )

        repair = repair_and_verify(
            raw,
            action_right=grasp_right,
            action_left=grasp_left,
            detector=detector,
            canonical_tray=canonical_tray,
            canonical_peg=canonical_peg,
            hold_steps=args.hold_steps,
            max_repair_iters=args.max_repair_iters,
        )

        lap_tray = laplacian_rmse(raw, canonical_tray, object_name="tray")
        lap_peg = laplacian_rmse(raw, canonical_peg, object_name="peg")
        grasp_src = "ik_blend" if args.apply_ik else "demo_grasp"

        print(
            f"IK tray success={tray_ik.success} peg success={peg_ik.success} "
            f"lap_tray={lap_tray*1e3:.1f}mm lap_peg={lap_peg*1e3:.1f}mm grasp={grasp_src}"
        )
        print(
            f"repair_iters={repair.repair_iters} hold_steps={repair.hold_steps} "
            f"tray_contact={repair.tray.contact_count} peg_contact={repair.peg.contact_count} "
            f"stable_tray={repair.stable_tray} stable_peg={repair.stable_peg} "
            f"success={repair.success}"
        )
        if not repair.success:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
