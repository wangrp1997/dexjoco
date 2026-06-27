#!/usr/bin/env python3
"""Diagnose reach chain: home→target, site_err, hand_rmse, contact (per arm)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEXJOJO_ROOT = _REPO_ROOT / "dexjoco"
if str(_DEXJOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXJOJO_ROOT))

from interaction_retarget.constants import TASK_ID, default_sidecar_dir
from interaction_retarget.grasp.agent import make_peg_agent, make_tray_agent
from interaction_retarget.grasp.distill import load_canonical_grasp
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.replay import make_assembly_env
from interaction_retarget.sim.settle import read_arm_action, vec_to_arm_action


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sidecar-dir", type=Path, default=None)
    p.add_argument("--skip-tray-lift", action="store_true")
    return p.parse_args()


def _print_reach(label: str, info: dict) -> None:
    print(
        f"  {label}: steps={info.get('reach_steps')} "
        f"site={info.get('site_err_m', 0)*1e3:.1f}mm "
        f"hand={info.get('hand_rmse_m', 0)*1e3:.1f}mm "
        f"lap={info.get('laplacian_rmse_m', 0)*1e3:.1f}mm "
        f"contact={info.get('contact_count')} "
        f"lap_ok={info.get('lap_converged')} hand_ok={info.get('hand_converged')}"
    )


def main() -> None:
    args = _parse_args()
    sidecar = args.sidecar_dir or default_sidecar_dir(TASK_ID)
    tray_c = load_canonical_grasp(sidecar / "canonical_tray_grasp.npz")
    peg_c = load_canonical_grasp(sidecar / "canonical_peg_grasp.npz")
    left = make_tray_agent(tray_c)
    right = make_peg_agent(peg_c)

    env = make_assembly_env(seed=int(args.seed), randomize=False)
    raw = env.unwrapped
    det = AssemblyContactDetector(raw)
    env.reset()
    det.reset_reference(raw)

    hr, hl = read_arm_action(raw, "right"), read_arm_action(raw, "left")
    print(f"seed={args.seed}")

    lt, tray_ik = left.plan(raw, hold_right=hr, hold_left=hl)
    print(f"  tray_ik: success={tray_ik.success} lap={tray_ik.laplacian_rmse_m*1e3:.1f}mm hand={tray_ik.hand_rmse_m*1e3:.1f}mm")
    hr, hl, _, tray_info = left.execute(
        raw, target23=lt, hold_right=hr, hold_left=hl, detector=det, ik=tray_ik
    )
    _print_reach("tray", tray_info)

    if not args.skip_tray_lift:
        from interaction_retarget.grasp.lift import execute_tray_lift

        hl = execute_tray_lift(raw, grasp_left=hl, hold_right=hr, lift_height_m=0.05, steps=20)

    hr, hl = read_arm_action(raw, "right"), read_arm_action(raw, "left")
    rt, peg_ik = right.plan(raw, hold_right=hr, hold_left=hl, restore_env=False, detector=det)
    print(f"  peg_ik: success={peg_ik.success} lap={peg_ik.laplacian_rmse_m*1e3:.1f}mm hand={peg_ik.hand_rmse_m*1e3:.1f}mm")
    hr, hl, _, peg_info = right.execute(
        raw, target23=rt, hold_right=hr, hold_left=hl, detector=det, ik=peg_ik, skip_approach=True
    )
    _print_reach("peg", peg_info)

    c = det.compute(raw)
    print(
        f"  final: tray_contact={c.tray_contact_count} peg_contact={c.peg_contact_count}"
    )
    env.close()


if __name__ == "__main__":
    main()
