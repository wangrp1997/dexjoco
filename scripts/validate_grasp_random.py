#!/usr/bin/env python3
"""Random object-pose bimanual grasp (phase-1 inference, no demo warm-start)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEXJOJO_ROOT = _REPO_ROOT / "dexjoco"
if str(_DEXJOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXJOJO_ROOT))

from interaction_retarget.constants import TASK_ID, default_sidecar_dir
from interaction_retarget.grasp.pipeline import run_random_grasp


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sidecar-dir",
        type=Path,
        default=None,
        help=f"Dir with canonical_*_grasp.npz (default: {default_sidecar_dir(TASK_ID)})",
    )
    p.add_argument("--seed", type=int, default=0, help="dexjoco env seed (peg/tray xy+yaw)")
    p.add_argument("--hold-steps", type=int, default=20)
    p.add_argument("--hold-warmup-steps", type=int, default=10)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Run multiple seeds (overrides --seed)",
    )
    p.add_argument(
        "--skip-tray-lift",
        action="store_true",
        help="Skip tray lift between left grasp and right peg grasp",
    )
    p.add_argument(
        "--peg-lift",
        action="store_true",
        help="Enable peg lift after right grasp",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fewer IK/approach steps (physics contact still required)",
    )
    p.add_argument(
        "--tray-hold-max-steps",
        type=int,
        default=72,
        help="Cap demo tray hold before peg grasp (fast mode caps at 16)",
    )
    p.add_argument("--tray-lift-height-m", type=float, default=0.05)
    p.add_argument("--tray-lift-steps", type=int, default=20)
    return p.parse_args()


def _print_reach(label: str, info: dict) -> None:
    if not info:
        return
    print(
        f"  reach_{label}: steps={info.get('reach_steps')} "
        f"site={info.get('site_err_m', 0)*1e3:.1f}mm "
        f"hand={info.get('hand_rmse_m', 0)*1e3:.1f}mm "
        f"contact={info.get('contact_count')} "
        f"lap_ok={info.get('lap_converged')} hand_ok={info.get('hand_converged')}"
    )


def _print_report(report) -> None:
    r = report.repair
    print(
        f"seed={report.seed} grasp=random "
        f"tray_contact={r.tray.contact_count} peg_contact={r.peg.contact_count} "
        f"success={r.success}"
    )
    print(
        f"  ik_tray: success={report.tray_ik.success} "
        f"lap={report.tray_ik.laplacian_rmse_m*1e3:.1f}mm hand={report.tray_ik.hand_rmse_m*1e3:.1f}mm "
        f"contact={report.tray_ik.contact_count} site={report.tray_ik.contact_site_rmse_m*1e3:.1f}mm"
    )
    print(
        f"  ik_peg: success={report.peg_ik.success} "
        f"lap={report.peg_ik.laplacian_rmse_m*1e3:.1f}mm hand={report.peg_ik.hand_rmse_m*1e3:.1f}mm "
        f"contact={report.peg_ik.contact_count} site={report.peg_ik.contact_site_rmse_m*1e3:.1f}mm"
    )
    _print_reach("tray", report.tray_reach)
    _print_reach("peg", report.peg_reach)
    if report.tray_lift_hold_stable is not None:
        print(f"  tray_lift_hold_stable={report.tray_lift_hold_stable}")
    print(
        f"  stable_tray={r.stable_tray} stable_peg={r.stable_peg} "
        f"repair_iters={r.repair_iters}"
    )


def main() -> None:
    args = _parse_args()
    sidecar_dir = args.sidecar_dir if args.sidecar_dir is not None else default_sidecar_dir(TASK_ID)
    seeds = args.seeds if args.seeds else [int(args.seed)]

    failed = 0
    for seed in seeds:
        t0 = time.perf_counter()
        report = run_random_grasp(
            sidecar_dir=sidecar_dir,
            seed=int(seed),
            hold_steps=args.hold_steps,
            hold_warmup_steps=args.hold_warmup_steps,
            skip_tray_lift=args.skip_tray_lift,
            skip_peg_lift=True if not getattr(args, "peg_lift", False) else False,
            tray_hold_max_steps=getattr(args, "tray_hold_max_steps", 72),
            tray_lift_height_m=args.tray_lift_height_m,
            tray_lift_steps=args.tray_lift_steps if args.tray_lift_steps != 20 else None,
            fast=getattr(args, "fast", False),
        )
        elapsed = time.perf_counter() - t0
        _print_report(report)
        print(f"  elapsed={elapsed:.1f}s")
        if not report.success:
            failed += 1

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
