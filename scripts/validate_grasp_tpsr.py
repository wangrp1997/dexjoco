#!/usr/bin/env python3
"""Phase-1: random tray grasp → lift → hold (default). No insert.

Do not use --fast for real validation: IK budget is too low and contact often stays 0.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEXJOJO_ROOT = _REPO_ROOT / "dexjoco"
if str(_DEXJOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXJOJO_ROOT))

from interaction_retarget.constants import MIN_GRASP_CONTACT_COUNT, TASK_ID, default_sidecar_dir
from interaction_retarget.grasp.pipeline_tpsr import run_random_grasp_tpsr


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hold-steps", type=int, default=20)
    p.add_argument("--hold-warmup-steps", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument(
        "--stage",
        choices=("tray_lift", "bimanual_grasp"),
        default="bimanual_grasp",
    )
    p.add_argument("--skip-tray-lift", action="store_true")
    p.add_argument("--skip-peg-lift", action="store_true", help="Skip right peg lift (bimanual stage)")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--tray-hold-max-steps", type=int, default=72)
    p.add_argument("--tray-lift-height-m", type=float, default=0.05)
    return p.parse_args()


def _print_report(report, *, stage: str) -> None:
    r = report.repair
    verdict = "PASS" if report.success else "FAIL"
    print(
        f"seed={report.seed} stage={stage} {verdict} "
        f"tray={r.tray.contact_count} peg={r.peg.contact_count} reason={report.fail_reason}"
    )
    print(
        f"  left  ik: ok={report.tray_ik.success} contact={report.tray_ik.contact_count} "
        f"lap={report.tray_ik.laplacian_rmse_m*1e3:.1f}mm"
    )
    if stage == "bimanual_grasp":
        print(
            f"  right peg: contact={int(report.peg_reach.get('contact_count', -1))} "
            f"lap={report.peg_laplacian_rmse_m*1e3:.1f}mm"
            if not np.isnan(report.peg_laplacian_rmse_m)
            else "  right peg: skipped"
        )
    if report.tray_lift_hold_stable is not None:
        print(f"  tray_lift_hold={report.tray_lift_hold_stable}")
    if report.bench_tray is not None:
        bt = report.bench_tray
        print(
            f"  bench_tray: stable={bt.stable} contact={bt.min_contact} "
            f"pose={bt.pose_ok} hole={bt.hole_ok}"
        )
    if report.bench_peg is not None:
        bp = report.bench_peg
        print(
            f"  bench_peg: stable={bp.stable} contact={bp.min_contact} "
            f"pose={bp.pose_ok} hole={bp.hole_ok}"
        )
    gl = report.tray_grasp_lift
    if gl is not None:
        print(
            f"  L0: grasp={gl.grasp_ok} lift={gl.lift_height_ok} hold={gl.hold_ok} "
            f"success={gl.success}"
        )
    if stage == "bimanual_grasp" and not report.success:
        need = MIN_GRASP_CONTACT_COUNT
        if r.peg.contact_count < need:
            print(f"  >> peg need contact>={need}, got {r.peg.contact_count}")


def main() -> None:
    args = _parse_args()
    sidecar_dir = args.sidecar_dir if args.sidecar_dir is not None else default_sidecar_dir(TASK_ID)
    seeds = args.seeds if args.seeds else [int(args.seed)]
    failed = 0
    passed = 0
    for seed in seeds:
        t0 = time.perf_counter()
        report = run_random_grasp_tpsr(
            sidecar_dir=sidecar_dir,
            seed=int(seed),
            hold_steps=args.hold_steps,
            hold_warmup_steps=args.hold_warmup_steps,
            skip_tray_lift=args.skip_tray_lift,
            skip_peg_lift=args.skip_peg_lift if args.stage == "bimanual_grasp" else True,
            tray_hold_max_steps=args.tray_hold_max_steps,
            tray_lift_height_m=args.tray_lift_height_m,
            fast=args.fast,
            stage=args.stage,
        )
        elapsed = time.perf_counter() - t0
        print(f"--- elapsed={elapsed:.1f}s ---")
        _print_report(report, stage=args.stage)
        if report.success:
            passed += 1
        else:
            failed += 1
    print(f"=== {passed}/{len(seeds)} passed stage={args.stage} ===")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
