#!/usr/bin/env python3
"""Skill replay: retrieve nearest demo → per-demo δ* grasp → lift blend → insert."""

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
from interaction_retarget.skill_replay.deploy import run_skill_replay


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--force-demo", type=int, default=None, help="Use this episode; restores its layout")
    p.add_argument(
        "--restore-demo-layout",
        action="store_true",
        help="Restore zarr initial_state of selected demo (auto with --force-demo)",
    )
    p.add_argument("--skip-insert", action="store_true", help="Stop after bimanual lift (tray+peg), no insert")
    p.add_argument("--skip-peg-lift", action="store_true", help="Skip right peg lift (debug only)")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--hold-steps", type=int, default=20)
    p.add_argument("--tray-hold-max-steps", type=int, default=72)
    return p.parse_args()


def _print_report(report) -> None:
    verdict = "PASS" if report.success else "FAIL"
    print(
        f"seed={report.seed} demo={report.demo_episode_index} "
        f"retrieval={report.retrieval_distance_m:.4f}m {verdict} "
        f"reason={report.fail_reason}"
    )
    print(
        f"  tray ik: contact={report.tray_ik.contact_count} "
        f"lap={report.tray_ik.laplacian_rmse_m*1e3:.1f}mm"
    )
    print(
        f"  peg ik: contact={report.peg_ik.contact_count} "
        f"lap={report.peg_laplacian_rmse_m*1e3:.1f}mm"
    )
    if report.tray_lift_hold_stable is not None:
        print(f"  tray_lift_hold={report.tray_lift_hold_stable}")
    ex = report.extra or {}
    if "tray_fc_ok" in ex:
        print(f"  tray_fc={ex['tray_fc_ok']} qp={ex.get('tray_qp_err', 0):.3f}")
    if "peg_fc_ok" in ex:
        print(f"  peg_fc={ex['peg_fc_ok']} qp={ex.get('peg_qp_err', 0):.3f}")
    tl = ex.get("tray_lift_track") or {}
    if tl.get("waypoint_rmse_m") is not None:
        print(f"  tray_lift_track rmse={tl['waypoint_rmse_m']*1e3:.1f}mm max={tl.get('max_waypoint_err_m', 0)*1e3:.1f}mm")
    pl = ex.get("peg_lift_track") or {}
    if pl.get("waypoint_rmse_m") is not None:
        print(f"  peg_lift_track rmse={pl['waypoint_rmse_m']*1e3:.1f}mm max={pl.get('max_waypoint_err_m', 0)*1e3:.1f}mm")
    if report.bench_tray is not None:
        bt = report.bench_tray
        print(f"  bench_tray: stable={bt.stable} contact={bt.min_contact} hole={bt.hole_ok}")
    if report.bench_peg is not None:
        bp = report.bench_peg
        print(f"  bench_peg: stable={bp.stable} contact={bp.min_contact} hole={bp.hole_ok}")
    ins = report.insert
    if ins is not None:
        print(
            f"  insert: ok={ins.success} handoff={ins.handoff} phase={ins.phase} "
            f"peg_lift={ins.peg_lift_m:.3f}m steps={ins.steps}"
        )


def main() -> None:
    args = _parse_args()
    sidecar_dir = args.sidecar_dir if args.sidecar_dir is not None else default_sidecar_dir(TASK_ID)
    seeds = args.seeds if args.seeds else [int(args.seed)]
    passed = failed = 0
    for seed in seeds:
        t0 = time.perf_counter()
        report = run_skill_replay(
            sidecar_dir=sidecar_dir,
            seed=int(seed),
            hold_steps=args.hold_steps,
            tray_hold_max_steps=args.tray_hold_max_steps,
            skip_insert=args.skip_insert,
            skip_peg_lift=args.skip_peg_lift,
            force_demo_episode=args.force_demo,
            restore_demo_layout=bool(args.restore_demo_layout or args.force_demo is not None),
            fast=args.fast,
        )
        elapsed = time.perf_counter() - t0
        print(f"--- elapsed={elapsed:.1f}s ---")
        _print_report(report)
        if report.success:
            passed += 1
        else:
            failed += 1
    stage = "lift_only" if args.skip_insert else "full+insert"
    print(f"=== {passed}/{len(seeds)} passed stage={stage} ===")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
