#!/usr/bin/env python3
"""Record skill_replay mp4 — dexquery parity: 30fps, max 50s, eef trajectory audit."""

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
from interaction_retarget.skill_replay.deploy import run_skill_replay
from interaction_retarget.skill_replay.trajectory_audit import EVAL_MAX_DURATION_S


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-demo", type=int, default=0)
    p.add_argument("--skip-insert", action="store_true", default=True)
    p.add_argument("--skip-peg-lift", action="store_true", help="Skip peg lift (debug only)")
    p.add_argument("--no-fast", action="store_true", help="Slow demo-length ramps (debug only)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Video output dir (default: outputs/skill_replay/videos)",
    )
    p.add_argument(
        "--allow-partial-video",
        action="store_true",
        help="Save video when tray lift ok even if peg lift fails (preview)",
    )
    p.add_argument(
        "--privileged-replay",
        action="store_true",
        help="L0: record full demo zarr replay instead of L1 algorithm playback",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sidecar_dir = args.sidecar_dir if args.sidecar_dir is not None else default_sidecar_dir(TASK_ID)
    out_dir = args.out_dir or (_REPO_ROOT / "outputs" / "skill_replay" / "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"demo{args.force_demo}_seed{args.seed}.mp4"

    report = run_skill_replay(
        sidecar_dir=sidecar_dir,
        seed=int(args.seed),
        skip_insert=bool(args.skip_insert),
        skip_peg_lift=bool(args.skip_peg_lift),
        force_demo_episode=int(args.force_demo),
        restore_demo_layout=True,
        fast=not args.no_fast,
        video_out=out,
        allow_partial_video=bool(args.allow_partial_video),
        l1_mode=not args.privileged_replay,
    )
    audit = (report.extra or {}).get("trajectory_audit")
    print(f"pass={report.success} reason={report.fail_reason}")
    print(f"demo={report.demo_episode_index} tray_hold={report.tray_lift_hold_stable}")
    print(f"mode={'L0_privileged' if args.privileged_replay else 'L1_planned'}")
    if not report.success and not args.allow_partial_video:
        print("DELIVERY FAIL: pipeline did not pass", file=sys.stderr)
        raise SystemExit(1)
    if audit is not None and not audit.ok_for_delivery() and not args.allow_partial_video:
        print("DELIVERY FAIL: video/trajectory audit did not pass", file=sys.stderr)
        raise SystemExit(1)
    if not out.exists():
        print("DELIVERY FAIL: video not saved", file=sys.stderr)
        raise SystemExit(1)
    print(f"PREVIEW: pass={report.success} (partial={args.allow_partial_video})")
    if out.exists():
        import subprocess

        dur = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(out),
            ],
            text=True,
        ).strip()
        print(f"saved {out} duration={float(dur):.1f}s (cap {EVAL_MAX_DURATION_S:.1f}s)")
        if float(dur) > EVAL_MAX_DURATION_S + 0.5:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
