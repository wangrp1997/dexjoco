#!/usr/bin/env python3
"""Export privileged sim insert poses for PoseInsert training (Phase B-1)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEXJOJO_ROOT = _REPO_ROOT / "dexjoco"
if str(_DEXJOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXJOJO_ROOT))

from interaction_retarget.constants import TASK_ID, default_sidecar_dir
from pose_insert.export import ExportSkip, export_episode, export_manifest
from pose_insert.paths import default_poseinsert_data_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar-dir", type=Path, default=None)
    p.add_argument("--ep", type=int, default=35, help="Single episode index (default: 35)")
    p.add_argument("--all", action="store_true", help="Export every manifest episode with peg timing")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"default: {default_poseinsert_data_dir(TASK_ID)}",
    )
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--seed", type=int, default=0, help="env.reset seed for privileged replay")
    p.add_argument(
        "--success-mode",
        type=str,
        default="xy",
        choices=["xy", "tip"],
        help="xy: success if min |peg-in-socket xy| <= threshold (default). "
        "tip: legacy tip-distance / insert_ok contact checks.",
    )
    p.add_argument(
        "--max-success-xy-mm",
        type=float,
        default=10.0,
        help="xy-mode: reject if min |rel xy| exceeds this (default: 10)",
    )
    p.add_argument(
        "--require-insert-ok",
        action="store_true",
        help="tip-mode only: require insert_ok contact in replay",
    )
    p.add_argument(
        "--include-failed",
        action="store_true",
        help="Deprecated alias: tip-mode without require_insert_ok",
    )
    p.add_argument(
        "--max-last-tip-dist-mm",
        type=float,
        default=20.0,
        help="tip-mode: reject if segment end tip distance exceeds this (default: 20)",
    )
    p.add_argument(
        "--max-min-tip-dist-mm",
        type=float,
        default=15.0,
        help="tip-mode: reject if closest tip distance exceeds this (default: 15)",
    )
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    return p.parse_args()


def _manifest_entry(sidecar_dir: Path, episode_index: int) -> dict:
    manifest = json.loads((sidecar_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["episodes"]:
        if int(entry["episode_index"]) == int(episode_index):
            return entry
    raise KeyError(f"episode {episode_index} not in manifest")


def main() -> int:
    args = _parse_args()
    sidecar_dir = args.sidecar_dir if args.sidecar_dir is not None else default_sidecar_dir(TASK_ID)
    output_dir = args.output_dir if args.output_dir is not None else default_poseinsert_data_dir(TASK_ID)
    manifest_path = sidecar_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    require_insert_ok = bool(args.require_insert_ok) and not bool(args.include_failed)
    if args.success_mode == "tip" and not args.require_insert_ok and not args.include_failed:
        # Preserve old tip-mode default: require contact unless --include-failed.
        require_insert_ok = True

    export_kwargs = {
        "success_mode": args.success_mode,
        "max_success_xy_mm": args.max_success_xy_mm,
        "require_insert_ok": require_insert_ok,
        "max_last_tip_dist_mm": args.max_last_tip_dist_mm,
        "max_min_tip_dist_mm": args.max_min_tip_dist_mm,
    }

    if args.all:
        reports, skipped = export_manifest(
            manifest_path,
            output_dir,
            split=args.split,
            seed=args.seed,
            show_progress=not args.no_progress,
            **export_kwargs,
        )
        print(
            f"done: exported={len(reports)} skipped={len(skipped)} "
            f"mode={args.success_mode} -> {output_dir}",
            flush=True,
        )
        return 0 if not skipped else 0

    entry = _manifest_entry(sidecar_dir, args.ep)
    result = export_episode(
        entry,
        output_dir,
        split=args.split,
        seed=args.seed,
        **export_kwargs,
    )
    if isinstance(result, ExportSkip):
        print(
            f"skip ep{result.episode_index}: {result.reason}\n"
            f"  min_xy={result.min_rel_xy_mm:.1f}mm "
            f"last_xy={result.last_rel_xy_mm:.1f}mm "
            f"last_z={result.last_rel_z_mm:.1f}mm "
            f"insert_ok={result.has_insert_ok}",
            flush=True,
        )
        return 2

    print(
        f"ok ep{result.episode_index} -> {result.output_dir}\n"
        f"  frames={result.num_frames} "
        f"segment=[{result.segment.start_frame},{result.segment.end_frame}] "
        f"peg_lift_end={result.segment.peg_lift_end_frame} "
        f"approach={result.segment.first_approach_frame}\n"
        f"  xy_mm: min={result.min_rel_xy_mm:.1f} end={result.last_rel_xy_mm:.1f} "
        f"end_z={result.last_rel_z_mm:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
