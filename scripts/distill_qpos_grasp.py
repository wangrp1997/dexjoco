#!/usr/bin/env python3
"""Distill canonical qpos + MuJoCo contact targets from demo grasp frames (phase-1 qpos path)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from interaction_retarget.constants import TASK_ID, default_sidecar_dir
from interaction_retarget.grasp.qpos_distill import distill_qpos_from_sidecar_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sidecar-dir",
        type=Path,
        default=None,
        help=f"manifest + episode_* (default: {default_sidecar_dir(TASK_ID)})",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--exclude-fallback", action="store_true")
    p.add_argument("--unified-rep-ep", type=int, default=None, help="Same demo ep for tray+peg qpos/contacts")
    p.add_argument("--no-filter-peg-off-table", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sidecar_dir = args.sidecar_dir if args.sidecar_dir is not None else default_sidecar_dir(TASK_ID)
    protos = distill_qpos_from_sidecar_dir(
        sidecar_dir,
        out_dir=args.out_dir,
        exclude_fallback=args.exclude_fallback,
        filter_peg_off_table=not args.no_filter_peg_off_table,
        unified_rep_episode=args.unified_rep_ep,
    )
    out_dir = args.out_dir if args.out_dir is not None else sidecar_dir
    for name, proto in protos.items():
        print(
            f"[{name}] qpos prototype rep=ep{proto.representative_episode_index} "
            f"contacts={proto.contact_targets.count} from {len(proto.source_episode_indices)} eps"
        )
    print(f"Wrote -> {out_dir}/canonical_tray_qpos_grasp.npz")
    print(f"Wrote -> {out_dir}/canonical_peg_qpos_grasp.npz")


if __name__ == "__main__":
    main()
