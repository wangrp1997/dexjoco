#!/usr/bin/env python3
"""Distill canonical grasp prototypes (δ*) from episode sidecars (phase-1 step 2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from interaction_retarget.constants import TASK_ID, default_sidecar_dir
from interaction_retarget.grasp.distill import distill_from_sidecar_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sidecar-dir",
        type=Path,
        default=None,
        help=f"Dir with manifest.json + episode_*/ (default: {default_sidecar_dir(TASK_ID)})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir for canonical_*.npz (default: same as --sidecar-dir)",
    )
    p.add_argument("--exclude-fallback", action="store_true", help="Skip episodes with tray/peg grasp_used_fallback in manifest")
    p.add_argument(
        "--no-filter-peg-off-table",
        action="store_true",
        help="Do not exclude peg grasp frames where peg z > rest + margin",
    )
    p.add_argument("--sample-seed", type=int, default=0, help="Seed for canonical object surface sampling")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sidecar_dir = args.sidecar_dir if args.sidecar_dir is not None else default_sidecar_dir(TASK_ID)
    prototypes = distill_from_sidecar_dir(
        sidecar_dir,
        out_dir=args.out_dir,
        exclude_fallback=args.exclude_fallback,
        filter_peg_off_table=not args.no_filter_peg_off_table,
        sample_seed=args.sample_seed,
    )
    for name, proto in prototypes.items():
        r = proto.report
        print(
            f"[{name}] canonical δ* from {r.num_episodes_used} eps "
            f"(excluded {len(r.excluded_episode_indices)}); "
            f"hand_std_mean={r.hand_points_std_mean_m*1e3:.2f}mm "
            f"laplacian_spread={r.laplacian_spread_mean_m*1e3:.2f}mm "
            f"contact_sites={proto.contact_sites_obj.shape[0]} "
            f"rep=ep{r.representative_episode_index}"
        )
    out_dir = args.out_dir if args.out_dir is not None else sidecar_dir
    print(f"Wrote -> {out_dir}/canonical_tray_grasp.npz")
    print(f"Wrote -> {out_dir}/canonical_peg_grasp.npz")
    print(f"Summary -> {out_dir}/canonical_grasp_summary.json")


if __name__ == "__main__":
    main()
