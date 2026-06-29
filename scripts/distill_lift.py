#!/usr/bin/env python3
"""Distill canonical tray lift pose from demo tray_lift_start frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from interaction_retarget.constants import TASK_ID, default_sidecar_dir
from interaction_retarget.grasp.distill_lift import distill_canonical_tray_lift, save_canonical_tray_lift


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--exclude-fallback", action="store_true")
    p.add_argument("--max-episodes", type=int, default=20, help="Cap episodes (default 20, zarr read only)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sidecar_dir = args.sidecar_dir or default_sidecar_dir(TASK_ID)
    out_dir = args.out_dir or sidecar_dir
    proto = distill_canonical_tray_lift(
        sidecar_dir,
        exclude_fallback=args.exclude_fallback,
        max_episodes=args.max_episodes,
    )
    out_path = save_canonical_tray_lift(proto, out_dir / "canonical_tray_lift.npz")
    r = proto.report
    print(
        f"tray lift canonical from {r.num_episodes_used} eps "
        f"(excluded {len(r.excluded_episode_indices)}); "
        f"mocap_std={r.mocap_pos_std_m*1e3:.1f}mm tray_dz={r.tray_z_delta_median_m*1e3:.1f}mm"
    )
    print(f"Wrote -> {out_path}")


if __name__ == "__main__":
    main()
