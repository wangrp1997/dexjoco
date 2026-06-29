#!/usr/bin/env python3
"""Extract tray/peg lift-end targets from one demo episode."""

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
from interaction_retarget.grasp.lift_reference import (
    DEMO_LIFT_REFERENCE_NAME,
    extract_demo_lift_reference,
    save_demo_lift_reference,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--episode", type=int, default=None, help="Fixed episode index (default: random)")
    p.add_argument("--pick-seed", type=int, default=0, help="RNG seed when picking a random episode")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sidecar_dir = args.sidecar_dir or default_sidecar_dir(TASK_ID)
    out_dir = args.out_dir or sidecar_dir
    ref = extract_demo_lift_reference(
        sidecar_dir,
        episode_index=args.episode,
        pick_seed=int(args.pick_seed),
    )
    out_path = save_demo_lift_reference(ref, out_dir / DEMO_LIFT_REFERENCE_NAME)
    f = ref.frames
    print(
        f"ep={ref.episode_index} "
        f"tray wp={ref.tray.mocap_pos_obj.shape[0]} frames={ref.tray.num_demo_frames} "
        f"peg wp={ref.peg.mocap_pos_obj.shape[0]} frames={ref.peg.num_demo_frames}"
    )
    print(f"Wrote -> {out_path}")


if __name__ == "__main__":
    main()
