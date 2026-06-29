#!/usr/bin/env python3
"""QA: peg grasp frame on-table rate before/after timing fix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from interaction_retarget.constants import PEG_ON_TABLE_MARGIN_M, TASK_ID, default_sidecar_dir
from interaction_retarget.grasp.distill import _peg_z_delta_at_grasp_m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar-dir", type=Path, default=default_sidecar_dir(TASK_ID))
    p.add_argument("--max-z-mm", type=float, default=PEG_ON_TABLE_MARGIN_M * 1e3)
    args = p.parse_args()
    manifest = json.loads((args.sidecar_dir / "manifest.json").read_text())
    ok = 0
    bad: list[tuple[int, float]] = []
    for entry in manifest["episodes"]:
        if not entry.get("has_peg"):
            continue
        ep = int(entry["episode_index"])
        dz = _peg_z_delta_at_grasp_m(entry)
        if dz is None:
            bad.append((ep, float("nan")))
            continue
        if dz * 1e3 <= args.max_z_mm:
            ok += 1
        else:
            bad.append((ep, dz))
    n = ok + len(bad)
    print(f"peg on-table @ grasp: {ok}/{n} (<= {args.max_z_mm:.1f}mm)")
    for ep, dz in bad[:12]:
        print(f"  ep{ep:03d} dz={dz*1e3:.1f}mm frame={manifest['episodes'][ep]['timing'].get('right_grasp_frame')}")
    if len(bad) > 12:
        print(f"  ... +{len(bad) - 12} more")


if __name__ == "__main__":
    main()
