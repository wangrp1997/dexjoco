#!/usr/bin/env python3
"""Allowlist-constrained rollback for pilot_micro_demo_v0.

Refuses paths outside embodied_grasp_insertion/data/pilot_micro_demo_v0/.
Does not encourage bare rm -rf.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEXJOCO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DEXJOCO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEXJOCO_ROOT))

from embodied_grasp_insertion.pilot import ALLOWED_OUT_ROOT  # noqa: E402
from embodied_grasp_insertion.pilot.paths import (  # noqa: E402
    PilotPathError,
    assert_under_allowlisted_out_root,
    safe_delete_under_pilot_root,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default=str(ALLOWED_OUT_ROOT),
        help="Path under allowlisted pilot root to delete",
    )
    parser.add_argument("--yes", action="store_true", help="Required to actually delete")
    args = parser.parse_args()
    try:
        target = assert_under_allowlisted_out_root(args.target)
    except PilotPathError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 2
    if not args.yes:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "refusing delete without --yes",
                    "would_delete": str(target),
                    "allowlisted_root": str(ALLOWED_OUT_ROOT),
                },
                ensure_ascii=False,
            )
        )
        return 3
    if not target.exists():
        print(json.dumps({"ok": True, "deleted": None, "note": "already absent"}))
        return 0
    safe_delete_under_pilot_root(target)
    print(json.dumps({"ok": True, "deleted": str(target)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
