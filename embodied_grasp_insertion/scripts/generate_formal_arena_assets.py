#!/usr/bin/env python3
"""Generate formal peg/socket/arena XMLs for all geometry families (P0-S0.2).

Default round_8mm official files are never overwritten.
No collection / no training.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embodied_grasp_insertion.geometry.family_spec import from_dict  # noqa: E402
from embodied_grasp_insertion.geometry.formal_xml_builder import (  # noqa: E402
    write_formal_family_assets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families-yaml",
        default=str(PROJECT_ROOT / "configs/geometry_families.yaml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--out-manifest",
        default=str(PROJECT_ROOT / "data/manifests/formal_arena_assets_v1.json"),
    )
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.families_yaml).read_text(encoding="utf-8"))
    specs = [from_dict(d) for d in raw["families"]]
    rows = []
    for spec in specs:
        row = write_formal_family_assets(spec, overwrite=args.overwrite)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    man = {
        "name": "formal_arena_assets_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": "P0-S0.2",
        "n_families": len(rows),
        "default_8mm_untouched": True,
        "families": rows,
        "allow_policy_training": False,
        "allow_full_collection": False,
    }
    Path(args.out_manifest).write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": args.out_manifest, "n": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
