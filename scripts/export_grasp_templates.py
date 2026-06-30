#!/usr/bin/env python3
"""Export object-frame grasp templates from sidecar demos."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from skill_graph.io.manifest import iter_graspable_episodes, load_manifest
from skill_graph.paths import sidecar_manifest, template_bank_dir
from skill_graph.skills.templates.export import export_all_from_manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=sidecar_manifest())
    p.add_argument("--out", type=Path, default=template_bank_dir())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    entries = iter_graspable_episodes(manifest)
    templates = export_all_from_manifest(
        entries,
        seed=int(args.seed),
        bank_root=args.out,
        max_episodes=args.max_episodes,
    )
    print(f"exported {len(templates)} templates -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
