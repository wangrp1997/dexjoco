#!/usr/bin/env python3
"""Upload a local LeRobot v3 dataset directory to Hugging Face Hub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Local LeRobot dataset root (contains meta/, data/, videos/).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hub dataset repo id, e.g. DexJoCo/bimanual_assembly_insert_force_lerobot_mix854",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--no-videos", action="store_true", help="Skip videos/ upload")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {root}")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit("pip install lerobot") from exc

    print(f"[push] loading local dataset root={root} repo_id={args.repo_id}", flush=True)
    ds = LeRobotDataset(repo_id=args.repo_id, root=root)
    print(f"[push] uploading to https://huggingface.co/datasets/{args.repo_id}", flush=True)
    ds.push_to_hub(
        push_videos=not args.no_videos,
        upload_large_folder=True,
        private=args.private or None,
        tags=["lerobot", "dexjoco", "bimanual_assembly", "insert", "force"],
        license="apache-2.0",
    )
    print(f"[push] done: {args.repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
