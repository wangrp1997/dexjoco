#!/usr/bin/env python3
"""Slice 44-dim insert LeRobot dataset to right-arm-only 22-dim actions (symlink videos)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RIGHT_SLICES = slice(0, 22)


def _slice_stats(stats: dict, src_root: Path, dst_root: Path) -> dict:
    out = json.loads(json.dumps(stats))
    if "action" in out:
        for key in ("min", "max", "mean", "std", "count"):
            if key in out["action"] and isinstance(out["action"][key], list):
                out["action"][key] = out["action"][key][:22]
    return out


def _rewrite_parquet(src: Path, dst: Path) -> None:
    table = pq.read_table(src)
    actions = table.column("action").to_pylist()
    sliced = [np.asarray(row, dtype=np.float32)[RIGHT_SLICES].tolist() for row in actions]
    action_type = pa.list_(pa.float32(), 22)
    new_action = pa.array(sliced, type=action_type)
    idx = table.column_names.index("action")
    cols = list(table.columns)
    cols[idx] = new_action
    pq.write_table(pa.table(cols, names=table.column_names), dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/datasets/bimanual_assembly_insert_force_lerobot"),
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/datasets/bimanual_assembly_insert_force_lerobot_right22"),
    )
    parser.add_argument("--repo-id", default="bimanual_assembly_insert_force_lerobot_right22")
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    if dst.exists():
        raise FileExistsError(f"destination exists: {dst}")
    if not (src / "meta/info.json").is_file():
        raise FileNotFoundError(f"missing meta/info.json under {src}")

    dst.mkdir(parents=True)
    meta_src = src / "meta"
    meta_dst = dst / "meta"
    meta_dst.mkdir()

    info = json.loads((meta_src / "info.json").read_text(encoding="utf-8"))
    info["features"]["action"]["shape"] = [22]
    (meta_dst / "info.json").write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    for name in ("tasks.parquet",):
        shutil.copy2(meta_src / name, meta_dst / name)

    ep_src = meta_src / "episodes"
    ep_dst = meta_dst / "episodes"
    shutil.copytree(ep_src, ep_dst)

    stats_path = meta_src / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        (meta_dst / "stats.json").write_text(
            json.dumps(_slice_stats(stats, src, dst), indent=4) + "\n",
            encoding="utf-8",
        )

    for extra in ("force_smoke_summary.json",):
        p = meta_src / extra
        if p.is_file():
            shutil.copy2(p, meta_dst / extra)

    os.symlink(src / "videos", dst / "videos", target_is_directory=True)

    data_src = src / "data"
    data_dst = dst / "data"
    data_dst.mkdir()
    for src_file in sorted(data_src.rglob("*.parquet")):
        rel = src_file.relative_to(data_src)
        out = data_dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        _rewrite_parquet(src_file, out)
        print(f"slice {rel}", flush=True)

    readme = dst / "README.md"
    readme.write_text(
        f"# {args.repo_id}\n\n"
        f"Sliced from `{src.name}`: action `[0:22]` right arm only; obs unchanged (46 + 3 cams).\n",
        encoding="utf-8",
    )
    print(f"done -> {dst} ({info.get('total_episodes')} ep, {info.get('total_frames')} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
