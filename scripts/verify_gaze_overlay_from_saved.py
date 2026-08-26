#!/usr/bin/env python3
"""Regenerate ego overlay from saved JPEG + labels.parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode-dir",
        type=Path,
        required=True,
        help="e.g. /mnt/hdd/dexjoco/datasets/gaze_spiral_ego_100/episode_14",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: outputs/gaze_spiral_verify_<episode>",
    )
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    ep_dir = args.episode_dir
    out = args.out_dir or Path(f"/mnt/hdd/dexjoco/outputs/gaze_spiral_verify_{ep_dir.name}")
    out.mkdir(parents=True, exist_ok=True)

    rows = pq.read_table(ep_dir / "labels.parquet").to_pylist()
    n = len(rows)
    picks = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    writer = imageio.get_writer(out / "ego_overlay_from_saved.mp4", fps=args.fps)

    for i, row in enumerate(rows):
        bgr = cv2.imread(str(ep_dir / row["image"]))
        vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tip = (int(round(row["tip_u"])), int(round(row["tip_v"])))
        hole = (int(round(row["hole_u"])), int(round(row["hole_v"])))
        if row["tip_in_frame"]:
            c = (0, 255, 0) if row["tip_visible"] else (0, 180, 255)
            cv2.drawMarker(vis, tip, c, cv2.MARKER_CROSS, 22, 2)
            cv2.circle(vis, tip, 10, c, 2)
        if row["hole_in_frame"]:
            c = (255, 0, 0) if row["hole_visible"] else (255, 180, 0)
            cv2.drawMarker(vis, hole, c, cv2.MARKER_TILTED_CROSS, 22, 2)
            cv2.circle(vis, hole, 10, c, 2)
        title = (
            f"saved {ep_dir.name} #{row['frame']} G=tip R=hole "
            f"tip_vis={int(row['tip_visible'])} hole_vis={int(row['hole_visible'])} "
            f"along={row['along_mm']:.0f}mm"
        )
        cv2.putText(vis, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        writer.append_data(vis)
        if i in picks:
            imageio.imwrite(out / f"overlay_{row['frame']:05d}.png", vis)
    writer.close()

    if (ep_dir / "meta.json").exists():
        (out / "source_meta.json").write_text((ep_dir / "meta.json").read_text(), encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "frames": n, "picks": picks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
