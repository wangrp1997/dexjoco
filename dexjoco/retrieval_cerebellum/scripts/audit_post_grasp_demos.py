"""Build a manifest of post-grasp-to-insert demonstration segments."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from retrieval_cerebellum.demo_segments import DemoSegmentationConfig, audit_label_files


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/post_grasp_demo_audit"),
    )
    parser.add_argument(
        "--insert-label-dir",
        type=Path,
        default=None,
        help="Defaults to force_labels_20260812_current_replay/episodes when available.",
    )
    parser.add_argument("--grasp-confirm-frames", type=int, default=5)
    parser.add_argument("--grasp-loss-confirm-frames", type=int, default=8)
    parser.add_argument("--min-segment-frames", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DemoSegmentationConfig(
        grasp_confirm_frames=args.grasp_confirm_frames,
        grasp_loss_confirm_frames=args.grasp_loss_confirm_frames,
        min_segment_frames=args.min_segment_frames,
    )
    label_dir = args.dataset_root / "dexquery_labels" / "episodes"
    insert_label_dir = args.insert_label_dir
    if insert_label_dir is None:
        candidate = (
            args.dataset_root
            / "force_labels_20260812_current_replay"
            / "episodes"
        )
        insert_label_dir = candidate if candidate.is_dir() else None
    label_paths = sorted(label_dir.glob("episode_*.parquet"))
    if not label_paths:
        raise FileNotFoundError(f"No episode label parquet files under {label_dir}")

    segments = audit_label_files(
        label_paths,
        config,
        insert_label_dir=insert_label_dir,
    )
    eligible = [segment for segment in segments if segment.eligible]
    reasons = Counter(
        segment.rejection_reason for segment in segments if not segment.eligible
    )
    lengths = np.asarray([segment.num_frames for segment in eligible], dtype=np.int64)
    summary = {
        "dataset_root": str(args.dataset_root),
        "grasp_label_dir": str(label_dir),
        "insert_label_dir": None if insert_label_dir is None else str(insert_label_dir),
        "total_episodes": len(segments),
        "eligible_episodes": len(eligible),
        "eligible_frames": int(lengths.sum()) if lengths.size else 0,
        "segment_frames_min": int(lengths.min()) if lengths.size else None,
        "segment_frames_median": float(np.median(lengths)) if lengths.size else None,
        "segment_frames_max": int(lengths.max()) if lengths.size else None,
        "rejection_reasons": dict(sorted(reasons.items())),
        "segments_with_confirmed_grasp_gap": sum(
            not segment.grasp_retained_to_insert
            for segment in eligible
        ),
        "config": {
            "grasp_confirm_frames": config.grasp_confirm_frames,
            "grasp_loss_confirm_frames": config.grasp_loss_confirm_frames,
            "min_segment_frames": config.min_segment_frames,
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "segments.jsonl").open("w", encoding="utf-8") as file:
        for segment in segments:
            file.write(json.dumps(segment.to_dict(), ensure_ascii=False) + "\n")
    with (args.output / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
