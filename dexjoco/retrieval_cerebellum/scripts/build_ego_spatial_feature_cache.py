"""Build an RGB-only ego spatial feature cache from a frozen visual model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from retrieval_cerebellum.ego_visual_state_estimation import EgoSpatialPredictor
from retrieval_cerebellum.visual_initialization import (
    DEFAULT_CAMERA_KEYS,
    decode_video_frames,
    load_episode_video_reference_series,
)


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--supervision-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _metadata(path: Path) -> tuple[int, str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return (
            int(data["episode_index"]),
            str(data["split"]),
            np.asarray(data["frame_index"], dtype=np.int64).copy(),
        )


def main() -> None:
    args = parse_args()
    predictor = EgoSpatialPredictor.load(args.checkpoint, device=args.device)
    paths = sorted((args.supervision_dir / "episodes").glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError("no spatial supervision episode metadata found")
    episode_values = []
    frame_values = []
    split_values = []
    feature_values = []
    reliability_values = []
    diagnostic_values: dict[str, list[np.ndarray]] = {}
    summaries = []
    for progress, path in enumerate(paths, start=1):
        episode_index, split, frame_index = _metadata(path)
        references = load_episode_video_reference_series(
            args.dataset_root,
            episode_index,
            frame_index,
            camera_keys=(DEFAULT_CAMERA_KEYS[0],),
        )
        images = decode_video_frames(references[DEFAULT_CAMERA_KEYS[0]])
        features, reliability, diagnostics = predictor.encode(
            images,
            batch_size=args.batch_size,
        )
        episode_values.append(np.full(len(frame_index), episode_index, dtype=np.int64))
        frame_values.append(frame_index)
        split_values.append(np.full(len(frame_index), split, dtype="U16"))
        feature_values.append(features)
        reliability_values.append(reliability)
        for name, values in diagnostics.items():
            diagnostic_values.setdefault(name, []).append(values)
        summaries.append(
            {
                "episode_index": episode_index,
                "split": split,
                "num_rows": len(frame_index),
                "reliability_mean": float(np.mean(reliability)),
                "reliability_p10": float(np.quantile(reliability, 0.1)),
            }
        )
        print(
            f"[{progress}/{len(paths)}] episode={episode_index} split={split} "
            f"rows={len(frame_index)}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        episode_index=np.concatenate(episode_values),
        frame_index=np.concatenate(frame_values),
        split=np.concatenate(split_values),
        projected_features=np.concatenate(feature_values),
        perceptual_reliability=np.concatenate(reliability_values),
        camera_keys=np.asarray([DEFAULT_CAMERA_KEYS[0]], dtype="U64"),
        model_name=np.asarray("frozen_spatial_visual_ego_v1"),
        checkpoint=np.asarray(str(args.checkpoint)),
        **{
            name: np.concatenate(values)
            for name, values in diagnostic_values.items()
        },
    )
    summary = {
        "stage": "V2 frozen ego RGB spatial feature cache",
        "output": str(args.output),
        "checkpoint": str(args.checkpoint),
        "num_episodes": len(paths),
        "num_rows": int(sum(item["num_rows"] for item in summaries)),
        "feature_dim": int(feature_values[0].shape[1]),
        "uses_rgb_only": True,
        "uses_teacher_geometry": False,
        "episodes": summaries,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
