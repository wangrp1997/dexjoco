"""Build P3B-B episode-start three-camera CLIP initialization features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from retrieval_cerebellum.visual_initialization import (
    DEFAULT_CAMERA_KEYS,
    ClipVisionEncoder,
    TrainOnlyPCA,
    decode_video_frame,
    load_episode_video_references,
)


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--estimation-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/visual_initialization/clip_vit_b16.npz"),
    )
    parser.add_argument("--model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--pca-dim", type=int, default=4)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def _episode_sources(estimation_dir: Path) -> list[tuple[int, str, int]]:
    import pyarrow.parquet as parquet

    sources = []
    for path in sorted((estimation_dir / "episodes").glob("episode_*.parquet")):
        table = parquet.read_table(path, columns=["split", "frame_index"])
        splits = set(str(value) for value in table["split"].to_pylist())
        if len(splits) != 1:
            raise ValueError(f"{path} has non-constant split")
        episode_index = int(path.stem.rsplit("_", 1)[-1])
        frame_index = int(table["frame_index"][0].as_py())
        sources.append((episode_index, next(iter(splits)), frame_index))
    if not sources:
        raise FileNotFoundError(f"No estimation shards under {estimation_dir}")
    return sources


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    estimation_dir = args.estimation_dir or args.dataset_root / "retrieval_cerebellum_estimation"
    sources = _episode_sources(estimation_dir)
    encoder = ClipVisionEncoder(
        args.model,
        device=args.device,
        local_files_only=not args.allow_download,
    )

    records = []
    images = []
    hidden_size = int(encoder.model.config.hidden_size)
    raw_embeddings = np.empty(
        (len(sources), len(DEFAULT_CAMERA_KEYS), hidden_size),
        dtype=np.float32,
    )
    for source_row, (episode_index, split, frame_index) in enumerate(sources):
        references = load_episode_video_references(
            args.dataset_root,
            episode_index,
            frame_index,
        )
        for camera_row, reference in enumerate(references):
            records.append((source_row, camera_row))
            images.append(decode_video_frame(reference))
            if len(images) >= args.batch_size:
                encoded = encoder.encode(images)
                for index, (episode_row, camera_index) in enumerate(records):
                    raw_embeddings[episode_row, camera_index] = encoded[index]
                records.clear()
                images.clear()
        print(
            f"[{source_row + 1}/{len(sources)}] episode={episode_index} "
            f"split={split} frame={frame_index}",
            flush=True,
        )
    if images:
        encoded = encoder.encode(images)
        for index, (episode_row, camera_index) in enumerate(records):
            raw_embeddings[episode_row, camera_index] = encoded[index]

    flattened = raw_embeddings.reshape(len(sources), -1)
    split = np.asarray([source[1] for source in sources], dtype="U16")
    train_mask = split == "train"
    if train_mask.sum() < 2:
        raise ValueError("visual PCA requires at least two train episodes")
    pca = TrainOnlyPCA.fit(flattened[train_mask], args.pca_dim)
    projected = pca.transform(flattened)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        episode_index=np.asarray([source[0] for source in sources], dtype=np.int64),
        split=split,
        source_frame_index=np.asarray([source[2] for source in sources], dtype=np.int64),
        camera_keys=np.asarray(DEFAULT_CAMERA_KEYS, dtype="U64"),
        raw_embeddings=raw_embeddings,
        pca_mean=pca.mean,
        pca_components=pca.components,
        projected_features=projected,
        model_name=np.asarray(args.model),
    )
    summary = {
        "output": str(args.output),
        "dataset_root": str(args.dataset_root),
        "estimation_dir": str(estimation_dir),
        "model": args.model,
        "camera_keys": list(DEFAULT_CAMERA_KEYS),
        "num_episodes": len(sources),
        "split_counts": {
            name: int(np.sum(split == name)) for name in ("train", "validation", "test")
        },
        "source": "first frame of each post-grasp estimation shard",
        "raw_feature_dim": int(flattened.shape[1]),
        "projected_feature_dim": int(projected.shape[1]),
        "pca_fit_split": "train only",
        "uses_teacher_geometry": False,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
