"""Build frame-aligned multi-camera CLIP features for V2 visual tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from retrieval_cerebellum.visual_initialization import (
    DEFAULT_CAMERA_KEYS,
    ClipVisionEncoder,
    decode_video_frames,
    load_episode_video_reference_series,
)


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--estimation-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/retrieval_cerebellum/continuous_visual/clip_vit_b16_pca32.npz"
        ),
    )
    parser.add_argument("--model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--maximum-rows-per-episode", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def _episode_sources(
    estimation_dir: Path,
    *,
    requested: set[int] | None,
    frame_stride: int,
    maximum_rows: int | None,
) -> list[tuple[int, str, np.ndarray]]:
    import pyarrow.parquet as parquet

    sources = []
    for path in sorted((estimation_dir / "episodes").glob("episode_*.parquet")):
        episode_index = int(path.stem.rsplit("_", 1)[-1])
        if requested is not None and episode_index not in requested:
            continue
        table = parquet.read_table(path, columns=["split", "frame_index"])
        split = str(table["split"][0].as_py())
        frames = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)[::frame_stride]
        if maximum_rows is not None:
            frames = frames[:maximum_rows]
        if frames.size:
            sources.append((episode_index, split, frames))
    if not sources:
        raise FileNotFoundError("no continuous visual episode sources selected")
    return sources


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.frame_stride <= 0 or args.pca_dim <= 0:
        raise ValueError("batch-size, frame-stride and pca-dim must be positive")
    estimation_dir = args.estimation_dir or args.dataset_root / "retrieval_cerebellum_estimation"
    requested = None if args.episodes is None else set(args.episodes)
    sources = _episode_sources(
        estimation_dir,
        requested=requested,
        frame_stride=args.frame_stride,
        maximum_rows=args.maximum_rows_per_episode,
    )
    if args.max_episodes is not None:
        sources = sources[: args.max_episodes]
    encoder = ClipVisionEncoder(
        args.model,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    hidden_size = int(encoder.model.config.hidden_size)
    total_rows = sum(len(item[2]) for item in sources)
    raw = np.empty(
        (total_rows, len(DEFAULT_CAMERA_KEYS), hidden_size),
        dtype=np.float16,
    )
    episode_index = np.empty(total_rows, dtype=np.int64)
    frame_index = np.empty(total_rows, dtype=np.int64)
    split = np.empty(total_rows, dtype="U16")
    cursor = 0
    for source_row, (episode, episode_split, frames) in enumerate(sources):
        references = load_episode_video_reference_series(
            args.dataset_root,
            episode,
            frames,
        )
        camera_images = {
            key: decode_video_frames(references[key]) for key in DEFAULT_CAMERA_KEYS
        }
        for start in range(0, len(frames), args.batch_size):
            stop = min(start + args.batch_size, len(frames))
            images = []
            records = []
            for local_row in range(start, stop):
                for camera_row, key in enumerate(DEFAULT_CAMERA_KEYS):
                    images.append(camera_images[key][local_row])
                    records.append((local_row, camera_row))
            encoded = encoder.encode(images)
            for encoded_row, (local_row, camera_row) in enumerate(records):
                raw[cursor + local_row, camera_row] = encoded[encoded_row].astype(
                    np.float16
                )
        count = len(frames)
        episode_index[cursor : cursor + count] = episode
        frame_index[cursor : cursor + count] = frames
        split[cursor : cursor + count] = episode_split
        cursor += count
        print(
            f"[{source_row + 1}/{len(sources)}] episode={episode} "
            f"split={episode_split} rows={count}",
            flush=True,
        )
    flattened = raw.astype(np.float32).reshape(total_rows, -1)
    train_mask = split == "train"
    if int(train_mask.sum()) <= args.pca_dim:
        raise ValueError("selected cache needs more train rows than PCA dimensions")
    pca = PCA(
        n_components=args.pca_dim,
        svd_solver="randomized",
        random_state=0,
    ).fit(flattened[train_mask])
    projected = pca.transform(flattened).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        episode_index=episode_index,
        frame_index=frame_index,
        split=split,
        camera_keys=np.asarray(DEFAULT_CAMERA_KEYS, dtype="U64"),
        projected_features=projected,
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        model_name=np.asarray(args.model),
        frame_stride=np.asarray(args.frame_stride),
    )
    summary = {
        "stage": "V2 continuous multi-camera visual cache",
        "output": str(args.output),
        "model": args.model,
        "camera_keys": list(DEFAULT_CAMERA_KEYS),
        "frame_stride": args.frame_stride,
        "num_episodes": len(sources),
        "num_rows": total_rows,
        "split_rows": {
            name: int(np.sum(split == name))
            for name in ("train", "validation", "test")
        },
        "projected_feature_dim": int(projected.shape[1]),
        "pca_fit_split": "train only",
        "uses_teacher_geometry": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
