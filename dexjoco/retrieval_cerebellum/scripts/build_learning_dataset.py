"""Build aligned post-grasp memory for retrieval and contact planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from retrieval_cerebellum.demo_segments import PostGraspSegment, load_state_action_segment
from retrieval_cerebellum.learning_data import (
    ACTION_DIM,
    GEOMETRY_FEATURE_DIM,
    RETRIEVAL_DESCRIPTOR_DIM,
    RetrievalEntry,
    RetrievalIndex,
    episode_split,
    geometry_feature_matrix,
    retrieval_descriptor,
    state46_to_action44,
    table_columns,
)


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--segments-manifest",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/post_grasp_demo_audit/segments.jsonl"),
    )
    parser.add_argument("--geometry-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_segments(path: Path) -> dict[int, PostGraspSegment]:
    segments: dict[int, PostGraspSegment] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            segment = PostGraspSegment(**json.loads(line))
            if segment.eligible:
                segments[segment.episode_index] = segment
    return segments


def _fixed_list(values: np.ndarray):
    import pyarrow as pa

    array = np.asarray(values, dtype=np.float32)
    flat = pa.array(array.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, array.shape[1])


def _write_episode(
    output_dir: Path,
    segment: PostGraspSegment,
    state_action: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    *,
    split: str,
) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    if not np.array_equal(state_action["index"], geometry["index"]):
        raise ValueError(f"episode {segment.episode_index} geometry index mismatch")
    if not np.array_equal(state_action["frame_index"], geometry["frame_index"]):
        raise ValueError(f"episode {segment.episode_index} geometry frame mismatch")

    state46 = state_action["state"]
    demo_action44 = state_action["action"]
    proprio_action44 = state46_to_action44(state46)
    geometry_features = geometry_feature_matrix(geometry)
    columns = {
        "index": state_action["index"],
        "episode_index": state_action["episode_index"],
        "frame_index": state_action["frame_index"],
        "split": [split] * len(state46),
        "family_id": geometry["family_id"].tolist(),
        "state46": _fixed_list(state46),
        "proprio_action44": _fixed_list(proprio_action44),
        "geometry_features": _fixed_list(geometry_features),
        "demo_action44": _fixed_list(demo_action44),
    }
    path = output_dir / "episodes" / f"episode_{segment.episode_index:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(pa.table(columns), path)
    return path


def main() -> None:
    args = parse_args()
    geometry_dir = args.geometry_dir or args.dataset_root / "retrieval_cerebellum_geometry"
    output_dir = args.output_dir or args.dataset_root / "retrieval_cerebellum_learning"
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = _load_segments(args.segments_manifest)
    if args.episodes is not None:
        requested = set(args.episodes)
        segments = {key: value for key, value in segments.items() if key in requested}

    prepared: list[tuple[PostGraspSegment, dict[str, np.ndarray], str, np.ndarray]] = []
    missing: list[int] = []
    for episode_index, segment in sorted(segments.items()):
        geometry_path = geometry_dir / "episodes" / f"episode_{episode_index:06d}.parquet"
        if not geometry_path.is_file():
            missing.append(episode_index)
            continue
        geometry = table_columns(geometry_path)
        family_ids = set(geometry["family_id"].tolist())
        if len(family_ids) != 1:
            raise ValueError(f"episode {episode_index} has family ids {family_ids}")
        family_id = str(next(iter(family_ids)))
        prepared.append(
            (segment, geometry, family_id, retrieval_descriptor(geometry))
        )
    if not prepared:
        raise FileNotFoundError(
            f"No geometry sidecars found under {geometry_dir}; run label_geometry first"
        )

    entries = [
        RetrievalEntry(
            episode_index=segment.episode_index,
            family_id=family_id,
            split=episode_split(segment.episode_index, family_id, seed=args.split_seed),
            descriptor=descriptor,
        )
        for segment, _, family_id, descriptor in prepared
    ]
    index = RetrievalIndex(entries)
    entry_by_episode = {entry.episode_index: entry for entry in entries}

    episodes: list[dict] = []
    for segment, geometry, family_id, descriptor in prepared:
        entry = entry_by_episode[segment.episode_index]
        state_action = load_state_action_segment(args.dataset_root, segment)
        path = _write_episode(
            output_dir,
            segment,
            state_action,
            geometry,
            split=entry.split,
        )
        neighbors = index.query(
            descriptor,
            family_id=family_id,
            top_k=args.top_k,
            exclude_episode=segment.episode_index,
        )
        episodes.append(
            {
                "episode_index": segment.episode_index,
                "family_id": family_id,
                "split": entry.split,
                "num_frames": segment.num_frames,
                "descriptor": descriptor.tolist(),
                "retrieved_train_episodes": [item.episode_index for item, _ in neighbors],
                "retrieval_distances": [distance for _, distance in neighbors],
                "path": str(path),
            }
        )

    manifest = {
        "dataset_root": str(args.dataset_root),
        "geometry_dir": str(geometry_dir),
        "segments_manifest": str(args.segments_manifest),
        "num_episodes": len(episodes),
        "num_frames": sum(item["num_frames"] for item in episodes),
        "missing_geometry_episodes": missing,
        "state_dim": 46,
        "action_dim": ACTION_DIM,
        "geometry_feature_dim": GEOMETRY_FEATURE_DIM,
        "retrieval_descriptor_dim": RETRIEVAL_DESCRIPTOR_DIM,
        "purpose": "retrieval memory and contact-planning supervision",
        "gallery": "train split only; same episode excluded",
        "episodes": episodes,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
