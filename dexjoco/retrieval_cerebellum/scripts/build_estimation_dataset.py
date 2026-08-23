"""Replay and export aligned sensor-only inputs with privileged teacher labels."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import numpy as np

from dexquery.data.lerobot_io import load_dataset_table
from dexquery.data.zarr_io import discover_zarr_demos, load_zarr_episode
from retrieval_cerebellum.demo_segments import PostGraspSegment
from retrieval_cerebellum.estimation_data import (
    METADATA_COLUMNS,
    SENSOR_VECTOR_DIMS,
    TEACHER_SCALAR_COLUMNS,
    TEACHER_VECTOR_DIMS,
    write_estimation_episode,
)
from retrieval_cerebellum.geometry_store import geometry_frames_to_columns
from retrieval_cerebellum.learning_data import episode_split, table_columns
from retrieval_cerebellum.sensor_replay import (
    replay_episode_sensors,
    replay_episode_sensors_and_geometry,
)


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")
DEFAULT_ZARR = Path("/mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--zarr-input-dir", type=Path, default=DEFAULT_ZARR)
    parser.add_argument(
        "--segments-manifest",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/post_grasp_demo_audit/segments.jsonl"),
    )
    parser.add_argument("--geometry-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--teacher-source",
        choices=("same-replay", "geometry-sidecar"),
        default="same-replay",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class _SegmentBounds:
    episode_index: int
    start_frame: int
    end_frame: int
    num_frames: int


def _load_segments(
    path: Path,
    geometry_dir: Path,
) -> dict[int, PostGraspSegment | _SegmentBounds]:
    segments: dict[int, PostGraspSegment | _SegmentBounds] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as file:
            for line in file:
                segment = PostGraspSegment(**json.loads(line))
                if segment.eligible:
                    segments[segment.episode_index] = segment
    else:
        import pyarrow.parquet as parquet

        for geometry_path in sorted((geometry_dir / "episodes").glob("episode_*.parquet")):
            table = parquet.read_table(
                geometry_path,
                columns=["episode_index", "frame_index"],
            )
            episode_values = set(int(value) for value in table["episode_index"].to_pylist())
            if len(episode_values) != 1:
                raise ValueError(f"{geometry_path} has episode ids {episode_values}")
            frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
            if frame_index.size == 0:
                raise ValueError(f"{geometry_path} contains no rows")
            episode_index = next(iter(episode_values))
            segments[episode_index] = _SegmentBounds(
                episode_index=episode_index,
                start_frame=int(frame_index.min()),
                end_frame=int(frame_index.max()),
                num_frames=int(frame_index.size),
            )
    if not segments:
        raise ValueError(
            f"No eligible segments in {path} and no geometry shards under {geometry_dir}"
        )
    return segments


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def _episode_rows(table, episode_index: int) -> dict[str, np.ndarray]:
    import pyarrow.compute as compute

    filtered = table.filter(compute.equal(table["episode_index"], episode_index))
    order = compute.sort_indices(filtered, sort_keys=[("frame_index", "ascending")])
    episode = compute.take(filtered, order)
    return {
        "index": np.asarray(episode["index"].to_numpy(), dtype=np.int64),
        "episode_index": np.asarray(
            episode["episode_index"].to_numpy(), dtype=np.int64
        ),
        "frame_index": np.asarray(episode["frame_index"].to_numpy(), dtype=np.int64),
        "timestamp": np.asarray(episode["timestamp"].to_numpy(), dtype=np.float32),
        "state": np.asarray(episode["observation.state"].to_pylist(), dtype=np.float32),
        "action": np.asarray(episode["action"].to_pylist(), dtype=np.float32),
    }


def main() -> None:
    args = parse_args()
    geometry_dir = args.geometry_dir or args.dataset_root / "retrieval_cerebellum_geometry"
    output_dir = args.output_dir or args.dataset_root / "retrieval_cerebellum_estimation"
    segments = _load_segments(args.segments_manifest, geometry_dir)
    if args.episodes is not None:
        requested = set(args.episodes)
        segments = {key: value for key, value in segments.items() if key in requested}
    episode_ids = sorted(segments)
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]
    if not episode_ids:
        raise ValueError("No eligible episodes selected")

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = {
        int(path.stem.rsplit("_", 1)[-1])
        for path in (output_dir / "episodes").glob("episode_*.parquet")
    }
    if not args.resume:
        completed.clear()

    source_table = load_dataset_table(
        args.dataset_root,
        columns=[
            "index",
            "episode_index",
            "frame_index",
            "timestamp",
            "observation.state",
            "action",
        ],
    )
    zarr_paths = discover_zarr_demos(args.zarr_input_dir)
    summaries: list[dict[str, object]] = []
    for progress, episode_index in enumerate(episode_ids, start=1):
        if episode_index in completed:
            print(
                f"[{progress}/{len(episode_ids)}] skip episode {episode_index}: already written",
                flush=True,
            )
            continue
        if episode_index >= len(zarr_paths):
            raise IndexError(f"Episode {episode_index} has no matching zarr demo")
        geometry_path = geometry_dir / "episodes" / f"episode_{episode_index:06d}.parquet"
        if not geometry_path.is_file():
            raise FileNotFoundError(f"Missing geometry teacher sidecar: {geometry_path}")

        segment = segments[episode_index]
        source = _episode_rows(source_table, episode_index)
        zarr_actions, initial_state = load_zarr_episode(zarr_paths[episode_index])
        if zarr_actions.shape != source["action"].shape:
            raise ValueError(
                f"Episode {episode_index} zarr actions {zarr_actions.shape} do not match "
                f"LeRobot actions {source['action'].shape}"
            )
        if not np.allclose(zarr_actions, source["action"], atol=1e-6, rtol=0.0):
            raise ValueError(f"Episode {episode_index} action values do not match zarr replay")

        print(
            f"[{progress}/{len(episode_ids)}] replay sensors episode {episode_index} "
            f"full={len(zarr_actions)} segment={segment.start_frame}:{segment.end_frame}",
            flush=True,
        )
        if args.teacher_source == "same-replay":
            observations, replay_geometry, replay_info = replay_episode_sensors_and_geometry(
                zarr_actions,
                source["state"],
                seed=args.seed_base + episode_index,
                initial_state=initial_state,
            )
        else:
            observations, replay_info = replay_episode_sensors(
                zarr_actions,
                source["state"],
                seed=args.seed_base + episode_index,
                initial_state=initial_state,
            )
            replay_geometry = None
        mask = (source["frame_index"] >= segment.start_frame) & (
            source["frame_index"] <= segment.end_frame
        )
        positions = np.flatnonzero(mask)
        if positions.size != segment.num_frames:
            raise ValueError(
                f"Episode {episode_index} manifest expects {segment.num_frames} frames, "
                f"selected {positions.size}"
            )
        selected_source = {
            name: values[mask]
            for name, values in source.items()
            if name not in {"state", "action"}
        }
        selected_observations = [observations[int(position)] for position in positions]
        if replay_geometry is None:
            geometry = table_columns(geometry_path)
        else:
            selected_geometry = [replay_geometry[int(position)] for position in positions]
            geometry = geometry_frames_to_columns(
                global_index=selected_source["index"],
                episode_index=episode_index,
                frame_index=selected_source["frame_index"],
                frames=selected_geometry,
            )
        family_ids = set(str(value) for value in geometry["family_id"].tolist())
        if len(family_ids) != 1:
            raise ValueError(f"Episode {episode_index} has family ids {family_ids}")
        family_id = next(iter(family_ids))
        split = episode_split(episode_index, family_id, seed=args.split_seed)
        path = write_estimation_episode(
            output_dir,
            episode_index,
            selected_source,
            selected_observations,
            geometry,
            split=split,
        )
        summary = {
            "episode_index": episode_index,
            "family_id": family_id,
            "split": split,
            "num_frames": int(positions.size),
            "start_frame": int(selected_source["frame_index"][0]),
            "end_frame": int(selected_source["frame_index"][-1]),
            "replay": replay_info,
            "teacher_source": args.teacher_source,
            "path": str(path),
        }
        summaries.append(summary)
        completed.add(episode_index)
        _write_json(
            output_dir / "checkpoint.json",
            {
                "completed_episodes": sorted(completed),
                "dataset_root": str(args.dataset_root),
                "zarr_input_dir": str(args.zarr_input_dir),
                "geometry_dir": str(geometry_dir),
                "segments_manifest": str(args.segments_manifest),
            },
        )

    import pyarrow.parquet as parquet

    episode_paths = sorted((output_dir / "episodes").glob("episode_*.parquet"))
    total_frames = sum(parquet.read_metadata(path).num_rows for path in episode_paths)
    previous_manifest_path = output_dir / "manifest.json"
    previous_same_replay: set[int] = set()
    if previous_manifest_path.is_file():
        previous_manifest = json.loads(previous_manifest_path.read_text())
        previous_same_replay.update(previous_manifest.get("same_replay_teacher_episodes", []))
    if args.teacher_source == "same-replay":
        previous_same_replay.update(summary["episode_index"] for summary in summaries)
    completed_episode_ids = {
        int(path.stem.rsplit("_", 1)[-1]) for path in episode_paths
    }
    manifest = {
        "schema_version": 1,
        "created_date": "2026-08-21",
        "dataset_root": str(args.dataset_root),
        "zarr_input_dir": str(args.zarr_input_dir),
        "geometry_dir": str(geometry_dir),
        "segments_manifest": str(args.segments_manifest),
        "num_completed_episodes": len(episode_paths),
        "num_frames": int(total_frames),
        "same_replay_teacher_episodes": sorted(previous_same_replay),
        "geometry_sidecar_teacher_episodes": sorted(
            completed_episode_ids - previous_same_replay
        ),
        "metadata_columns": list(METADATA_COLUMNS),
        "sensor_vector_dims": SENSOR_VECTOR_DIMS,
        "teacher_vector_dims": TEACHER_VECTOR_DIMS,
        "teacher_scalar_columns": list(TEACHER_SCALAR_COLUMNS),
        "temporal_semantics": (
            "Row t stores deployable sensors captured after executing demo action t; "
            "sensor_previous_action44 is that executed action."
        ),
        "image_source": {
            "dataset_videos": str(args.dataset_root / "videos"),
            "keys": [
                "observation.images.ego",
                "observation.images.wrist_left",
                "observation.images.wrist_right",
            ],
            "alignment": "episode_index + frame_index; pixels are not duplicated",
        },
        "hardware_mapping": {
            "sensor_state46": "dual wrist pose and finger joint encoders",
            "sensor_previous_action44": "controller command history",
            "sensor_arm_joint_torque": "MuJoCo actuator generalized force; map to robot joint torque",
            "sensor_fingertip_force_world": (
                "MuJoCo fingertip external-force proxy; replace with calibrated tactile estimate"
            ),
            "sensor_wrist_wrench_world": "MuJoCo wrist force/torque sensor; map to wrist F/T sensor",
        },
        "firewall": (
            "Online estimators must load only sensor_* columns via load_sensor_history; "
            "teacher_* columns are training/evaluation labels."
        ),
        "new_episode_summaries": summaries,
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
