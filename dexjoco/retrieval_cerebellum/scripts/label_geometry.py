"""Replay eligible demos and export post-grasp geometry-prior sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from dexquery.data.lerobot_io import iter_episode_actions
from dexquery.data.zarr_io import discover_zarr_demos, load_zarr_episode
from retrieval_cerebellum.demo_segments import PostGraspSegment
from retrieval_cerebellum.geometry_replay import replay_episode_geometry
from retrieval_cerebellum.geometry_store import VECTOR_COLUMNS, write_geometry_episode


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
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_eligible_segments(path: Path) -> dict[int, PostGraspSegment]:
    segments: dict[int, PostGraspSegment] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            segment = PostGraspSegment(**json.loads(line))
            if segment.eligible:
                segments[segment.episode_index] = segment
    if not segments:
        raise ValueError(f"No eligible segments in {path}")
    return segments


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.dataset_root / "retrieval_cerebellum_geometry"
    segments = _load_eligible_segments(args.segments_manifest)
    if args.episodes is not None:
        requested = set(args.episodes)
        segments = {episode: segment for episode, segment in segments.items() if episode in requested}
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

    zarr_paths = discover_zarr_demos(args.zarr_input_dir)
    lerobot = {
        episode: (global_index, frame_index, actions)
        for episode, global_index, frame_index, actions in iter_episode_actions(
            args.dataset_root,
            episode_indices=episode_ids,
        )
    }
    summaries: list[dict] = []
    for progress, episode in enumerate(episode_ids, start=1):
        if episode in completed:
            print(f"[{progress}/{len(episode_ids)}] skip episode {episode}: already written")
            continue
        if episode >= len(zarr_paths):
            raise IndexError(f"Episode {episode} has no matching zarr demo")
        segment = segments[episode]
        global_index, frame_index, lerobot_actions = lerobot[episode]
        zarr_actions, initial_state = load_zarr_episode(zarr_paths[episode])
        if initial_state is None:
            raise ValueError(f"Episode {episode} zarr has no initial_state")
        if zarr_actions.shape != lerobot_actions.shape:
            raise ValueError(
                f"Episode {episode} zarr actions {zarr_actions.shape} do not match "
                f"LeRobot actions {lerobot_actions.shape}"
            )

        print(
            f"[{progress}/{len(episode_ids)}] replay episode {episode} "
            f"full={len(zarr_actions)} segment={segment.start_frame}:{segment.end_frame}",
            flush=True,
        )
        frames, _ = replay_episode_geometry(
            zarr_actions,
            seed=args.seed_base + episode,
            initial_state=initial_state,
        )
        mask = (frame_index >= segment.start_frame) & (frame_index <= segment.end_frame)
        positions = np.flatnonzero(mask)
        if positions.size != segment.num_frames:
            raise ValueError(
                f"Episode {episode} manifest expects {segment.num_frames} frames, "
                f"selected {positions.size}"
            )
        selected_frames = [frames[int(position)] for position in positions]
        path = write_geometry_episode(
            output_dir,
            episode,
            global_index=global_index[mask],
            frame_index=frame_index[mask],
            frames=selected_frames,
        )
        summary = {
            "episode_index": episode,
            "num_frames": len(selected_frames),
            "start_frame": int(frame_index[mask][0]),
            "end_frame": int(frame_index[mask][-1]),
            "insert_frames": int(sum(frame.insert_ok for frame in selected_frames)),
            "lateral_error_start_m": float(selected_frames[0].lateral_error_m),
            "lateral_error_end_m": float(selected_frames[-1].lateral_error_m),
            "axis_error_start_rad": float(selected_frames[0].axis_error_rad),
            "axis_error_end_rad": float(selected_frames[-1].axis_error_rad),
            "path": str(path),
        }
        summaries.append(summary)
        completed.add(episode)
        _write_json(
            output_dir / "checkpoint.json",
            {
                "completed_episodes": sorted(completed),
                "dataset_root": str(args.dataset_root),
                "zarr_input_dir": str(args.zarr_input_dir),
                "segments_manifest": str(args.segments_manifest),
            },
        )

    manifest = {
        "dataset_root": str(args.dataset_root),
        "zarr_input_dir": str(args.zarr_input_dir),
        "segments_manifest": str(args.segments_manifest),
        "num_completed_episodes": len(completed),
        "columns": [
            "index",
            "episode_index",
            "frame_index",
            "family_id",
            *VECTOR_COLUMNS,
            "lateral_error_m",
            "axis_error_rad",
            "approach_height_m",
            "insertion_depth_m",
            "target_depth_m",
            "nominal_peg_size_m",
            "peg_ok",
            "tray_ok",
            "insert_ok",
            "peg_contact_count",
            "tray_contact_count",
        ],
        "new_episode_summaries": summaries,
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
