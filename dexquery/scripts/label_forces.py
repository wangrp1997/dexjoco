#!/usr/bin/env python3
"""Generate per-frame finger forces and wrist FT via MuJoCo replay.

Writes sidecar files under ``<dataset>/force_labels/`` and does **not**
modify the original LeRobot parquet or video assets.

Recommended usage:

    cd ~/dexjoco
    export PYTHONPATH=/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco
    MUJOCO_GL=egl python -u dexquery/scripts/label_forces.py \\
      --task bimanual_assembly \\
      --dataset-root /mnt/ssd/datasets/dexjoco_lerobot_datasets \\
      --zarr-input-dir /mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dexquery.data.episode_replay import replay_episode_forces
from dexquery.data.finger_contact_forces import (
    FINGER_FORCE_DIM,
    WRIST_FT_DIM,
    format_episode_force_summary,
    summarize_force_episode,
)
from dexquery.data.force_label_store import (
    ForceLabelManifest,
    clear_label_checkpoints,
    default_force_label_dir,
    load_checkpoint,
    merge_episode_shards,
    save_checkpoint,
    utc_now_iso,
    write_episode_force_parquet,
    write_manifest,
)
from dexquery.data.lerobot_io import iter_episode_actions
from dexquery.data.zarr_io import discover_zarr_demos, iter_zarr_episodes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="bimanual_assembly")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets"),
    )
    parser.add_argument(
        "--zarr-input-dir",
        type=Path,
        default=None,
        help="Directory of recorded zarr demos (each contains replay.zarr with full state)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Sidecar label directory (default: <dataset>/force_labels)",
    )
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument(
        "--allow-seed-only",
        action="store_true",
        help="Allow labeling without zarr initial-state restore (approximate)",
    )
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--contact-eps",
        type=float,
        default=0.5,
        help="Force norm threshold (N) for finger contact frame ratio stats",
    )
    return parser.parse_args()


def _load_zarr_by_episode(zarr_root: Path) -> dict[int, tuple[Path, np.ndarray, np.ndarray | None]]:
    mapping: dict[int, tuple[Path, np.ndarray, np.ndarray | None]] = {}
    demos = discover_zarr_demos(zarr_root)
    print(f"Loading {len(demos)} zarr demos from {zarr_root} ...", flush=True)
    for episode_index, zarr_path, actions, initial_state in iter_zarr_episodes(zarr_root):
        mapping[episode_index] = (zarr_path, actions, initial_state)
        if (episode_index + 1) % 10 == 0 or episode_index + 1 == len(demos):
            print(f"  loaded zarr {episode_index + 1}/{len(demos)}", flush=True)
    return mapping


def _frames_to_arrays(frames) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    right = np.stack([f.right_finger_force for f in frames], axis=0).astype(np.float32)
    left = np.stack([f.left_finger_force for f in frames], axis=0).astype(np.float32)
    wrist_r = np.stack([f.wrist_ft_right for f in frames], axis=0).astype(np.float32)
    wrist_l = np.stack([f.wrist_ft_left for f in frames], axis=0).astype(np.float32)
    insert_ok = np.asarray([float(f.outcome.insert_ok) for f in frames], dtype=np.float32)
    return right, left, wrist_r, wrist_l, insert_ok


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_root.expanduser() / args.task
    if not dataset_root.exists():
        sys.exit(f"Dataset not found: {dataset_root}")

    label_dir = (
        args.output_dir.expanduser()
        if args.output_dir
        else default_force_label_dir(dataset_root)
    )

    lerobot_episodes = list(
        iter_episode_actions(dataset_root, episode_indices=args.episodes),
    )
    if args.max_episodes is not None:
        lerobot_episodes = lerobot_episodes[: args.max_episodes]

    zarr_map: dict[int, tuple[Path, np.ndarray, np.ndarray | None]] = {}
    if args.zarr_input_dir is not None:
        zarr_root = args.zarr_input_dir.expanduser()
        zarr_map = _load_zarr_by_episode(zarr_root)
        print(f"Zarr demos: {len(zarr_map)} under {zarr_root}")
    elif not args.allow_seed_only:
        sys.exit(
            "Refusing seed-only labeling (inaccurate without scene restore).\n"
            "Pass --zarr-input-dir with raw replay.zarr demos, or --allow-seed-only."
        )
    else:
        warnings.warn(
            "Running seed-only replay without zarr initial-state restore.",
            stacklevel=2,
        )

    print(f"Task: {args.task}")
    print(f"Dataset: {dataset_root}")
    print(f"Sidecar output: {label_dir}")
    print(f"LeRobot episodes to label: {len(lerobot_episodes)}")
    if args.dry_run:
        return

    label_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clear_label_checkpoints(label_dir)
        print(f"Cleared existing checkpoints under {label_dir}", flush=True)

    completed: set[int] = set()
    if args.resume and not args.overwrite:
        checkpoint = load_checkpoint(label_dir)
        if checkpoint is not None:
            completed = {int(x) for x in checkpoint.get("completed_episodes", [])}
        for path in (label_dir / "episodes").glob("episode_*.parquet"):
            completed.add(int(path.stem.rsplit("_", 1)[-1]))
        if completed:
            print(f"Resuming: {len(completed)} episodes already labeled.", flush=True)

    pending = [item for item in lerobot_episodes if int(item[0]) not in completed]
    if len(pending) < len(lerobot_episodes):
        print(f"Episodes remaining: {len(pending)}/{len(lerobot_episodes)}", flush=True)

    episode_summaries: list[dict] = []
    total_frames = 0
    insert_hits = 0

    for i, (episode_index, global_indices, frame_indices, lerobot_actions) in enumerate(
        tqdm(pending, desc="label_forces"),
    ):
        print(
            f"[{i + 1}/{len(pending)}] replay episode {episode_index} "
            f"({len(lerobot_actions)} frames)",
            flush=True,
        )
        initial_state = None
        actions44 = lerobot_actions
        seed = int(args.seed_base + episode_index)

        if episode_index in zarr_map:
            zarr_path, zarr_actions, initial_state = zarr_map[episode_index]
            if zarr_actions.shape != lerobot_actions.shape:
                warnings.warn(
                    f"Episode {episode_index}: zarr action length {zarr_actions.shape[0]} "
                    f"!= LeRobot {lerobot_actions.shape[0]}; using zarr actions.",
                    stacklevel=2,
                )
                actions44 = zarr_actions.astype(np.float32, copy=False)
            if initial_state is None:
                warnings.warn(
                    f"Episode {episode_index}: {zarr_path} has no state[0]; using seed replay.",
                    stacklevel=2,
                )

        frames, _info = replay_episode_forces(
            actions44,
            seed=seed,
            initial_state=initial_state,
            randomize=args.randomize,
        )
        if len(frames) != actions44.shape[0]:
            raise RuntimeError(
                f"Episode {episode_index}: force frame length {len(frames)} != "
                f"action length {actions44.shape[0]}"
            )
        if len(frames) != len(global_indices):
            raise RuntimeError(
                f"Episode {episode_index}: cannot align {len(frames)} frames to "
                f"{len(global_indices)} LeRobot indices"
            )

        stats = summarize_force_episode(frames, contact_eps=args.contact_eps)
        print(f"  {format_episode_force_summary(episode_index, stats)}", flush=True)
        episode_summaries.append({"episode_index": int(episode_index), **stats})

        right, left, wrist_r, wrist_l, insert_ok = _frames_to_arrays(frames)
        write_episode_force_parquet(
            label_dir,
            episode_index,
            global_index=np.asarray(global_indices, dtype=np.int64),
            frame_index=np.asarray(frame_indices, dtype=np.int64),
            right_finger_force=right,
            left_finger_force=left,
            wrist_ft_right=wrist_r,
            wrist_ft_left=wrist_l,
            insert_ok=insert_ok,
        )
        completed.add(int(episode_index))
        save_checkpoint(
            label_dir,
            {
                "task": args.task,
                "source_dataset": str(dataset_root),
                "completed_episodes": sorted(completed),
                "updated_at": utc_now_iso(),
                "zarr_input_dir": str(args.zarr_input_dir) if args.zarr_input_dir else None,
                "seed_base": int(args.seed_base),
                "randomize": bool(args.randomize),
            },
        )
        merge_episode_shards(label_dir)
        total_frames += len(frames)
        insert_hits += int(insert_ok.sum())

    out_path = merge_episode_shards(label_dir)
    if out_path is None:
        sys.exit(f"No labeled episodes found under {label_dir / 'episodes'}")

    manifest = ForceLabelManifest(
        task=args.task,
        source_dataset=str(dataset_root),
        created_at=utc_now_iso(),
        num_frames=total_frames,
        num_episodes=len(completed),
        columns=[
            "index",
            "episode_index",
            "frame_index",
            "right_finger_force",
            "left_finger_force",
            "wrist_ft_right",
            "wrist_ft_left",
            "insert_ok",
        ],
        label_file=out_path.name,
        seed_base=int(args.seed_base),
        randomize=bool(args.randomize),
    )
    manifest_path = write_manifest(label_dir, manifest)

    replay_success = sum(1 for s in episode_summaries if s["insert_ok_rate"] > 0.0)
    summary = {
        "num_episodes": len(completed),
        "num_frames": total_frames,
        "episodes_with_insert_contact": replay_success,
        "replay_insert_episode_rate": replay_success / len(episode_summaries)
        if episode_summaries
        else 0.0,
        "insert_ok_frame_rate": insert_hits / total_frames if total_frames else 0.0,
        "finger_force_dim": FINGER_FORCE_DIM,
        "wrist_ft_dim": WRIST_FT_DIM,
        "contact_eps_n": float(args.contact_eps),
        "zarr_input_dir": str(args.zarr_input_dir) if args.zarr_input_dir else None,
        "episodes": episode_summaries,
    }
    (label_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote labels: {out_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(
        f"Done: replay_insert_episodes={replay_success}/{len(episode_summaries)} "
        f"insert_ok_frames={insert_hits / total_frames:.1%}"
        if total_frames
        else "Done: no frames",
        flush=True,
    )


if __name__ == "__main__":
    main()
