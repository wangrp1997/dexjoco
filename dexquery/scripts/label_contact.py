#!/usr/bin/env python3
"""Generate per-frame tray_ok / peg_ok labels via MuJoCo contact replay.

Writes sidecar files under ``<dataset>/dexquery_labels/`` and does **not**
modify the original LeRobot parquet or video assets.

Recommended usage (accurate labels via full initial-state restore):

    python dexquery/scripts/label_contact.py \\
      --task bimanual_assembly \\
      --dataset-root /mnt/ssd/datasets/dexjoco_lerobot_datasets \\
      --zarr-input-dir /path/to/raw/bimanual_assembly/demos

Fallback (seed-only replay, approximate — use only when zarr is unavailable):

    python dexquery/scripts/label_contact.py \\
      --task bimanual_assembly \\
      --allow-seed-only
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

from dexquery.data.episode_replay import replay_episode_actions
from dexquery.data.label_store import (
    LabelManifest,
    clear_label_checkpoints,
    default_label_dir,
    load_checkpoint,
    merge_episode_shards,
    save_checkpoint,
    utc_now_iso,
    write_episode_outcome_parquet,
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
        help="Root directory that contains one LeRobot dataset folder per task",
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
        help="Sidecar label directory (default: <dataset>/dexquery_labels)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Env seed when not restoring from zarr (seed_base + episode_index)",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Enable visual randomization during replay (default: off)",
    )
    parser.add_argument(
        "--allow-seed-only",
        action="store_true",
        help="Allow labeling without zarr initial-state restore (approximate labels)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of episode indices (default: all)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Process at most this many episodes (debug)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print episode counts without running sim",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip episodes already saved under output-dir/episodes/ (default: on)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing episode shards and restart from scratch",
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


def _load_rate_totals(label_dir: Path) -> tuple[float, float, float, int]:
    path = label_dir / "outcomes.parquet"
    if not path.exists():
        return 0.0, 0.0, 0.0, 0
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["tray_ok", "peg_ok", "insert_ok"])
    tray = table.column("tray_ok").to_numpy(zero_copy_only=False)
    peg = table.column("peg_ok").to_numpy(zero_copy_only=False)
    insert = table.column("insert_ok").to_numpy(zero_copy_only=False)
    return float(tray.sum()), float(peg.sum()), float(insert.sum()), int(len(tray))


def _format_rates(tray_sum: float, peg_sum: float, insert_sum: float, frames: int) -> str:
    if frames <= 0:
        return "tray=0.0% peg=0.0% insert=0.0%"
    return (
        f"tray={tray_sum / frames:.1%} "
        f"peg={peg_sum / frames:.1%} "
        f"insert={insert_sum / frames:.1%}"
    )


def _write_summary_json(
    label_dir: Path,
    *,
    tray_sum: float,
    peg_sum: float,
    insert_sum: float,
    frames: int,
    num_episodes: int,
    args: argparse.Namespace,
) -> None:
    summary = {
        "tray_ok_rate": tray_sum / frames if frames else 0.0,
        "peg_ok_rate": peg_sum / frames if frames else 0.0,
        "insert_ok_rate": insert_sum / frames if frames else 0.0,
        "num_episodes": num_episodes,
        "num_frames": frames,
        "zarr_input_dir": str(args.zarr_input_dir) if args.zarr_input_dir else None,
        "seed_only": args.zarr_input_dir is None,
    }
    (label_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_root.expanduser() / args.task
    if not dataset_root.exists():
        sys.exit(f"Dataset not found: {dataset_root}")

    label_dir = args.output_dir.expanduser() if args.output_dir else default_label_dir(dataset_root)

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
        if len(zarr_map) < len(lerobot_episodes):
            warnings.warn(
                f"Fewer zarr demos ({len(zarr_map)}) than LeRobot episodes "
                f"({len(lerobot_episodes)}); missing episodes fall back to seed replay.",
                stacklevel=2,
            )
    elif not args.allow_seed_only:
        sys.exit(
            "Refusing seed-only labeling (inaccurate without scene restore).\n"
            "Pass --zarr-input-dir with raw replay.zarr demos, or --allow-seed-only."
        )
    else:
        warnings.warn(
            "Running seed-only replay without zarr initial-state restore. "
            "Contact labels may not align with recorded demonstrations.",
            stacklevel=2,
        )

    print(f"Task: {args.task}")
    print(f"Dataset: {dataset_root}")
    print(f"Sidecar output: {label_dir}")
    print(f"LeRobot episodes to label: {len(lerobot_episodes)}")
    if args.dry_run:
        if args.zarr_input_dir:
            print(f"Zarr files: {len(discover_zarr_demos(args.zarr_input_dir.expanduser()))}")
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
            print(
                f"Resuming: {len(completed)} episodes already labeled, will skip them.",
                flush=True,
            )

    pending = [
        item for item in lerobot_episodes if int(item[0]) not in completed
    ]
    if not pending and completed:
        print("All requested episodes already labeled; merging shards.", flush=True)
    elif len(pending) < len(lerobot_episodes):
        print(f"Episodes remaining: {len(pending)}/{len(lerobot_episodes)}", flush=True)

    tray_sum, peg_sum, insert_sum, total_frames = _load_rate_totals(label_dir)
    if total_frames:
        print(f"Current {_format_rates(tray_sum, peg_sum, insert_sum, total_frames)}", flush=True)

    for i, (episode_index, global_indices, frame_indices, lerobot_actions) in enumerate(
        tqdm(pending, desc="label_contact"),
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
                    f"!= LeRobot {lerobot_actions.shape[0]}; using zarr actions for replay.",
                    stacklevel=2,
                )
                actions44 = zarr_actions.astype(np.float32, copy=False)
            if initial_state is None:
                warnings.warn(
                    f"Episode {episode_index}: {zarr_path} has no state[0]; using seed replay.",
                    stacklevel=2,
                )

        outcomes, _info = replay_episode_actions(
            actions44,
            seed=seed,
            initial_state=initial_state,
            randomize=args.randomize,
        )
        if len(outcomes) != actions44.shape[0]:
            raise RuntimeError(
                f"Episode {episode_index}: outcome length {len(outcomes)} != "
                f"action length {actions44.shape[0]}"
            )

        if len(outcomes) != len(global_indices):
            raise RuntimeError(
                f"Episode {episode_index}: cannot align {len(outcomes)} outcomes to "
                f"{len(global_indices)} LeRobot frames"
            )

        ep_index = np.asarray(global_indices, dtype=np.int64)
        ep_frames = np.asarray(frame_indices, dtype=np.int64)
        ep_tray = np.asarray([float(o.tray_ok) for o in outcomes], dtype=np.float32)
        ep_peg = np.asarray([float(o.peg_ok) for o in outcomes], dtype=np.float32)
        ep_insert = np.asarray([float(o.insert_ok) for o in outcomes], dtype=np.float32)

        write_episode_outcome_parquet(
            label_dir,
            episode_index,
            global_index=ep_index,
            frame_index=ep_frames,
            tray_ok=ep_tray,
            peg_ok=ep_peg,
            insert_ok=ep_insert,
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
        tray_sum += float(ep_tray.sum())
        peg_sum += float(ep_peg.sum())
        insert_sum += float(ep_insert.sum())
        total_frames += int(len(ep_tray))
        print(f"  {_format_rates(tray_sum, peg_sum, insert_sum, total_frames)}", flush=True)
        _write_summary_json(
            label_dir,
            tray_sum=tray_sum,
            peg_sum=peg_sum,
            insert_sum=insert_sum,
            frames=total_frames,
            num_episodes=len(completed),
            args=args,
        )

    out_path = merge_episode_shards(label_dir)
    if out_path is None:
        sys.exit(f"No labeled episodes found under {label_dir / 'episodes'}")

    manifest = LabelManifest(
        task=args.task,
        source_dataset=str(dataset_root),
        created_at=utc_now_iso(),
        num_frames=total_frames,
        num_episodes=len(completed),
        columns=["index", "episode_index", "frame_index", "tray_ok", "peg_ok", "insert_ok"],
        label_file=out_path.name,
        seed_base=int(args.seed_base),
        randomize=bool(args.randomize),
    )
    manifest_path = write_manifest(label_dir, manifest)

    summary = {
        "tray_ok_rate": tray_sum / total_frames if total_frames else 0.0,
        "peg_ok_rate": peg_sum / total_frames if total_frames else 0.0,
        "insert_ok_rate": insert_sum / total_frames if total_frames else 0.0,
        "num_episodes": len(completed),
        "num_frames": total_frames,
        "zarr_input_dir": str(args.zarr_input_dir) if args.zarr_input_dir else None,
        "seed_only": args.zarr_input_dir is None,
    }
    (label_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote labels: {out_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Done: {_format_rates(tray_sum, peg_sum, insert_sum, total_frames)}")


if __name__ == "__main__":
    main()
