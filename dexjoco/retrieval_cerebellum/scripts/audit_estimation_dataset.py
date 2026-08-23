"""Audit P2 estimation shards and their alignment with existing force replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from retrieval_cerebellum.estimation_data import SENSOR_VECTOR_DIMS


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--estimation-dir", type=Path, default=None)
    parser.add_argument("--reference-force-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _vector(table, name: str, width: int) -> np.ndarray:
    values = np.asarray(table[name].to_pylist(), dtype=np.float64)
    if values.shape != (table.num_rows, width):
        raise ValueError(f"{name} expected shape ({table.num_rows}, {width}), got {values.shape}")
    return values


def _stats(values: np.ndarray) -> dict[str, float | bool]:
    return {
        "finite": bool(np.isfinite(values).all()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "zero_fraction": float(np.mean(values == 0.0)),
    }


def main() -> None:
    import pyarrow.parquet as parquet

    args = parse_args()
    estimation_dir = args.estimation_dir or args.dataset_root / "retrieval_cerebellum_estimation"
    reference_force_dir = (
        args.reference_force_dir
        or args.dataset_root / "force_labels_20260812_current_replay"
    )
    output = args.output or estimation_dir / "audit.json"
    paths = sorted((estimation_dir / "episodes").glob("episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No estimation shards under {estimation_dir}")

    sensor_parts = {name: [] for name in SENSOR_VECTOR_DIMS}
    all_indices: list[np.ndarray] = []
    split_episodes: dict[str, set[int]] = {}
    family_episodes: dict[str, set[int]] = {}
    teacher_totals = {
        "teacher_peg_ok": 0,
        "teacher_tray_ok": 0,
        "teacher_insert_ok": 0,
    }
    reference_max_abs = {"fingertip_force_world": 0.0, "wrist_wrench_world": 0.0}
    reference_mean_abs_sum = {"fingertip_force_world": 0.0, "wrist_wrench_world": 0.0}
    reference_mismatched_episodes: list[int] = []
    reference_exact_episodes = 0
    reference_rows = 0
    timestamp_offsets: list[np.ndarray] = []

    for path in paths:
        table = parquet.read_table(path)
        episode_values = set(int(value) for value in table["episode_index"].to_pylist())
        if len(episode_values) != 1:
            raise ValueError(f"{path} has episode ids {episode_values}")
        episode_index = next(iter(episode_values))
        split_values = set(str(value) for value in table["split"].to_pylist())
        family_values = set(str(value) for value in table["family_id"].to_pylist())
        if len(split_values) != 1 or len(family_values) != 1:
            raise ValueError(f"{path} has non-constant split or family")
        split_episodes.setdefault(next(iter(split_values)), set()).add(episode_index)
        family_episodes.setdefault(next(iter(family_values)), set()).add(episode_index)

        indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)
        if np.any(np.diff(indices) <= 0):
            raise ValueError(f"{path} index is not strictly increasing")
        all_indices.append(indices)
        for name, width in SENSOR_VECTOR_DIMS.items():
            sensor_parts[name].append(_vector(table, name, width))
        for name in teacher_totals:
            teacher_totals[name] += int(np.count_nonzero(table[name].to_numpy()))
        timestamp_offsets.append(
            np.asarray(table["sensor_timestamp_s"].to_numpy(), dtype=np.float64)
            - np.asarray(table["dataset_timestamp_s"].to_numpy(), dtype=np.float64)
        )

        reference_path = reference_force_dir / "episodes" / path.name
        if reference_path.is_file():
            reference = parquet.read_table(reference_path)
            reference_index = np.asarray(reference["index"].to_numpy(), dtype=np.int64)
            lookup = {int(value): row for row, value in enumerate(reference_index)}
            try:
                positions = np.asarray([lookup[int(value)] for value in indices], dtype=np.int64)
            except KeyError as exc:
                raise ValueError(f"{reference_path} is missing estimation index {exc.args[0]}") from exc
            expected_fingertip = np.concatenate(
                [
                    np.asarray(reference["right_finger_force"].to_pylist(), dtype=np.float64)[positions],
                    np.asarray(reference["left_finger_force"].to_pylist(), dtype=np.float64)[positions],
                ],
                axis=1,
            )
            expected_wrist = np.concatenate(
                [
                    np.asarray(reference["wrist_ft_right"].to_pylist(), dtype=np.float64)[positions],
                    np.asarray(reference["wrist_ft_left"].to_pylist(), dtype=np.float64)[positions],
                ],
                axis=1,
            )
            fingertip_abs = np.abs(
                sensor_parts["sensor_fingertip_force_world"][-1] - expected_fingertip
            )
            wrist_abs = np.abs(
                sensor_parts["sensor_wrist_wrench_world"][-1] - expected_wrist
            )
            reference_max_abs["fingertip_force_world"] = max(
                reference_max_abs["fingertip_force_world"],
                float(np.max(fingertip_abs)),
            )
            reference_max_abs["wrist_wrench_world"] = max(
                reference_max_abs["wrist_wrench_world"],
                float(np.max(wrist_abs)),
            )
            reference_mean_abs_sum["fingertip_force_world"] += float(
                fingertip_abs.sum()
            )
            reference_mean_abs_sum["wrist_wrench_world"] += float(wrist_abs.sum())
            if np.count_nonzero(fingertip_abs) == 0 and np.count_nonzero(wrist_abs) == 0:
                reference_exact_episodes += 1
            else:
                reference_mismatched_episodes.append(episode_index)
            reference_rows += table.num_rows

    indices = np.concatenate(all_indices)
    sensor_values = {
        name: np.concatenate(parts, axis=0) for name, parts in sensor_parts.items()
    }
    offsets = np.concatenate(timestamp_offsets)
    if np.unique(indices).size != indices.size:
        raise ValueError("global estimation indices are not unique")
    num_frames = int(indices.size)
    audit = {
        "estimation_dir": str(estimation_dir),
        "num_episodes": len(paths),
        "num_frames": num_frames,
        "global_index_unique": True,
        "split_episode_counts": {
            name: len(episodes) for name, episodes in sorted(split_episodes.items())
        },
        "family_episode_counts": {
            name: len(episodes) for name, episodes in sorted(family_episodes.items())
        },
        "sensor_stats": {
            name: _stats(values) for name, values in sensor_values.items()
        },
        "teacher_true_frame_counts": teacher_totals,
        "teacher_true_frame_rates": {
            name: float(count / num_frames) for name, count in teacher_totals.items()
        },
        "sensor_minus_dataset_timestamp_s": _stats(offsets),
        "reference_force_comparison": {
            "reference_dir": str(reference_force_dir),
            "compared_rows": reference_rows,
            "exact_episode_count": reference_exact_episodes,
            "mismatched_episodes": sorted(reference_mismatched_episodes),
            "max_abs_error": reference_max_abs,
            "mean_abs_error": {
                "fingertip_force_world": (
                    reference_mean_abs_sum["fingertip_force_world"]
                    / (reference_rows * SENSOR_VECTOR_DIMS["sensor_fingertip_force_world"])
                ),
                "wrist_wrench_world": (
                    reference_mean_abs_sum["wrist_wrench_world"]
                    / (reference_rows * SENSOR_VECTOR_DIMS["sensor_wrist_wrench_world"])
                ),
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
