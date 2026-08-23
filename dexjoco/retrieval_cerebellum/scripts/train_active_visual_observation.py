"""Train a deployable wrist-action to ego-visibility transition model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from retrieval_cerebellum.active_visual_observation import (
    ActiveVisualReliabilityModel,
)
from retrieval_cerebellum.active_view_probe import load_active_view_transitions
from retrieval_cerebellum.assembly_kinematics import wrist_twists_from_action44
from retrieval_cerebellum.continuous_visual_estimation import (
    ContinuousVisualFeatureStore,
)


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--estimation-dir", type=Path, default=None)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--probe-transitions", type=Path, nargs="*", default=())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/active_visual_observation"),
    )
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    )
    return parser.parse_args()


def _perceptual_reliability(path: Path) -> dict[tuple[int, int], float]:
    with np.load(path, allow_pickle=False) as data:
        if "perceptual_reliability" not in data:
            raise ValueError("feature cache lacks deployable perceptual_reliability")
        episodes = np.asarray(data["episode_index"], dtype=np.int64)
        frames = np.asarray(data["frame_index"], dtype=np.int64)
        reliability = np.asarray(data["perceptual_reliability"], dtype=np.float64)
    return {
        (int(episode), int(frame)): float(value)
        for episode, frame, value in zip(episodes, frames, reliability, strict=True)
    }


def _fixed_list(table, name: str, width: int) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().values).reshape(table.num_rows, width)


def _episode_pairs(
    estimation_path: Path,
    frames: np.ndarray,
    features: np.ndarray,
    reliability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import pyarrow.parquet as parquet

    table = parquet.read_table(
        estimation_path,
        columns=["frame_index", "sensor_previous_action44"],
    )
    available_frames = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    lookup = {int(frame): row for row, frame in enumerate(available_frames)}
    try:
        rows = np.asarray([lookup[int(frame)] for frame in frames], dtype=np.int64)
    except KeyError as error:
        raise ValueError(
            f"visual frame {error.args[0]} missing from {estimation_path}"
        ) from error
    actions44 = _fixed_list(table, "sensor_previous_action44", 44)[rows]
    right_controls = wrist_twists_from_action44(actions44, side="right")
    left_controls = wrist_twists_from_action44(actions44, side="left")
    controls12 = np.concatenate([right_controls, left_controls], axis=1)
    return features[:-1], controls12, reliability[:-1], reliability[1:]


def _metrics(
    model: ActiveVisualReliabilityModel,
    features: np.ndarray,
    controls: np.ndarray,
    current: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | int]:
    prediction = np.asarray(
        [
            model.predict(feature, reliability, control)
            for feature, reliability, control in zip(
                features,
                current,
                controls,
                strict=True,
            )
        ]
    )
    persistence_error = current - target
    model_error = prediction - target
    return {
        "num_pairs": int(len(target)),
        "persistence_mae": float(np.mean(np.abs(persistence_error))),
        "model_mae": float(np.mean(np.abs(model_error))),
        "model_rmse": float(np.sqrt(np.mean(model_error**2))),
        "target_mean": float(np.mean(target)),
        "prediction_mean": float(np.mean(prediction)),
    }


def _probe_pairs(path: Path) -> tuple[int, tuple[np.ndarray, ...]]:
    transitions = load_active_view_transitions(path)
    with np.load(path, allow_pickle=False) as data:
        episodes = np.asarray(data["episode_index"], dtype=np.int64)
    unique_episodes = np.unique(episodes)
    if unique_episodes.shape != (1,):
        raise ValueError(f"probe transition file must contain one episode: {path}")
    return int(unique_episodes[0]), (
        np.stack([item.feature_before for item in transitions]),
        np.stack([item.control12 for item in transitions]),
        np.asarray(
            [item.reliability_before for item in transitions], dtype=np.float64
        ),
        np.asarray(
            [item.reliability_after for item in transitions], dtype=np.float64
        ),
    )


def _stack(items, field: int) -> np.ndarray:
    return np.concatenate([item[field] for item in items], axis=0)


def main() -> None:
    args = parse_args()
    if min(args.ridge_alphas) <= 0.0:
        raise ValueError("ridge alphas must be positive")
    estimation_dir = (
        args.estimation_dir
        or args.dataset_root / "retrieval_cerebellum_estimation"
    )
    store = ContinuousVisualFeatureStore.load(args.feature_cache)
    reliability_lookup = _perceptual_reliability(args.feature_cache)
    grouped: dict[str, list[tuple[np.ndarray, ...]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    episode_splits: dict[int, str] = {}
    for episode in np.unique(store.episode_index):
        frames, features, split = store.episode(int(episode))
        episode_splits[int(episode)] = split
        reliability = np.asarray(
            [reliability_lookup[(int(episode), int(frame))] for frame in frames],
            dtype=np.float64,
        )
        if len(frames) < 2:
            continue
        pairs = _episode_pairs(
            estimation_dir / "episodes" / f"episode_{int(episode):06d}.parquet",
            frames,
            features,
            reliability,
        )
        grouped[split].append(pairs)
    probe_pair_count = 0
    for path in args.probe_transitions:
        episode, pairs = _probe_pairs(path)
        if episode not in episode_splits:
            raise ValueError(
                f"probe episode {episode} is absent from feature-cache split metadata"
            )
        if pairs[0].shape[1] != store.features.shape[1]:
            raise ValueError(
                f"probe feature dimension {pairs[0].shape[1]} does not match "
                f"cache dimension {store.features.shape[1]}"
            )
        grouped[episode_splits[episode]].append(pairs)
        probe_pair_count += len(pairs[0])
    if not grouped["train"] or not grouped["validation"] or not grouped["test"]:
        raise ValueError("train, validation, and test splits all require transition pairs")

    train = tuple(_stack(grouped["train"], index) for index in range(4))
    validation = tuple(_stack(grouped["validation"], index) for index in range(4))
    test = tuple(_stack(grouped["test"], index) for index in range(4))
    candidates = []
    best = None
    for alpha in args.ridge_alphas:
        model = ActiveVisualReliabilityModel.fit(
            *train,
            alpha=alpha,
        )
        metrics = _metrics(model, *validation)
        record = {"alpha": alpha, **metrics}
        candidates.append(record)
        score = float(metrics["model_mae"])
        if best is None or score < best[0]:
            best = (score, model, alpha)
    if best is None:
        raise RuntimeError("no active visual reliability model was trained")
    _, model, alpha = best
    validation_metrics = _metrics(model, *validation)
    test_metrics = _metrics(model, *test)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.npz"
    model.save(model_path)
    summary = {
        "stage": "deployable ego active-visual reliability dynamics",
        "model": str(model_path),
        "feature_cache": str(args.feature_cache),
        "probe_transition_files": [str(path) for path in args.probe_transitions],
        "num_probe_pairs": probe_pair_count,
        "selected_alpha": alpha,
        "validation": validation_metrics,
        "test": test_metrics,
        "improves_over_persistence_on_test": bool(
            test_metrics["model_mae"] < test_metrics["persistence_mae"]
        ),
        "approved_for_active_control": bool(
            test_metrics["model_mae"] < test_metrics["persistence_mae"]
        ),
        "runtime_inputs": [
            "ego RGB feature",
            "current perceptual reliability",
            "executed bilateral wrist twist",
        ],
        "reads_teacher_geometry_at_runtime": False,
        "reads_simulator_object_pose_at_runtime": False,
        "candidates": candidates,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
