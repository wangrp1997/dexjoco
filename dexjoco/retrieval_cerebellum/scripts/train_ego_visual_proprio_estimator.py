"""Train the deployable ego RGB + proprio coarse-alignment state model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as parquet
from sklearn.ensemble import ExtraTreesRegressor

from retrieval_cerebellum.continuous_visual_estimation import ContinuousVisualFeatureStore
from retrieval_cerebellum.ego_visual_state_estimation import (
    causal_feature_history,
    deployable_visual_proprio_features,
)
from retrieval_cerebellum.scripts.train_continuous_visual_estimator import (
    _local_mask,
    _state_metrics,
    _visual_targets,
)


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--estimation-dir", type=Path, default=None)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/ego_visual_proprio_estimator"),
    )
    parser.add_argument("--history-lengths", type=int, nargs="+", default=(1, 3, 5))
    parser.add_argument("--minimum-leaves", type=int, nargs="+", default=(1, 2, 5, 10))
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _array(table, name: str, width: int) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().values).reshape(table.num_rows, width)


def _episode_deployable_data(
    estimation_path: Path,
    frames: np.ndarray,
    visual_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    columns = [
        "frame_index",
        "sensor_state46",
        "sensor_previous_action44",
        "teacher_peg_in_hole_position",
        "teacher_peg_in_hole_rotvec",
        "teacher_peg_in_right_palm_position",
        "teacher_peg_in_right_palm_rotvec",
    ]
    table = parquet.read_table(estimation_path, columns=columns)
    frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    lookup = {int(frame): row for row, frame in enumerate(frame_index)}
    rows = np.asarray([lookup[int(frame)] for frame in frames], dtype=np.int64)
    features = deployable_visual_proprio_features(
        visual_features,
        _array(table, "sensor_state46", 46)[rows],
        _array(table, "sensor_previous_action44", 44)[rows],
    )
    return features, _visual_targets(table, rows)[:, :5]


def _stack(items, field: int) -> np.ndarray:
    return np.concatenate([item[field] for item in items], axis=0)


def _local_metrics(prediction: np.ndarray, target5: np.ndarray) -> dict[str, float]:
    target11 = np.pad(target5, ((0, 0), (0, 6)))
    local = _local_mask(target11)
    return _state_metrics(
        prediction[local] if np.any(local) else prediction,
        target5[local] if np.any(local) else target5,
    )


def _selection_score(metrics: dict[str, float]) -> float:
    return float(
        metrics["lateral_mean_m"] / 0.01
        + metrics["tilt_mean_rad"] / 0.15
        + metrics["depth_mae_m"] / 0.02
    )


def _direction_metrics(
    prediction: np.ndarray,
    target5: np.ndarray,
) -> dict[str, object]:
    local = _local_mask(np.pad(target5, ((0, 0), (0, 6))))
    predicted = prediction[local]
    target = target5[local]

    def evaluate(
        predicted_vectors: np.ndarray,
        target_vectors: np.ndarray,
        minimum_norm: float,
    ) -> dict[str, float | int]:
        target_norm = np.linalg.norm(target_vectors, axis=1)
        selected = target_norm >= minimum_norm
        predicted_vectors = predicted_vectors[selected]
        target_vectors = target_vectors[selected]
        if not len(target_vectors):
            return {
                "num_rows": 0,
                "same_halfspace_accuracy": 0.0,
                "within_45deg_accuracy": 0.0,
                "angle_p90_rad": float("inf"),
            }
        predicted_norm = np.linalg.norm(predicted_vectors, axis=1)
        cosine = np.sum(predicted_vectors * target_vectors, axis=1) / (
            np.maximum(predicted_norm, 1e-9)
            * np.linalg.norm(target_vectors, axis=1)
        )
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return {
            "num_rows": int(len(angle)),
            "same_halfspace_accuracy": float(np.mean(cosine > 0.0)),
            "within_45deg_accuracy": float(np.mean(angle <= np.pi / 4.0)),
            "angle_p90_rad": float(np.quantile(angle, 0.9)),
        }

    return {
        "lateral": evaluate(predicted[:, :2], target[:, :2], 0.002),
        "tilt": evaluate(predicted[:, 2:4], target[:, 2:4], 0.03),
    }


def main() -> None:
    args = parse_args()
    if min(args.history_lengths) <= 0 or min(args.minimum_leaves) <= 0:
        raise ValueError("history lengths and minimum leaves must be positive")
    if args.trees <= 0:
        raise ValueError("trees must be positive")
    estimation_dir = args.estimation_dir or args.dataset_root / "retrieval_cerebellum_estimation"
    store = ContinuousVisualFeatureStore.load(args.feature_cache)
    raw_episodes = []
    for episode in np.unique(store.episode_index):
        frames, visual_features, split = store.episode(int(episode))
        deployable, target5 = _episode_deployable_data(
            estimation_dir / "episodes" / f"episode_{int(episode):06d}.parquet",
            frames,
            visual_features,
        )
        raw_episodes.append((int(episode), split, deployable, target5))

    candidates = []
    best = None
    for history in args.history_lengths:
        grouped = {"train": [], "validation": [], "test": []}
        for episode, split, deployable, target5 in raw_episodes:
            grouped[split].append(
                (episode, causal_feature_history(deployable, history), target5)
            )
        train_x = _stack(grouped["train"], 1)
        train_y = _stack(grouped["train"], 2)
        validation_x = _stack(grouped["validation"], 1)
        validation_y = _stack(grouped["validation"], 2)
        for minimum_leaf in args.minimum_leaves:
            model = ExtraTreesRegressor(
                n_estimators=args.trees,
                min_samples_leaf=minimum_leaf,
                max_features=0.7,
                n_jobs=-1,
                random_state=args.seed,
            ).fit(train_x, train_y)
            metrics = _local_metrics(model.predict(validation_x), validation_y)
            score = _selection_score(metrics)
            record = {
                "history": history,
                "minimum_leaf": minimum_leaf,
                "selection_score": score,
                "validation_local": metrics,
            }
            candidates.append(record)
            if best is None or score < best[0]:
                best = (score, history, minimum_leaf, model, grouped)
    if best is None:
        raise RuntimeError("no visual-proprio candidate was trained")
    _, history, minimum_leaf, model, grouped = best
    validation_x = _stack(grouped["validation"], 1)
    validation_y = _stack(grouped["validation"], 2)
    test_x = _stack(grouped["test"], 1)
    test_y = _stack(grouped["test"], 2)
    validation_prediction = model.predict(validation_x)
    validation_local = _local_metrics(validation_prediction, validation_y)
    validation_direction = _direction_metrics(validation_prediction, validation_y)
    test_prediction = model.predict(test_x)
    test_local = _local_metrics(test_prediction, test_y)
    test_global = _state_metrics(test_prediction, test_y)
    p1_gates = {
        "lateral_p90_m": bool(test_local["lateral_p90_m"] <= 0.0012),
        "tilt_p90_rad": bool(test_local["tilt_p90_rad"] <= 0.03),
        "depth_p90_m": bool(test_local["depth_p90_m"] <= 0.003),
    }
    test_direction = _direction_metrics(test_prediction, test_y)

    def coarse_gates(metrics, direction):
        return {
            "lateral_p90_m": bool(metrics["lateral_p90_m"] <= 0.01),
            "tilt_p90_rad": bool(metrics["tilt_p90_rad"] <= 0.15),
            "depth_p90_m": bool(metrics["depth_p90_m"] <= 0.02),
            "lateral_same_halfspace": bool(
                direction["lateral"]["same_halfspace_accuracy"] >= 0.8
            ),
            "lateral_within_45deg": bool(
                direction["lateral"]["within_45deg_accuracy"] >= 0.6
            ),
            "tilt_same_halfspace": bool(
                direction["tilt"]["same_halfspace_accuracy"] >= 0.8
            ),
        }

    validation_coarse_gates = coarse_gates(validation_local, validation_direction)
    test_coarse_gates = coarse_gates(test_local, test_direction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "state_model.joblib"
    joblib.dump(model, model_path)
    summary = {
        "stage": "V2 deployable ego RGB + proprio coarse state estimator",
        "feature_cache": str(args.feature_cache),
        "selected_history": history,
        "selected_minimum_leaf": minimum_leaf,
        "trees": args.trees,
        "model": str(model_path),
        "validation_local": validation_local,
        "validation_direction": validation_direction,
        "test_global": test_global,
        "test_local": test_local,
        "test_direction": test_direction,
        "p1_gates": p1_gates,
        "approved_for_p1": bool(all(p1_gates.values())),
        "validation_coarse_alignment_gates": validation_coarse_gates,
        "test_coarse_alignment_gates": test_coarse_gates,
        "approved_for_visual_coarse_alignment": bool(
            all(validation_coarse_gates.values())
            and all(test_coarse_gates.values())
        ),
        "uses_rgb_proprio_action_only_at_inference": True,
        "uses_teacher_geometry_at_inference": False,
        "candidates": candidates,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
