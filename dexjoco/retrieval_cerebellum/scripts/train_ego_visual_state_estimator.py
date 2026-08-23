"""Train and freeze an ego-only RGB spatial V2 state estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from retrieval_cerebellum.continuous_visual_estimation import (
    ContinuousVisualFeatureStore,
    ContinuousVisualStateModel,
)
from retrieval_cerebellum.ego_visual_state_estimation import causal_feature_history
from retrieval_cerebellum.scripts.train_continuous_visual_estimator import (
    _episode_data,
    _local_mask,
    _rotation_metrics,
    _selection_score,
    _state_metrics,
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
        default=Path("outputs/retrieval_cerebellum/ego_visual_state_estimator"),
    )
    parser.add_argument("--history-lengths", type=int, nargs="+", default=(1, 3, 5))
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=(1.0, 10.0, 100.0, 1000.0, 10000.0),
    )
    return parser.parse_args()


def _load_perceptual_reliability(path: Path) -> dict[tuple[int, int], float]:
    with np.load(path, allow_pickle=False) as data:
        episodes = np.asarray(data["episode_index"], dtype=np.int64)
        frames = np.asarray(data["frame_index"], dtype=np.int64)
        reliability = np.asarray(data["perceptual_reliability"], dtype=np.float64)
    return {
        (int(episode), int(frame)): float(value)
        for episode, frame, value in zip(episodes, frames, reliability, strict=True)
    }


def _stack(items, field: int) -> np.ndarray:
    return np.concatenate([item[field] for item in items], axis=0)


def _metrics(
    model: ContinuousVisualStateModel,
    items,
    perceptual_lookup: dict[tuple[int, int], float],
) -> dict[str, object]:
    features = _stack(items, 2)
    targets = _stack(items, 3)
    mean5, _, rotations, model_reliability = model.predict_arrays(features)
    result: dict[str, object] = {
        "global": {
            **_state_metrics(mean5, targets[:, :5]),
            **_rotation_metrics(rotations, targets),
        }
    }
    local = _local_mask(targets)
    result["local"] = (
        {
            **_state_metrics(mean5[local], targets[local, :5]),
            **_rotation_metrics(rotations[local], targets[local]),
        }
        if np.any(local)
        else None
    )
    perception = np.concatenate(
        [
            np.asarray(
                [
                    perceptual_lookup[(item[0], int(frame))]
                    for frame in item[1]
                ],
                dtype=np.float64,
            )
            for item in items
        ]
    )
    combined_reliability = np.sqrt(
        np.clip(model_reliability, 0.0, 1.0) * np.clip(perception, 0.0, 1.0)
    )
    errors = mean5 - targets[:, :5]
    lateral = np.linalg.norm(errors[:, :2], axis=1)
    tilt = np.linalg.norm(errors[:, 2:4], axis=1)
    depth = np.abs(errors[:, 4])
    coverage = {}
    order = np.argsort(combined_reliability)[::-1]
    for fraction in (1.0, 0.75, 0.5, 0.25):
        count = max(1, int(np.ceil(len(order) * fraction)))
        selected = order[:count]
        coverage[f"{int(fraction * 100)}pct"] = {
            "num_rows": count,
            "minimum_reliability": float(combined_reliability[selected].min()),
            "lateral_p90_m": float(np.quantile(lateral[selected], 0.9)),
            "tilt_p90_rad": float(np.quantile(tilt[selected], 0.9)),
            "depth_p90_m": float(np.quantile(depth[selected], 0.9)),
        }
    result["reliability_coverage"] = coverage
    result["per_episode"] = {}
    cursor = 0
    for episode, frames, episode_features, episode_targets in items:
        count = len(frames)
        episode_prediction = mean5[cursor : cursor + count]
        result["per_episode"][str(episode)] = _state_metrics(
            episode_prediction,
            episode_targets[:, :5],
        )
        cursor += count
    return result


def main() -> None:
    args = parse_args()
    if min(args.history_lengths) <= 0 or min(args.ridge_alphas) <= 0.0:
        raise ValueError("history lengths and ridge alphas must be positive")
    estimation_dir = args.estimation_dir or args.dataset_root / "retrieval_cerebellum_estimation"
    store = ContinuousVisualFeatureStore.load(args.feature_cache)
    perceptual_lookup = _load_perceptual_reliability(args.feature_cache)
    raw_episodes = {}
    for episode in np.unique(store.episode_index):
        frames, features, split = store.episode(int(episode))
        raw_episodes[int(episode)] = (frames, features, split)

    candidates = []
    best = None
    for history in args.history_lengths:
        grouped = {"train": [], "validation": [], "test": []}
        for episode, (frames, raw_features, split) in raw_episodes.items():
            history_features = causal_feature_history(raw_features, history)
            estimation_path = estimation_dir / "episodes" / f"episode_{episode:06d}.parquet"
            episode_features, targets = _episode_data(
                estimation_path,
                frames,
                history_features,
            )
            grouped[split].append((episode, frames, episode_features, targets))
        train_x = _stack(grouped["train"], 2)
        train_y = _stack(grouped["train"], 3)
        validation_x = _stack(grouped["validation"], 2)
        validation_y = _stack(grouped["validation"], 3)
        for alpha in args.ridge_alphas:
            model = ContinuousVisualStateModel.fit(
                train_x,
                train_y,
                alpha=alpha,
                calibration_features=validation_x,
                calibration_targets11=validation_y,
            )
            prediction, _, _, _ = model.predict_arrays(validation_x)
            local = _local_mask(validation_y)
            selection_metrics = _state_metrics(
                prediction[local] if np.any(local) else prediction,
                validation_y[local, :5] if np.any(local) else validation_y[:, :5],
            )
            score = _selection_score(selection_metrics)
            record = {
                "history": history,
                "alpha": alpha,
                "selection_score": score,
                "validation_selection_metrics": selection_metrics,
            }
            candidates.append(record)
            if best is None or score < best[0]:
                best = (score, history, alpha, model, grouped)
    if best is None:
        raise RuntimeError("no ego visual model candidate was trained")
    _, history, alpha, model, grouped = best
    validation_metrics = _metrics(model, grouped["validation"], perceptual_lookup)
    test_metrics = _metrics(model, grouped["test"], perceptual_lookup)
    test_local = test_metrics["local"] or test_metrics["global"]
    gates = {
        "lateral_p90_m": bool(test_local["lateral_p90_m"] <= 0.0012),
        "tilt_p90_rad": bool(test_local["tilt_p90_rad"] <= 0.03),
        "depth_p90_m": bool(test_local["depth_p90_m"] <= 0.003),
    }
    coarse_gate = {
        "lateral_p90_m": bool(test_local["lateral_p90_m"] <= 0.01),
        "tilt_p90_rad": bool(test_local["tilt_p90_rad"] <= 0.15),
        "depth_p90_m": bool(test_local["depth_p90_m"] <= 0.02),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.npz"
    model.save(model_path)
    summary = {
        "stage": "V2 ego-only RGB spatial five-dimensional estimator",
        "feature_cache": str(args.feature_cache),
        "selected_history": history,
        "selected_alpha": alpha,
        "model": str(model_path),
        "validation": validation_metrics,
        "test": test_metrics,
        "p1_thresholds": {
            "lateral_p90_m": 0.0012,
            "tilt_p90_rad": 0.03,
            "depth_p90_m": 0.003,
        },
        "p1_gates": gates,
        "approved_for_p1": bool(all(gates.values())),
        "coarse_alignment_thresholds": {
            "lateral_p90_m": 0.01,
            "tilt_p90_rad": 0.15,
            "depth_p90_m": 0.02,
        },
        "coarse_alignment_gates": coarse_gate,
        "approved_for_visual_coarse_alignment": bool(all(coarse_gate.values())),
        "uses_rgb_only_at_inference": True,
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
