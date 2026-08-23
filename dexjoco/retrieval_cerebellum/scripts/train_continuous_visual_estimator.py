"""Train and evaluate the V2 continuous visual assembly estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from retrieval_cerebellum.assembly_kinematics import (
    palm_pose_from_action44,
    pose_matrix,
)
from retrieval_cerebellum.continuous_visual_estimation import (
    ContinuousVisualFeatureStore,
    ContinuousVisualStateModel,
    rotation_to_sixd,
)


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--estimation-dir", type=Path, default=None)
    parser.add_argument(
        "--visual-cache",
        type=Path,
        default=Path(
            "outputs/retrieval_cerebellum/continuous_visual/clip_vit_b16_pca32.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/continuous_visual_estimator"),
    )
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=(1.0, 10.0, 100.0, 1000.0, 10000.0),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _array(table, name: str, width: int) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().values).reshape(table.num_rows, width)


def _visual_targets(table, rows: np.ndarray) -> np.ndarray:
    state = _array(table, "sensor_state46", 46)[rows]
    peg_in_hole_position = _array(table, "teacher_peg_in_hole_position", 3)[rows]
    peg_in_hole_rotvec = _array(table, "teacher_peg_in_hole_rotvec", 3)[rows]
    peg_attachment_position = _array(
        table,
        "teacher_peg_in_right_palm_position",
        3,
    )[rows]
    peg_attachment_rotvec = _array(
        table,
        "teacher_peg_in_right_palm_rotvec",
        3,
    )[rows]
    targets = np.empty((len(rows), 11), dtype=np.float64)
    targets[:, :5] = np.column_stack(
        [
            peg_in_hole_position[:, 0],
            peg_in_hole_position[:, 1],
            peg_in_hole_rotvec[:, 0],
            peg_in_hole_rotvec[:, 1],
            -peg_in_hole_position[:, 2],
        ]
    )
    for output_row, source_row in enumerate(range(len(rows))):
        action = np.zeros(44, dtype=np.float64)
        action[:3] = state[source_row, :3]
        action[3:6] = Rotation.from_quat(
            state[source_row, 3:7],
            scalar_first=True,
        ).as_rotvec()
        right_palm_world = palm_pose_from_action44(action, side="right")
        peg_world = right_palm_world @ pose_matrix(
            peg_attachment_position[source_row],
            peg_attachment_rotvec[source_row],
        )
        peg_in_hole_rotation = Rotation.from_rotvec(
            peg_in_hole_rotvec[source_row]
        ).as_matrix()
        hole_rotation_world = peg_world[:3, :3] @ peg_in_hole_rotation.T
        targets[output_row, 5:] = rotation_to_sixd(hole_rotation_world)
    return targets


def _episode_data(
    estimation_path: Path,
    cache_frames: np.ndarray,
    cache_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as parquet

    columns = [
        "frame_index",
        "sensor_state46",
        "teacher_peg_in_hole_position",
        "teacher_peg_in_hole_rotvec",
        "teacher_peg_in_right_palm_position",
        "teacher_peg_in_right_palm_rotvec",
    ]
    table = parquet.read_table(estimation_path, columns=columns)
    frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    lookup = {int(frame): row for row, frame in enumerate(frame_index)}
    try:
        rows = np.asarray([lookup[int(frame)] for frame in cache_frames], dtype=np.int64)
    except KeyError as error:
        raise ValueError(
            f"visual cache frame {error.args[0]} missing from {estimation_path}"
        ) from error
    return cache_features, _visual_targets(table, rows)


def _state_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(target)
    lateral = np.linalg.norm(error[:, :2], axis=1)
    tilt = np.linalg.norm(error[:, 2:4], axis=1)
    depth = np.abs(error[:, 4])
    return {
        "num_rows": int(len(error)),
        "lateral_mean_m": float(np.mean(lateral)),
        "lateral_p90_m": float(np.quantile(lateral, 0.9)),
        "tilt_mean_rad": float(np.mean(tilt)),
        "tilt_p90_rad": float(np.quantile(tilt, 0.9)),
        "depth_mae_m": float(np.mean(depth)),
        "depth_p90_m": float(np.quantile(depth, 0.9)),
    }


def _rotation_metrics(prediction: np.ndarray, target11: np.ndarray) -> dict[str, float]:
    errors = []
    for predicted, target in zip(prediction, target11, strict=True):
        target_columns = target[5:].reshape(3, 2)
        first = target_columns[:, 0]
        second = target_columns[:, 1]
        third = np.cross(first, second)
        target_rotation = np.column_stack([first, second, third])
        delta = Rotation.from_matrix(predicted.T @ target_rotation).magnitude()
        errors.append(delta)
    values = np.asarray(errors)
    return {
        "hole_rotation_mean_rad": float(np.mean(values)),
        "hole_rotation_p90_rad": float(np.quantile(values, 0.9)),
    }


def _selection_score(metrics: dict[str, float]) -> float:
    return float(
        metrics["lateral_mean_m"] / 0.0012
        + metrics["tilt_mean_rad"] / 0.03
        + metrics["depth_mae_m"] / 0.003
    )


def _local_mask(target11: np.ndarray) -> np.ndarray:
    state = np.asarray(target11, dtype=np.float64)[:, :5]
    approach_height = -state[:, 4]
    return (
        (approach_height >= 0.015)
        & (approach_height <= 0.085)
        & (np.linalg.norm(state[:, :2], axis=1) <= 0.012)
        & (np.linalg.norm(state[:, 2:4], axis=1) <= 0.25)
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if not args.ridge_alphas or min(args.ridge_alphas) <= 0.0:
        raise ValueError("ridge-alphas must be positive")
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    estimation_dir = args.estimation_dir or args.dataset_root / "retrieval_cerebellum_estimation"
    store = ContinuousVisualFeatureStore.load(args.visual_cache)
    grouped: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for episode in np.unique(store.episode_index):
        frames, features, split = store.episode(int(episode))
        estimation_path = estimation_dir / "episodes" / f"episode_{episode:06d}.parquet"
        episode_features, targets = _episode_data(estimation_path, frames, features)
        grouped[split].append((int(episode), episode_features, targets))
    fit = [item for item in grouped["train"] if item[0] % 5 != 0]
    calibration = [item for item in grouped["train"] if item[0] % 5 == 0]
    if not fit or not calibration:
        raise ValueError("visual cache cannot form fit and calibration train groups")
    fit_x = np.concatenate([item[1] for item in fit])
    fit_y = np.concatenate([item[2] for item in fit])
    calibration_x = np.concatenate([item[1] for item in calibration])
    calibration_y = np.concatenate([item[2] for item in calibration])
    alpha_results = []
    best_model = None
    best_score = np.inf
    for alpha in args.ridge_alphas:
        model = ContinuousVisualStateModel.fit(
            fit_x,
            fit_y,
            alpha=alpha,
            calibration_features=calibration_x,
            calibration_targets11=calibration_y,
        )
        mean5, _, _, _ = model.predict_arrays(calibration_x)
        local = _local_mask(calibration_y)
        metrics = _state_metrics(mean5[local], calibration_y[local, :5])
        score = _selection_score(metrics)
        alpha_results.append({"alpha": alpha, "score": score, "metrics": metrics})
        if score < best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise RuntimeError("failed to select visual model")
    model_path = args.output_dir / "continuous_visual_model.npz"
    best_model.save(model_path)
    split_metrics = {}
    for split_name in ("validation", "test"):
        if not grouped[split_name]:
            split_metrics[split_name] = None
            continue
        features = np.concatenate([item[1] for item in grouped[split_name]])
        targets = np.concatenate([item[2] for item in grouped[split_name]])
        mean5, covariance5, rotations, reliability = best_model.predict_arrays(features)
        global_metrics = _state_metrics(mean5, targets[:, :5])
        global_metrics.update(_rotation_metrics(rotations, targets))
        errors = mean5 - targets[:, :5]
        nse = np.einsum(
            "bi,bij,bj->b",
            errors,
            np.linalg.pinv(covariance5),
            errors,
        )
        global_metrics["mean_state5_nse"] = float(np.mean(nse))
        global_metrics["mean_visual_reliability"] = float(np.mean(reliability))
        local = _local_mask(targets)
        if np.any(local):
            local_metrics = _state_metrics(mean5[local], targets[local, :5])
            local_metrics.update(_rotation_metrics(rotations[local], targets[local]))
            local_metrics["mean_state5_nse"] = float(np.mean(nse[local]))
            local_metrics["mean_visual_reliability"] = float(
                np.mean(reliability[local])
            )
        else:
            local_metrics = None
        split_metrics[split_name] = {
            "global": global_metrics,
            "precision_handoff_window": local_metrics,
        }
    test_metrics = (
        None
        if split_metrics["test"] is None
        else split_metrics["test"]["precision_handoff_window"]
    )
    deployment_gate = None
    if test_metrics is not None:
        checks = {
            "lateral_p90_m": bool(test_metrics["lateral_p90_m"] <= 0.0012),
            "tilt_p90_rad": bool(test_metrics["tilt_p90_rad"] <= 0.03),
            "depth_p90_m": bool(test_metrics["depth_p90_m"] <= 0.003),
            "hole_rotation_p90_rad": bool(
                test_metrics["hole_rotation_p90_rad"] <= 0.03
            ),
        }
        deployment_gate = {
            "approved_for_v2_control": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "lateral_p90_m": 0.0012,
                "tilt_p90_rad": 0.03,
                "depth_p90_m": 0.003,
                "hole_rotation_p90_rad": 0.03,
            },
        }
    summary = {
        "stage": "V2 continuous visual state estimation",
        "visual_cache": str(args.visual_cache),
        "model_path": str(model_path),
        "fit_train_episodes": [item[0] for item in fit],
        "calibration_train_episodes": [item[0] for item in calibration],
        "alpha_selection": alpha_results,
        "selected_alpha": best_model.alpha,
        "split_metrics": split_metrics,
        "deployment_gate": deployment_gate,
        "uses_privileged_features_at_inference": False,
        "teacher_use": "offline targets and external evaluation only",
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
