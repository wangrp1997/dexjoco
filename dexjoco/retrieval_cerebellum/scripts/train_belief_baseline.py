"""Train and evaluate the P3A sensor-history observation plus MHE baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from retrieval_cerebellum.belief_estimation import (
    RidgeObservationModel,
    SlidingWindowMHE,
    belief_metrics,
    belief_target,
    covariance_calibration,
    default_process_variance,
    sensor_feature,
    stack_causal_history,
)
from retrieval_cerebellum.finger_kinematics import AllegroFingertipKinematics
from retrieval_cerebellum.real_sensor_model import SensorDegrader, SensorModelConfig
from retrieval_cerebellum.sensor_observation import CerebellumSensorObservation
from retrieval_cerebellum.visual_initialization import EpisodeVisualFeatureStore


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")
DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "sensor_profiles"
    / "sim_stress_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--estimation-dir", type=Path, default=None)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/belief_baseline"),
    )
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--include-fingertip-kinematics", action="store_true")
    parser.add_argument("--visual-cache", type=Path, default=None)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--mhe-window", type=int, default=8)
    parser.add_argument("--process-position-std-m", type=float, default=0.002)
    parser.add_argument("--process-rotation-std-rad", type=float, default=0.02)
    parser.add_argument("--position-target-m", type=float, default=0.005)
    parser.add_argument("--rotation-target-rad", type=float, default=0.0872664626)
    return parser.parse_args()


def _array(table, name: str, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(table[name].to_pylist(), dtype=np.float32)
    expected = (table.num_rows, *shape)
    if values.shape != expected:
        raise ValueError(f"{name} expected shape {expected}, got {values.shape}")
    return values


def _episode_data(
    path: Path,
    profile: SensorModelConfig,
    *,
    history_size: int,
    kinematics: AllegroFingertipKinematics | None,
    visual_store: EpisodeVisualFeatureStore | None,
) -> tuple[str, np.ndarray, np.ndarray]:
    import pyarrow.parquet as parquet

    columns = [
        "split",
        "dataset_timestamp_s",
        "sensor_state46",
        "sensor_previous_action44",
        "sensor_arm_joint_torque",
        "sensor_fingertip_force_world",
        "sensor_wrist_wrench_world",
        "teacher_peg_in_hole_position",
        "teacher_peg_in_hole_rotvec",
        "teacher_peg_in_right_palm_position",
        "teacher_peg_in_right_palm_rotvec",
        "teacher_tray_in_left_palm_position",
        "teacher_tray_in_left_palm_rotvec",
    ]
    table = parquet.read_table(path, columns=columns)
    split_values = set(str(value) for value in table["split"].to_pylist())
    if len(split_values) != 1:
        raise ValueError(f"{path} has non-constant split")
    split = next(iter(split_values))
    state = _array(table, "sensor_state46", (46,))
    action = _array(table, "sensor_previous_action44", (44,))
    arm = _array(table, "sensor_arm_joint_torque", (14,)).reshape(-1, 2, 7)
    fingertip = _array(
        table,
        "sensor_fingertip_force_world",
        (24,),
    ).reshape(-1, 2, 4, 3)
    wrist = _array(table, "sensor_wrist_wrench_world", (12,)).reshape(-1, 2, 6)
    timestamp = np.asarray(table["dataset_timestamp_s"].to_numpy(), dtype=np.float64)
    target = belief_target(
        {
            name: np.asarray(table[name].to_pylist(), dtype=np.float32)
            for name in columns
            if name.startswith("teacher_")
        }
    )
    episode_index = int(path.stem.rsplit("_", 1)[-1])
    degrader = SensorDegrader(
        profile,
        random_seed=profile.random_seed + episode_index,
    )
    features: list[np.ndarray] = []
    aligned_targets: list[np.ndarray] = []
    for row in range(table.num_rows):
        degraded = degrader.transform(
            CerebellumSensorObservation(
                timestamp_s=float(timestamp[row]),
                state46=state[row],
                arm_joint_torque=arm[row],
                fingertip_force_world=fingertip[row],
                wrist_wrench_world=wrist[row],
                images={},
                previous_action44=action[row],
            )
        )
        if degraded is None:
            continue
        target_row = row - profile.latency_frames
        fingertip_position = (
            None
            if kinematics is None
            else kinematics.positions_in_palm(degraded.state46)
        )
        features.append(
            sensor_feature(
                degraded,
                fingertip_position_in_palm=fingertip_position,
            )
        )
        aligned_targets.append(target[target_row])
    feature_matrix = stack_causal_history(np.asarray(features), history_size)
    if visual_store is not None:
        visual_feature = visual_store.feature_for(episode_index, split)
        repeated_visual = np.broadcast_to(
            visual_feature,
            (feature_matrix.shape[0], visual_feature.size),
        )
        feature_matrix = np.concatenate(
            [feature_matrix, repeated_visual],
            axis=1,
        ).astype(np.float32)
    target_matrix = np.asarray(aligned_targets, dtype=np.float32)
    if feature_matrix.shape[0] != target_matrix.shape[0]:
        raise RuntimeError(f"{path} feature and target row counts differ")
    return split, feature_matrix, target_matrix


def _evaluate_split(
    episodes: list[tuple[np.ndarray, np.ndarray]],
    model: RidgeObservationModel,
    mhe: SlidingWindowMHE,
) -> dict[str, object]:
    observation_predictions = []
    mhe_predictions = []
    targets = []
    mhe_variances = []
    for features, target in episodes:
        observation = model.predict(features)
        result = mhe.smooth(observation)
        observation_predictions.append(observation)
        mhe_predictions.append(result.mean)
        mhe_variances.append(result.variance)
        targets.append(target)
    observation_prediction = np.concatenate(observation_predictions)
    mhe_prediction = np.concatenate(mhe_predictions)
    target = np.concatenate(targets)
    variance = np.concatenate(mhe_variances)
    observation_variance = np.broadcast_to(
        model.residual_variance,
        observation_prediction.shape,
    )
    return {
        "num_episodes": len(episodes),
        "num_frames": int(target.shape[0]),
        "observation": {
            "metrics": belief_metrics(observation_prediction, target),
            "calibration": covariance_calibration(
                observation_prediction,
                target,
                observation_variance,
            ),
        },
        "mhe": {
            "metrics": belief_metrics(mhe_prediction, target),
            "calibration": covariance_calibration(
                mhe_prediction,
                target,
                variance,
            ),
        },
    }


def main() -> None:
    args = parse_args()
    estimation_dir = args.estimation_dir or args.dataset_root / "retrieval_cerebellum_estimation"
    profile = SensorModelConfig.from_json(args.profile)
    paths = sorted((estimation_dir / "episodes").glob("episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No estimation shards under {estimation_dir}")

    episodes: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    kinematics = (
        AllegroFingertipKinematics()
        if args.include_fingertip_kinematics
        else None
    )
    visual_store = (
        None
        if args.visual_cache is None
        else EpisodeVisualFeatureStore.load(args.visual_cache)
    )
    for progress, path in enumerate(paths, start=1):
        split, features, target = _episode_data(
            path,
            profile,
            history_size=args.history_size,
            kinematics=kinematics,
            visual_store=visual_store,
        )
        episodes[split].append((features, target))
        print(
            f"[{progress}/{len(paths)}] {path.stem} split={split} rows={len(target)}",
            flush=True,
        )
    if not episodes["train"] or not episodes["validation"] or not episodes["test"]:
        raise ValueError("train, validation and test splits must all be non-empty")

    train_features = np.concatenate([item[0] for item in episodes["train"]])
    train_targets = np.concatenate([item[1] for item in episodes["train"]])
    model = RidgeObservationModel.fit(
        train_features,
        train_targets,
        history_size=args.history_size,
        alpha=args.ridge_alpha,
    )
    process_variance = default_process_variance(
        position_std_m=args.process_position_std_m,
        rotation_std_rad=args.process_rotation_std_rad,
    )
    mhe = SlidingWindowMHE(
        model.residual_variance,
        process_variance,
        window_size=args.mhe_window,
    )
    evaluations = {
        split: _evaluate_split(split_episodes, model, mhe)
        for split, split_episodes in episodes.items()
    }
    test_mhe = evaluations["test"]["mhe"]["metrics"]
    feasibility = {
        name: {
            "position_target_met": bool(
                test_mhe[name]["position_mean_m"] <= args.position_target_m
            ),
            "rotation_target_met": bool(
                test_mhe[name]["rotation_mean_rad"] <= args.rotation_target_rad
            ),
        }
        for name in (
            "peg_in_hole",
            "peg_in_right_palm",
            "tray_in_left_palm",
        )
    }
    summary = {
        "created_date": "2026-08-21",
        "estimation_dir": str(estimation_dir),
        "profile": profile.to_dict(),
        "profile_hardware_verified": profile.hardware_verified,
        "feature_contract": {
            "history_size": args.history_size,
            "per_frame_dim": 143 + (24 if args.include_fingertip_kinematics else 0),
            "history_stacked_dim": (
                143 + (24 if args.include_fingertip_kinematics else 0)
            )
            * args.history_size,
            "stacked_dim": int(train_features.shape[1]),
            "includes_fingertip_kinematics": args.include_fingertip_kinematics,
            "uses_images": visual_store is not None,
            "visual_feature_dim": 0 if visual_store is None else visual_store.feature_dim,
            "visual_cache": None if args.visual_cache is None else str(args.visual_cache),
            "uses_teacher_at_inference": False,
            "uses_world_fingertip_force_at_inference": False,
        },
        "model": {
            "type": "standardized_multi_output_ridge",
            "alpha": args.ridge_alpha,
            "train_rows": int(train_features.shape[0]),
        },
        "mhe": {
            "window_size": args.mhe_window,
            "process_position_std_m": args.process_position_std_m,
            "process_rotation_std_rad": args.process_rotation_std_rad,
        },
        "evaluations": evaluations,
        "provisional_control_targets": {
            "position_mean_m": args.position_target_m,
            "rotation_mean_rad": args.rotation_target_rad,
        },
        "test_mhe_feasibility": feasibility,
        "limitations": [
            "simulation-only unverified sensor profile",
            "episode-start image prior only; no per-frame visual tracking"
            if visual_store is not None
            else "no image features",
            "no tactile contact location or normal",
            "linear observation model",
            "random-walk MHE without rigid contact factors",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "observation_model.npz")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
