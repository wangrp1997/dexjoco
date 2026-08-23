"""Train a deployable local belief model and export handoff estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from retrieval_cerebellum.local_belief_estimation import (
    LocalAssemblyBeliefModel,
    local_target,
)
from retrieval_cerebellum.real_sensor_model import SensorModelConfig
from retrieval_cerebellum.scripts.train_belief_baseline import _episode_data
from retrieval_cerebellum.visual_initialization import EpisodeVisualFeatureStore


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")
DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "sensor_profiles"
    / "sim_stress_v1.json"
)
DEFAULT_VISUAL = Path(
    "outputs/retrieval_cerebellum/visual_initialization/clip_vit_b16_pca4.npz"
)
DEFAULT_HANDOFF_PLAN = Path(
    "outputs/retrieval_cerebellum/belief_space_sqp_bilateral_handoff_validation10"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--estimation-dir", type=Path, default=None)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--visual-cache", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--handoff-plan-dir", type=Path, default=DEFAULT_HANDOFF_PLAN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/local_belief"),
    )
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=(100.0, 1000.0, 10000.0, 100000.0),
    )
    parser.add_argument("--minimum-approach-height-m", type=float, default=0.015)
    parser.add_argument("--maximum-approach-height-m", type=float, default=0.085)
    parser.add_argument("--maximum-lateral-error-m", type=float, default=0.012)
    parser.add_argument("--maximum-tilt-error-rad", type=float, default=0.25)
    parser.add_argument("--radial-clearance-m", type=float, default=0.0024)
    parser.add_argument("--deployment-lateral-p90-fraction", type=float, default=0.5)
    parser.add_argument("--deployment-tilt-p90-rad", type=float, default=0.03)
    parser.add_argument("--deployment-depth-p90-m", type=float, default=0.003)
    parser.add_argument("--deployment-maximum-mean-nse", type=float, default=7.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _local_mask(
    target23: np.ndarray,
    *,
    minimum_height: float,
    maximum_height: float,
    maximum_lateral: float,
    maximum_tilt: float,
) -> np.ndarray:
    values = np.asarray(target23, dtype=np.float64)
    state = values[:, 18:]
    approach_height = -state[:, 4]
    return (
        (approach_height >= minimum_height)
        & (approach_height <= maximum_height)
        & (np.linalg.norm(state[:, :2], axis=1) <= maximum_lateral)
        & (np.linalg.norm(state[:, 2:4], axis=1) <= maximum_tilt)
    )


def _state_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = np.asarray(prediction)[:, 18:] - np.asarray(target)[:, 18:]
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


def _selection_score(metrics: dict) -> float:
    return float(
        metrics["lateral_mean_m"] / 0.0024
        + metrics["tilt_mean_rad"] / 0.12
        + metrics["depth_mae_m"] / 0.005
    )


def main() -> None:
    args = parse_args()
    if args.history_size <= 0:
        raise ValueError("history_size must be positive")
    if not args.ridge_alphas or min(args.ridge_alphas) <= 0.0:
        raise ValueError("ridge_alphas must be positive")
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    estimation_dir = (
        args.estimation_dir
        or args.dataset_root / "retrieval_cerebellum_estimation"
    )
    profile = SensorModelConfig.from_json(args.profile)
    visual_store = EpisodeVisualFeatureStore.load(args.visual_cache)
    episodes: dict[int, dict] = {}
    grouped: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for path in sorted((estimation_dir / "episodes").glob("episode_*.parquet")):
        episode_index = int(path.stem.rsplit("_", 1)[-1])
        split, features, belief18 = _episode_data(
            path,
            profile,
            history_size=args.history_size,
            kinematics=None,
            visual_store=visual_store,
        )
        target23 = local_target(belief18)
        mask = _local_mask(
            target23,
            minimum_height=args.minimum_approach_height_m,
            maximum_height=args.maximum_approach_height_m,
            maximum_lateral=args.maximum_lateral_error_m,
            maximum_tilt=args.maximum_tilt_error_rad,
        )
        grouped[split].append((episode_index, features[mask], target23[mask]))
        episodes[episode_index] = {
            "split": split,
            "features": features,
            "target23": target23,
        }

    fit_episodes = [item for item in grouped["train"] if item[0] % 5 != 0]
    calibration_episodes = [item for item in grouped["train"] if item[0] % 5 == 0]
    if not fit_episodes or not calibration_episodes:
        raise ValueError("train split cannot form fit and calibration episode groups")
    fit_x = np.concatenate([item[1] for item in fit_episodes])
    fit_y = np.concatenate([item[2] for item in fit_episodes])
    calibration_x = np.concatenate([item[1] for item in calibration_episodes])
    calibration_y = np.concatenate([item[2] for item in calibration_episodes])

    alpha_results = []
    best_alpha = None
    best_score = np.inf
    for alpha in args.ridge_alphas:
        model = LocalAssemblyBeliefModel.fit(
            fit_x,
            fit_y,
            history_size=args.history_size,
            alpha=alpha,
            calibration_features=calibration_x,
            calibration_targets=calibration_y,
        )
        prediction, _ = model.predict(calibration_x)
        metrics = _state_metrics(prediction, calibration_y)
        score = _selection_score(metrics)
        alpha_results.append({"alpha": alpha, "score": score, "metrics": metrics})
        if score < best_score:
            best_score = score
            best_alpha = alpha
    if best_alpha is None:
        raise RuntimeError("failed to select ridge alpha")

    model = LocalAssemblyBeliefModel.fit(
        fit_x,
        fit_y,
        history_size=args.history_size,
        alpha=best_alpha,
        calibration_features=calibration_x,
        calibration_targets=calibration_y,
    )
    model_path = args.output_dir / "local_belief_model.npz"
    model.save(model_path)

    split_metrics = {}
    for split in ("validation", "test"):
        split_x = np.concatenate([item[1] for item in grouped[split]])
        split_y = np.concatenate([item[2] for item in grouped[split]])
        prediction, covariance = model.predict(split_x)
        metrics = _state_metrics(prediction, split_y)
        state_error = prediction[:, 18:] - split_y[:, 18:]
        state_covariance = covariance[:, 18:, 18:]
        normalized_squared_error = np.einsum(
            "bi,bij,bj->b",
            state_error,
            np.linalg.pinv(state_covariance),
            state_error,
        )
        metrics["mean_state5_nse"] = float(np.mean(normalized_squared_error))
        split_metrics[split] = metrics

    test_metrics = split_metrics["test"]
    deployment_thresholds = {
        "lateral_p90_m": (
            args.radial_clearance_m * args.deployment_lateral_p90_fraction
        ),
        "tilt_p90_rad": args.deployment_tilt_p90_rad,
        "depth_p90_m": args.deployment_depth_p90_m,
        "mean_state5_nse": args.deployment_maximum_mean_nse,
    }
    deployment_checks = {
        name: bool(test_metrics[name] <= threshold)
        for name, threshold in deployment_thresholds.items()
    }
    approved_for_sqp = all(deployment_checks.values())

    handoff_summary = json.loads((args.handoff_plan_dir / "summary.json").read_text())
    handoff_records = []
    for record in handoff_summary["records"]:
        handoff_row = record.get("handoff_row")
        if handoff_row is None:
            handoff_records.append(
                {
                    "episode_index": int(record["episode_index"]),
                    "split": record["split"],
                    "handoff_row": None,
                    "available": False,
                    "reason": "handoff_unavailable",
                }
            )
            continue
        episode_index = int(record["episode_index"])
        episode = episodes[episode_index]
        row = int(handoff_row)
        mean23, covariance23 = model.predict(episode["features"][row : row + 1])
        mean23 = mean23[0]
        covariance23 = covariance23[0]
        belief18 = model.belief18(mean23).copy()
        state5 = model.state5(mean23)
        belief18[0] = state5[0]
        belief18[1] = state5[1]
        belief18[2] = -state5[4]
        belief18[3] = state5[2]
        belief18[4] = state5[3]
        state_covariance = model.state5_covariance(covariance23)
        truth_state5 = episode["target23"][row, 18:]
        error5 = state5 - truth_state5
        stem = args.output_dir / "episodes" / f"episode_{episode_index:06d}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            stem.with_suffix(".npz"),
            handoff_row=np.asarray(row, dtype=np.int64),
            belief18=belief18,
            state5=state5,
            state5_covariance=state_covariance,
        )
        episode_record = {
            "episode_index": episode_index,
            "split": episode["split"],
            "handoff_row": row,
            "available": True,
            "state5": state5.tolist(),
            "state5_covariance_diagonal": np.diag(state_covariance).tolist(),
            "teacher_state5": truth_state5.tolist(),
            "error5": error5.tolist(),
            "lateral_error_m": float(np.linalg.norm(error5[:2])),
            "tilt_error_rad": float(np.linalg.norm(error5[2:4])),
            "depth_error_m": float(abs(error5[4])),
            "estimate_path": str(stem.with_suffix(".npz")),
        }
        _write_json(stem.with_suffix(".json"), episode_record)
        handoff_records.append(episode_record)

    summary = {
        "stage": "P5A deployable local belief estimation",
        "model_path": str(model_path),
        "sensor_profile": profile.to_dict(),
        "visual_cache": str(args.visual_cache),
        "history_size": args.history_size,
        "fit_train_episodes": [item[0] for item in fit_episodes],
        "calibration_train_episodes": [item[0] for item in calibration_episodes],
        "alpha_selection": alpha_results,
        "selected_alpha": best_alpha,
        "split_metrics": split_metrics,
        "deployment_gate": {
            "approved_for_sqp": approved_for_sqp,
            "radial_clearance_m": args.radial_clearance_m,
            "thresholds": deployment_thresholds,
            "checks": deployment_checks,
            "reason": (
                "all held-out precision thresholds passed"
                if approved_for_sqp
                else "held-out belief error exceeds the insertion-scale safety budget"
            ),
        },
        "handoff_records": handoff_records,
        "uses_privileged_features_at_inference": False,
        "uses_oracle_handoff_time_for_current_evaluation": True,
        "limitations": [
            "teacher geometry is used only for offline training and evaluation",
            "the current visual feature is a fixed first-frame episode embedding",
            "the estimator is a local linear baseline rather than contact-factor MHE",
            "SQP integration is prohibited unless deployment_gate.approved_for_sqp is true",
        ],
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
