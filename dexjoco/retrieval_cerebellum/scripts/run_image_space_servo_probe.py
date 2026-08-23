"""Probe and evaluate RGB-only local image-space visual servoing in MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import restore_initial_state
from dexquery.data.action_utils import policy_dual_arm_to_raw, rotvec_dual_arm_to_policy
from dexquery.data.episode_replay import make_assembly_env
from dexquery.data.zarr_io import discover_zarr_demos, load_zarr_episode
from retrieval_cerebellum.assembly_kinematics import apply_bimanual_wrist_twists
from retrieval_cerebellum.geometry_labels import PrivilegedGeometryLabeler
from retrieval_cerebellum.image_space_servo import (
    EgoSpatialVisualEstimator,
    damped_servo_command,
    identify_central_difference_jacobian,
)


DEFAULT_ZARR = Path("/mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly")
DEFAULT_SUPERVISION = Path(
    "/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly/"
    "retrieval_cerebellum_spatial_visual"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-input-dir", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--supervision-dir", type=Path, default=DEFAULT_SUPERVISION)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/spatial_visual_model_full/best_model.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=17)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--settle-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _relative_twist(control4: np.ndarray) -> np.ndarray:
    control = np.asarray(control4, dtype=np.float64).reshape(4)
    return np.asarray([control[0], control[1], 0.0, control[2], control[3], 0.0])


def _perturbed_action(base_action44: np.ndarray, control4: np.ndarray) -> np.ndarray:
    relative = _relative_twist(control4)
    return apply_bimanual_wrist_twists(
        base_action44,
        0.8 * relative,
        -0.2 * relative,
        finger_reference44=base_action44,
    )


def _step(raw_env, action44: np.ndarray) -> None:
    raw_env.step(policy_dual_arm_to_raw(rotvec_dual_arm_to_policy(action44)))


def _run_branch(
    *,
    actions44: np.ndarray,
    initial_state: np.ndarray,
    episode: int,
    frame: int,
    command4: np.ndarray,
    settle_steps: int,
    estimator: EgoSpatialVisualEstimator,
) -> dict[str, object]:
    env = make_assembly_env(seed=episode, randomize=False)
    raw_env = env.unwrapped
    labeler = PrivilegedGeometryLabeler(raw_env)
    try:
        env.reset()
        restore_initial_state(
            env,
            "bimanual_assembly",
            CONFIG_MAPPING["bimanual_assembly"](),
            initial_state,
        )
        labeler.reset_reference(raw_env)
        for action44 in actions44[: frame + 1]:
            _step(raw_env, action44)
        base_action = actions44[frame]
        action = _perturbed_action(base_action, command4)
        for _ in range(settle_steps):
            _step(raw_env, action)
        observation = raw_env._compute_observation()
        visual = estimator.predict(observation["images"]["ego"])
        truth = labeler.compute(raw_env)
        return {
            "feature4": visual.feature4,
            "feature_norm": float(np.linalg.norm(visual.feature4)),
            "reliability": visual.reliability,
            "keypoints_uv": visual.keypoints_uv,
            "lateral_error_m": float(truth.lateral_error_m),
            "axis_error_rad": float(truth.axis_error_rad),
            "approach_height_m": float(truth.approach_height_m),
            "insertion_depth_m": float(truth.insertion_depth_m),
        }
    finally:
        env.close()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> None:
    args = parse_args()
    sidecar_path = args.supervision_dir / "episodes" / f"episode_{args.episode:06d}.npz"
    with np.load(sidecar_path, allow_pickle=False) as sidecar:
        available_frames = np.asarray(sidecar["frame_index"], dtype=np.int64)
    frame = int(available_frames[0] if args.frame is None else args.frame)
    if frame not in set(available_frames.tolist()):
        raise ValueError(f"frame {frame} is not in the spatial supervision window")
    zarr_paths = discover_zarr_demos(args.zarr_input_dir)
    actions44, initial_state = load_zarr_episode(zarr_paths[args.episode])
    if initial_state is None:
        raise ValueError("accurate replay requires initial_state")
    estimator = EgoSpatialVisualEstimator(args.checkpoint, device=args.device)
    baseline = _run_branch(
        actions44=actions44,
        initial_state=initial_state,
        episode=args.episode,
        frame=frame,
        command4=np.zeros(4),
        settle_steps=args.settle_steps,
        estimator=estimator,
    )
    amplitudes = np.asarray([0.001, 0.001, 0.008, 0.008], dtype=np.float64)
    positive = []
    negative = []
    probes = []
    for index, amplitude in enumerate(amplitudes):
        direction = np.zeros(4, dtype=np.float64)
        direction[index] = amplitude
        plus = _run_branch(
            actions44=actions44,
            initial_state=initial_state,
            episode=args.episode,
            frame=frame,
            command4=direction,
            settle_steps=args.settle_steps,
            estimator=estimator,
        )
        minus = _run_branch(
            actions44=actions44,
            initial_state=initial_state,
            episode=args.episode,
            frame=frame,
            command4=-direction,
            settle_steps=args.settle_steps,
            estimator=estimator,
        )
        positive.append(plus["feature4"])
        negative.append(minus["feature4"])
        probes.append({"axis": index, "amplitude": amplitude, "plus": plus, "minus": minus})
    jacobian = identify_central_difference_jacobian(
        np.asarray(positive),
        np.asarray(negative),
        amplitudes,
        minimum_singular_value=2e-3,
    )
    approved = bool(
        jacobian.rank == 4
        and np.isfinite(jacobian.condition_number)
        and jacobian.condition_number <= 100.0
        and baseline["reliability"] >= 0.35
    )
    correction = None
    corrected = None
    alignment_matrix = jacobian.matrix[:2, :2]
    alignment_singular_values = np.linalg.svd(alignment_matrix, compute_uv=False)
    alignment_condition_number = float(
        alignment_singular_values[0] / alignment_singular_values[-1]
    )
    approved_for_tip_alignment = bool(
        alignment_singular_values[-1] >= 2e-3
        and alignment_condition_number <= 100.0
        and baseline["reliability"] >= 0.35
    )
    if approved:
        correction = damped_servo_command(
            baseline["feature4"],
            jacobian,
            damping=0.03,
            gain=0.7,
            limits=np.asarray([0.002, 0.002, 0.015, 0.015]),
        )
        corrected = _run_branch(
            actions44=actions44,
            initial_state=initial_state,
            episode=args.episode,
            frame=frame,
            command4=correction,
            settle_steps=args.settle_steps,
            estimator=estimator,
        )
    elif approved_for_tip_alignment:
        tip_error = np.asarray(baseline["feature4"], dtype=np.float64)[:2]
        damping = 0.03
        translation = -0.7 * alignment_matrix.T @ np.linalg.solve(
            alignment_matrix @ alignment_matrix.T + damping**2 * np.eye(2),
            tip_error,
        )
        correction = np.zeros(4, dtype=np.float64)
        correction[:2] = np.clip(translation, -0.002, 0.002)
        corrected = _run_branch(
            actions44=actions44,
            initial_state=initial_state,
            episode=args.episode,
            frame=frame,
            command4=correction,
            settle_steps=args.settle_steps,
            estimator=estimator,
        )
    payload = {
        "stage": "V2 RGB-only local image-space servo probe",
        "episode": args.episode,
        "frame": frame,
        "online_inputs": ["ego_rgb", "executed_probe_actions"],
        "teacher_used_by_controller": False,
        "teacher_used_for_external_evaluation_only": True,
        "baseline": baseline,
        "probe_amplitudes": amplitudes,
        "probes": probes,
        "jacobian": {
            "matrix": jacobian.matrix,
            "singular_values": jacobian.singular_values,
            "rank": jacobian.rank,
            "condition_number": jacobian.condition_number,
        },
        "approved_for_correction": approved,
        "tip_alignment_jacobian": {
            "matrix": alignment_matrix,
            "singular_values": alignment_singular_values,
            "condition_number": alignment_condition_number,
        },
        "approved_for_tip_alignment": approved_for_tip_alignment,
        "correction4": correction,
        "corrected": corrected,
        "image_error_reduced": bool(
            corrected is not None and corrected["feature_norm"] < baseline["feature_norm"]
        ),
        "external_lateral_error_reduced": bool(
            corrected is not None
            and corrected["lateral_error_m"] < baseline["lateral_error_m"]
        ),
        "externally_validated_tip_alignment": bool(
            corrected is not None
            and corrected["feature_norm"] < baseline["feature_norm"]
            and corrected["lateral_error_m"] < baseline["lateral_error_m"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable))


if __name__ == "__main__":
    main()
