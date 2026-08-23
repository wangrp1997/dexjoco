"""Run an RGB-only SEARCH→ALIGN→INSERT prototype from a natural demo state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import restore_initial_state
from dexquery.data.action_utils import policy_dual_arm_to_raw, rotvec_dual_arm_to_policy
from dexquery.data.episode_replay import make_assembly_env
from dexquery.data.zarr_io import discover_zarr_demos, load_zarr_episode
from retrieval_cerebellum.assembly_kinematics import apply_bimanual_wrist_twists
from retrieval_cerebellum.geometry_labels import PrivilegedGeometryLabeler
from retrieval_cerebellum.learning_data import state46_to_action44
from retrieval_cerebellum.rgb_search_align import (
    RGBSearchAlignEstimator,
    damped_visual_command,
)
from retrieval_cerebellum.sim_sensor_adapter import SimCerebellumSensorAdapter


DEFAULT_ZARR = Path("/mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-input-dir", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--episode", type=int, default=31)
    parser.add_argument("--start-frame", type=int, default=250)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--align-steps", type=int, default=80)
    parser.add_argument("--insert-steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def _step(raw_env, action44: np.ndarray) -> None:
    raw_env.step(policy_dual_arm_to_raw(rotvec_dual_arm_to_policy(action44)))


def _state46(raw_env) -> np.ndarray:
    state = raw_env._compute_observation()["state"]
    return np.concatenate(
        [np.asarray(state["tcp_pose"]).ravel(), np.asarray(state["gripper_pose"]).ravel()]
    )


def _ego(raw_env) -> np.ndarray:
    return np.asarray(raw_env._compute_observation()["images"]["ego"], dtype=np.uint8)


def _annotated_frame(
    image: np.ndarray,
    feature,
    *,
    mode: str,
    step: int,
    force_n: float,
) -> np.ndarray:
    frame = image.copy()
    if feature is not None:
        peg = tuple(np.round(feature.peg_tip_uv).astype(int))
        hole = tuple(np.round(feature.hole_uv).astype(int))
        cv2.circle(frame, peg, 7, (255, 255, 0), 2)
        cv2.circle(frame, hole, 9, (0, 255, 255), 2)
        cv2.line(frame, peg, hole, (255, 255, 255), 2)
        error = feature.error3
        detail = f"du={error[0]:.1f} dv={error[1]:.1f} da={error[2]:.3f}"
    else:
        detail = "target not visible"
    cv2.putText(
        frame,
        f"{mode} step={step} force={force_n:.2f}N",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        detail,
        (12, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _bounded(command: np.ndarray) -> np.ndarray:
    result = np.asarray(command, dtype=np.float64).copy()
    translation = np.linalg.norm(result[:3])
    if translation > 0.004:
        result[:3] *= 0.004 / translation
    rotation = np.linalg.norm(result[3:])
    if rotation > 0.04:
        result[3:] *= 0.04 / rotation
    return result


def main() -> None:
    args = parse_args()
    paths = discover_zarr_demos(args.zarr_input_dir)
    actions44, initial_state = load_zarr_episode(paths[args.episode])
    if initial_state is None:
        raise ValueError("episode has no initial state")
    env = make_assembly_env(seed=args.episode, randomize=False, render_mode="rgb_array")
    raw_env = env.unwrapped
    estimator = RGBSearchAlignEstimator()
    sensor_adapter = SimCerebellumSensorAdapter(raw_env)
    evaluator = PrivilegedGeometryLabeler(raw_env)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.video, fps=args.fps)
    trace = []
    try:
        env.reset()
        restore_initial_state(
            env,
            "bimanual_assembly",
            CONFIG_MAPPING["bimanual_assembly"](),
            initial_state,
        )
        for action in actions44[: args.start_frame + 1]:
            _step(raw_env, action)
        evaluator.reset_reference(raw_env)
        baseline = sensor_adapter.capture({"state": _state46(raw_env)})
        baseline_wrench = np.asarray(baseline.wrist_wrench_world, dtype=np.float64)

        feature = estimator.estimate(_ego(raw_env))
        for search_step in range(3):
            current = state46_to_action44(_state46(raw_env))
            _step(raw_env, current)
            feature = estimator.estimate(_ego(raw_env))
            writer.append_data(
                _annotated_frame(
                    _ego(raw_env),
                    feature,
                    mode="SEARCH",
                    step=search_step,
                    force_n=0.0,
                )
            )
        if feature is None:
            raise RuntimeError("RGB SEARCH could not find peg and socket")

        probe = np.asarray([0.003, 0.003, 0.003, 0.03, 0.03, 0.03])
        jacobian = np.zeros((3, 6), dtype=np.float64)
        for dimension in range(6):
            before = estimator.estimate(_ego(raw_env))
            if before is None:
                raise RuntimeError("target lost during visual Jacobian identification")
            current = state46_to_action44(_state46(raw_env))
            twist = np.zeros(6)
            twist[dimension] = probe[dimension]
            _step(
                raw_env,
                apply_bimanual_wrist_twists(
                    current,
                    twist,
                    np.zeros(6),
                    finger_reference44=current,
                ),
            )
            after = estimator.estimate(_ego(raw_env))
            if after is None:
                raise RuntimeError("target lost during visual Jacobian probe")
            jacobian[:, dimension] = (after.error3 - before.error3) / probe[dimension]
            current = state46_to_action44(_state46(raw_env))
            _step(
                raw_env,
                apply_bimanual_wrist_twists(
                    current,
                    -twist,
                    np.zeros(6),
                    finger_reference44=current,
                ),
            )

        mode = "ALIGN"
        success = False
        exit_reason = "timeout"
        total_step = 0
        for _ in range(args.align_steps + args.insert_steps):
            total_step += 1
            image = _ego(raw_env)
            feature = estimator.estimate(image)
            if feature is None:
                exit_reason = "search_failure"
                break
            state46 = _state46(raw_env)
            sensor = sensor_adapter.capture({"state": state46})
            residual = np.asarray(sensor.wrist_wrench_world) - baseline_wrench
            force_n = float(np.max(np.linalg.norm(residual[:, :3], axis=1)))
            if force_n >= 20.0:
                exit_reason = "safety_violation"
                break
            target_error = feature.error3 - np.asarray([0.0, -7.0, 0.0])
            aligned = bool(
                np.linalg.norm(target_error[:2]) <= 10.0
                and abs(target_error[2]) <= 0.12
            )
            if mode == "ALIGN" and aligned:
                mode = "INSERT"
            command = damped_visual_command(jacobian, target_error, damping=6.0)
            command = _bounded(0.55 * command)
            if mode == "INSERT":
                command[2] -= 0.0008
                command[:2] *= 0.35
                command[3:] *= 0.35
            current = state46_to_action44(state46)
            action = apply_bimanual_wrist_twists(
                current,
                command,
                np.zeros(6),
                finger_reference44=current,
            )
            _step(raw_env, action)
            truth = evaluator.compute(raw_env)
            writer.append_data(
                _annotated_frame(
                    _ego(raw_env),
                    feature,
                    mode=mode,
                    step=total_step,
                    force_n=force_n,
                )
            )
            trace.append(
                {
                    "step": total_step,
                    "mode": mode,
                    "image_error": target_error.tolist(),
                    "confidence": feature.confidence,
                    "force_n": force_n,
                    "lateral_error_m": float(truth.lateral_error_m),
                    "axis_error_rad": float(truth.axis_error_rad),
                    "approach_height_m": float(truth.approach_height_m),
                    "insert_ok": bool(truth.insert_ok),
                }
            )
            if truth.insert_ok:
                success = True
                exit_reason = "external_success_evaluator"
                break
            if mode == "ALIGN" and total_step >= args.align_steps:
                exit_reason = "alignment_failure"
                break
        payload = {
            "stage": "RGB SEARCH-ALIGN-INSERT prototype",
            "episode": args.episode,
            "start_frame": args.start_frame,
            "controller_online_inputs": ["ego_rgb", "proprioception", "wrist_wrenches"],
            "teacher_used_by_controller": False,
            "teacher_used_for_external_initialization": True,
            "success": success,
            "exit_reason": exit_reason,
            "steps": len(trace),
            "trace": trace,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({key: value for key, value in payload.items() if key != "trace"}, indent=2))
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
