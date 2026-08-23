"""Evaluate plain sensor-only guarded descent on natural held-out handoffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import yaml
from openpi_client import image_tools, websocket_client_policy

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import restore_initial_state
from dexquery.data.action_utils import policy_dual_arm_to_raw, rotvec_dual_arm_to_policy
from dexquery.data.episode_replay import make_assembly_env
from dexquery.data.zarr_io import discover_zarr_demos, load_zarr_episode
from retrieval_cerebellum.assembly_kinematics import (
    apply_bimanual_wrist_twists,
    pose_from_action44,
    world_wrist_twist,
)
from retrieval_cerebellum.geometry_labels import PrivilegedGeometryLabeler
from retrieval_cerebellum.learning_data import state46_to_action44
from retrieval_cerebellum.intent_chunk_execution import (
    IntentChunkExecutionConfig,
    OnlineIntentChunkExecutor,
)
from retrieval_cerebellum.sim_sensor_adapter import SimCerebellumSensorAdapter


DEFAULT_ZARR = Path("/mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly")
DEFAULT_HANDOFFS = Path(
    "outputs/retrieval_cerebellum/local_dynamics_validation10/summary.json"
)
DEFAULT_OPENPI_CONFIG = Path("configs/multi_task/bimanual_assembly.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-input-dir", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--handoff-summary", type=Path, default=DEFAULT_HANDOFFS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--descent-step-mm", type=float, default=0.5)
    parser.add_argument("--intent-horizon", type=int, default=12)
    parser.add_argument("--hard-force-limit-n", type=float, default=20.0)
    parser.add_argument("--openpi-config", type=Path, default=DEFAULT_OPENPI_CONFIG)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument(
        "--enforce-fingertip-grasp-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--intent-mode",
        choices=("constant_axis", "recorded_delta_trajectory", "online_pi05_chunk"),
        default="constant_axis",
    )
    parser.add_argument(
        "--contact-response",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _policy_observation(raw_env, config: dict) -> dict[str, object]:
    raw_observation = raw_env._compute_observation()
    images = raw_observation["images"]
    observation = {
        policy_key: image_tools.convert_to_uint8(
            image_tools.resize_with_pad(images[env_key], 224, 224)
        )
        for policy_key, env_key in config["camera_mapping"].items()
    }
    observation["state"] = _state46(raw_env)
    observation["prompt"] = config["prompt"]
    return observation


def _video_frame(
    raw_env,
    *,
    episode: int,
    step: int,
    force_n: float,
    status: str,
) -> np.ndarray:
    images = raw_env._compute_observation()["images"]
    names = ("ego", "wrist_left", "wrist_right")
    panels = []
    for name in names:
        panel = np.asarray(images[name], dtype=np.uint8).copy()
        cv2.putText(
            panel,
            name,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    frame = np.concatenate(panels, axis=1)
    cv2.putText(
        frame,
        f"episode={episode} step={step} force={force_n:.2f}N {status}",
        (8, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def _step(raw_env, action44: np.ndarray) -> None:
    raw_env.step(policy_dual_arm_to_raw(rotvec_dual_arm_to_policy(action44)))


def _state46(raw_env) -> np.ndarray:
    state = raw_env._compute_observation()["state"]
    return np.concatenate(
        [np.asarray(state["tcp_pose"]).ravel(), np.asarray(state["gripper_pose"]).ravel()]
    )


def _handoff_intent_axis(
    actions44: np.ndarray,
    frame: int,
    horizon: int,
) -> np.ndarray:
    end = min(len(actions44) - 1, frame + horizon)
    relative_start = actions44[frame, :3] - actions44[frame, 22:25]
    relative_end = actions44[end, :3] - actions44[end, 22:25]
    axis = np.asarray(relative_end - relative_start, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        raise ValueError("recorded handoff continuation has no approach intent")
    return axis / norm


def _run_episode(
    *,
    episode: int,
    handoff_frame: int,
    actions44: np.ndarray,
    initial_state: np.ndarray,
    steps: int,
    descent_step_m: float,
    intent_horizon: int,
    hard_force_limit_n: float,
    enforce_fingertip_grasp_gate: bool,
    contact_response_enabled: bool,
    intent_mode: str,
    policy_client=None,
    openpi_config: dict | None = None,
    video_dir: Path | None = None,
    video_fps: int = 15,
) -> dict[str, object]:
    axis = (
        _handoff_intent_axis(actions44, handoff_frame, intent_horizon)
        if intent_mode != "online_pi05_chunk"
        else None
    )
    env = make_assembly_env(seed=episode, randomize=False, render_mode="rgb_array")
    raw_env = env.unwrapped
    adapter = SimCerebellumSensorAdapter(raw_env)
    evaluator = PrivilegedGeometryLabeler(raw_env)
    video_writer = None
    try:
        env.reset()
        restore_initial_state(
            env,
            "bimanual_assembly",
            CONFIG_MAPPING["bimanual_assembly"](),
            initial_state,
        )
        evaluator.reset_reference(raw_env)
        for action44 in actions44[: handoff_frame + 1]:
            _step(raw_env, action44)
        initial_truth = evaluator.compute(raw_env)
        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(
                video_dir / f"episode_{episode:06d}_online_pi05.mp4",
                fps=video_fps,
            )
            video_writer.append_data(
                _video_frame(
                    raw_env,
                    episode=episode,
                    step=0,
                    force_n=0.0,
                    status="handoff",
                )
            )
        initial_state46 = _state46(raw_env)
        finger_reference = state46_to_action44(initial_state46)
        baseline_sensor = adapter.capture({"state": initial_state46})
        baseline_wrench = np.asarray(baseline_sensor.wrist_wrench_world, dtype=np.float64)
        intent_executor = None
        intent_last_step = None
        online_chunk_steps = None
        if intent_mode == "online_pi05_chunk":
            if policy_client is None or openpi_config is None:
                raise ValueError("online_pi05_chunk requires a policy client and config")
            policy_result = policy_client.infer(_policy_observation(raw_env, openpi_config))
            online_chunk = np.asarray(policy_result["actions"], dtype=np.float64)
            intent_executor = OnlineIntentChunkExecutor(
                IntentChunkExecutionConfig(
                    hard_force_limit_n=hard_force_limit_n,
                    contact_response_enabled=contact_response_enabled,
                )
            )
            intent_executor.start(online_chunk, baseline_sensor)
            online_chunk_steps = int(online_chunk.shape[0])
        peak_force = 0.0
        minimum_stability = 1.0
        success = False
        exit_reason = "horizon_exhausted"
        trace = []
        for step_index in range(steps):
            state46 = _state46(raw_env)
            sensor = adapter.capture({"state": state46})
            residual = np.asarray(sensor.wrist_wrench_world, dtype=np.float64) - baseline_wrench
            force = float(np.max(np.linalg.norm(residual[:, :3], axis=1)))
            fingertip = np.linalg.norm(sensor.fingertip_force_world, axis=-1)
            stability = np.clip(np.sum(fingertip >= 0.5, axis=1) / 3.0, 0.0, 1.0)
            peak_force = max(peak_force, force)
            minimum_stability = min(minimum_stability, float(np.min(stability)))
            if force >= hard_force_limit_n and intent_mode != "online_pi05_chunk":
                exit_reason = "hard_force_limit"
                break
            if enforce_fingertip_grasp_gate and float(np.min(stability)) < 0.34:
                exit_reason = "grasp_unstable"
                break
            current_action = state46_to_action44(state46)
            if intent_mode == "online_pi05_chunk":
                assert intent_executor is not None
                intent_step = intent_executor.step(sensor, current_action)
                intent_last_step = intent_step
                command = intent_step.action44
            elif intent_mode == "recorded_delta_trajectory":
                intent_row = handoff_frame + step_index
                if intent_row + 1 >= len(actions44):
                    exit_reason = "handoff_intent_exhausted"
                    break
                right_twist = world_wrist_twist(
                    pose_from_action44(actions44[intent_row], side="right"),
                    pose_from_action44(actions44[intent_row + 1], side="right"),
                )
                left_twist = world_wrist_twist(
                    pose_from_action44(actions44[intent_row], side="left"),
                    pose_from_action44(actions44[intent_row + 1], side="left"),
                )
                command = apply_bimanual_wrist_twists(
                    current_action,
                    right_twist,
                    left_twist,
                    finger_reference44=actions44[intent_row + 1],
                )
            else:
                translation = descent_step_m * axis
                twist = np.concatenate([translation, np.zeros(3)])
                command = apply_bimanual_wrist_twists(
                    current_action,
                    0.8 * twist,
                    -0.2 * twist,
                    finger_reference44=finger_reference,
                )
            _step(raw_env, command)
            truth = evaluator.compute(raw_env)
            trace_row = {
                    "step": step_index,
                    "force_n": force,
                    "minimum_grasp_stability": float(np.min(stability)),
                    "lateral_error_m": float(truth.lateral_error_m),
                    "axis_error_rad": float(truth.axis_error_rad),
                    "approach_height_m": float(truth.approach_height_m),
                    "insert_ok": bool(truth.insert_ok),
                }
            if intent_mode == "online_pi05_chunk":
                trace_row.update(
                    {
                        "grasp_observable": intent_step.grasp_observable,
                        "contact_phase": intent_step.contact_phase,
                        "contact_correction_m": intent_step.contact_correction_m,
                        "contact_rotation_correction_rad": (
                            intent_step.contact_rotation_correction_rad
                        ),
                    }
                )
            trace.append(trace_row)
            if video_writer is not None:
                video_writer.append_data(
                    _video_frame(
                        raw_env,
                        episode=episode,
                        step=step_index + 1,
                        force_n=force,
                        status="inserted" if truth.insert_ok else "executing",
                    )
                )
            if truth.insert_ok:
                success = True
                exit_reason = "external_success_evaluator"
                break
            if intent_mode == "online_pi05_chunk" and not intent_step.active:
                exit_reason = intent_step.outcome
                break
        final = evaluator.compute(raw_env)
        return {
            "episode": episode,
            "handoff_frame": handoff_frame,
            "approach_axis_world": None if axis is None else axis.tolist(),
            "online_pi05_chunk_steps": online_chunk_steps,
            "online_pi05_final_phase": (
                None if intent_last_step is None else intent_last_step.phase
            ),
            "online_pi05_executor_outcome": (
                None if intent_last_step is None else intent_last_step.outcome
            ),
            "initial_lateral_error_m": float(initial_truth.lateral_error_m),
            "initial_axis_error_rad": float(initial_truth.axis_error_rad),
            "initial_approach_height_m": float(initial_truth.approach_height_m),
            "success": success,
            "exit_reason": exit_reason,
            "steps_executed": len(trace),
            "peak_force_n": peak_force,
            "minimum_grasp_stability": minimum_stability,
            "contact_response_enabled": contact_response_enabled,
            "contact_response_steps": sum(
                row.get("contact_correction_m", 0.0) > 0.0 for row in trace
            ),
            "maximum_contact_correction_m": max(
                (row.get("contact_correction_m", 0.0) for row in trace),
                default=0.0,
            ),
            "maximum_contact_rotation_correction_rad": max(
                (
                    row.get("contact_rotation_correction_rad", 0.0)
                    for row in trace
                ),
                default=0.0,
            ),
            "final_lateral_error_m": float(final.lateral_error_m),
            "final_axis_error_rad": float(final.axis_error_rad),
            "final_approach_height_m": float(final.approach_height_m),
            "trace": trace,
        }
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()


def main() -> None:
    args = parse_args()
    handoff_payload = json.loads(args.handoff_summary.read_text())
    handoffs = [
        record
        for record in handoff_payload["records"]
        if record.get("handoff_frame") is not None
    ]
    paths = discover_zarr_demos(args.zarr_input_dir)
    openpi_config = None
    policy_client = None
    if args.intent_mode == "online_pi05_chunk":
        openpi_config = yaml.safe_load(args.openpi_config.read_text())
        policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
    results = []
    for index, record in enumerate(handoffs, start=1):
        episode = int(record["episode_index"])
        actions44, initial_state = load_zarr_episode(paths[episode])
        if initial_state is None:
            raise ValueError(f"episode {episode} has no initial_state")
        print(f"[{index}/{len(handoffs)}] episode={episode}", flush=True)
        results.append(
            _run_episode(
                episode=episode,
                handoff_frame=int(record["handoff_frame"]),
                actions44=actions44,
                initial_state=initial_state,
                steps=args.steps,
                descent_step_m=args.descent_step_mm / 1000.0,
                intent_horizon=args.intent_horizon,
                hard_force_limit_n=args.hard_force_limit_n,
                enforce_fingertip_grasp_gate=args.enforce_fingertip_grasp_gate,
                contact_response_enabled=args.contact_response,
                intent_mode=args.intent_mode,
                policy_client=policy_client,
                openpi_config=openpi_config,
                video_dir=args.video_dir,
                video_fps=args.video_fps,
            )
        )
    successes = sum(result["success"] for result in results)
    delta_intent = args.intent_mode == "recorded_delta_trajectory"
    online_intent = args.intent_mode == "online_pi05_chunk"
    payload = {
        "stage": (
            "V2 natural-handoff intent-conditioned execution baseline"
            if delta_intent or online_intent
            else "V2 natural-handoff plain guarded-descent baseline"
        ),
        "num_requested_held_out_episodes": len(handoff_payload["records"]),
        "num_feasible_handoffs": len(results),
        "num_handoff_infeasible": len(handoff_payload["records"]) - len(results),
        "num_successes": successes,
        "success_rate_feasible_handoffs": successes / max(len(results), 1),
        "success_rate_all_requested": successes / max(len(handoff_payload["records"]), 1),
        "controller_online_inputs": [
            "proprioception",
            "fingertip_forces",
            "wrist_wrenches",
            (
                "pi05_online_handoff_action_chunk"
                if online_intent
                else "pi05_handoff_action_chunk_proxy"
                if delta_intent
                else "pi05_handoff_approach_axis_proxy"
            ),
        ],
        "approach_axis_proxy": (
            None if online_intent else "recorded_pi05_continuation_actions"
        ),
        "teacher_used_by_controller": False,
        "teacher_used_for_external_evaluation_only": True,
        "external_evaluator_terminates_on_success": True,
        "fingertip_grasp_gate_enforced": args.enforce_fingertip_grasp_gate,
        "contact_response_enabled": args.contact_response if online_intent else False,
        "intent_mode": args.intent_mode,
        "recorded_intent_is_deployable_pi05_output": False,
        "intent_is_live_pi05_output": online_intent,
        "frozen_handoff_is_external_evaluation_initialization": True,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
