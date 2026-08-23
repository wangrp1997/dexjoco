"""Compare fixed and adaptive sensor-only contact search in frozen replays."""

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
from retrieval_cerebellum.contact_response_search import (
    BoundedContactResponseSearch,
    ContactSearchObservation,
    ContactSearchStrategy,
)
from retrieval_cerebellum.geometry_labels import PrivilegedGeometryLabeler
from retrieval_cerebellum.learning_data import state46_to_action44
from retrieval_cerebellum.sim_sensor_adapter import SimCerebellumSensorAdapter


DEFAULT_ZARR = Path("/mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-input-dir", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--episode", type=int, default=17)
    parser.add_argument("--frame", type=int, default=717)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--initial-offset-mm", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _step(raw_env, action44: np.ndarray) -> None:
    raw_env.step(policy_dual_arm_to_raw(rotvec_dual_arm_to_policy(action44)))


def _observed_state46(raw_env) -> np.ndarray:
    state = raw_env._compute_observation()["state"]
    return np.concatenate(
        [np.asarray(state["tcp_pose"]).ravel(), np.asarray(state["gripper_pose"]).ravel()]
    )


def _approach_axis(actions44: np.ndarray, frame: int, history: int) -> np.ndarray:
    start = max(0, frame - history)
    relative_start = actions44[start, :3] - actions44[start, 22:25]
    relative_end = actions44[frame, :3] - actions44[frame, 22:25]
    axis = np.asarray(relative_end - relative_start, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        raise ValueError("recent deployable action history has no approach direction")
    return axis / norm


def _run_strategy(
    *,
    strategy: ContactSearchStrategy,
    episode: int,
    frame: int,
    steps: int,
    actions44: np.ndarray,
    initial_state: np.ndarray,
    approach_axis: np.ndarray,
    initial_offset_m: float,
) -> dict[str, object]:
    env = make_assembly_env(seed=episode, randomize=False, render_mode=None)
    raw_env = env.unwrapped
    adapter = SimCerebellumSensorAdapter(raw_env)
    evaluator = PrivilegedGeometryLabeler(raw_env)
    try:
        env.reset()
        restore_initial_state(
            env,
            "bimanual_assembly",
            CONFIG_MAPPING["bimanual_assembly"](),
            initial_state,
        )
        evaluator.reset_reference(raw_env)
        for action44 in actions44[: frame + 1]:
            _step(raw_env, action44)
        baseline_state = _observed_state46(raw_env)
        baseline_action = state46_to_action44(baseline_state)
        hint = np.asarray([1.0, 0.0, 0.0])
        if abs(float(approach_axis @ hint)) > 0.9:
            hint = np.asarray([0.0, 1.0, 0.0])
        tangent = hint - approach_axis * float(approach_axis @ hint)
        tangent /= np.linalg.norm(tangent) + 1e-12
        offset_twist = np.concatenate([initial_offset_m * tangent, np.zeros(3)])
        offset_action = apply_bimanual_wrist_twists(
            baseline_action,
            0.8 * offset_twist,
            -0.2 * offset_twist,
            finger_reference44=baseline_action,
        )
        for _ in range(3):
            _step(raw_env, offset_action)
        initial_truth = evaluator.compute(raw_env)
        baseline_sensor = adapter.capture({"state": baseline_state})
        baseline_wrench = np.asarray(baseline_sensor.wrist_wrench_world, dtype=np.float64)
        search = BoundedContactResponseSearch(approach_axis, strategy)
        trace = []
        peak_force = 0.0
        success = False
        retreat = False
        for step_index in range(steps):
            state46 = _observed_state46(raw_env)
            sensor = adapter.capture({"state": state46})
            residual = np.asarray(sensor.wrist_wrench_world, dtype=np.float64) - baseline_wrench
            fingertip = np.linalg.norm(sensor.fingertip_force_world, axis=-1)
            stability = np.clip(np.sum(fingertip >= 0.5, axis=1) / 3.0, 0.0, 1.0)
            observation = ContactSearchObservation(residual, stability)
            command = search.step(observation)
            current_action = state46_to_action44(state46)
            relative = command.relative_translation_world
            twist = np.concatenate([relative, np.zeros(3)])
            next_action = apply_bimanual_wrist_twists(
                current_action,
                command.right_motion_fraction * twist,
                -(1.0 - command.right_motion_fraction) * twist,
                finger_reference44=baseline_action,
            )
            _step(raw_env, next_action)
            truth = evaluator.compute(raw_env)
            force = float(np.max(np.linalg.norm(residual[:, :3], axis=1)))
            peak_force = max(peak_force, force)
            trace.append(
                {
                    "step": step_index,
                    "phase": command.phase,
                    "right_motion_fraction": command.right_motion_fraction,
                    "force_n": force,
                    "lateral_error_m": float(truth.lateral_error_m),
                    "axis_error_rad": float(truth.axis_error_rad),
                    "approach_height_m": float(truth.approach_height_m),
                    "insert_ok": bool(truth.insert_ok),
                }
            )
            success = bool(truth.insert_ok)
            retreat = command.retreat
            if success or retreat:
                break
        final = evaluator.compute(raw_env)
        return {
            "strategy": strategy.value,
            "initial_lateral_error_m": float(initial_truth.lateral_error_m),
            "initial_axis_error_rad": float(initial_truth.axis_error_rad),
            "initial_approach_height_m": float(initial_truth.approach_height_m),
            "success": success,
            "retreat": retreat,
            "steps_executed": len(trace),
            "peak_force_n": peak_force,
            "final_lateral_error_m": float(final.lateral_error_m),
            "final_axis_error_rad": float(final.axis_error_rad),
            "final_approach_height_m": float(final.approach_height_m),
            "trace": trace,
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    paths = discover_zarr_demos(args.zarr_input_dir)
    actions44, initial_state = load_zarr_episode(paths[args.episode])
    if initial_state is None:
        raise ValueError("accurate replay requires initial_state")
    axis = _approach_axis(actions44, args.frame, args.history)
    results = [
        _run_strategy(
            strategy=strategy,
            episode=args.episode,
            frame=args.frame,
            steps=args.steps,
            actions44=actions44,
            initial_state=initial_state,
            approach_axis=axis,
            initial_offset_m=args.initial_offset_mm / 1000.0,
        )
        for strategy in ContactSearchStrategy
    ]
    payload = {
        "stage": "V2 sensor-only bimanual contact response search prototype",
        "episode": args.episode,
        "frame": args.frame,
        "approach_axis_source": "previous_executed_action_history",
        "controller_online_inputs": [
            "proprioception",
            "previous_actions",
            "fingertip_forces",
            "wrist_wrenches",
        ],
        "teacher_used_by_controller": False,
        "teacher_used_for_external_evaluation_only": True,
        "approach_axis_world": axis.tolist(),
        "initial_offset_mm": args.initial_offset_mm,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
