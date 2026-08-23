"""Collect sensor-only ego-view response transitions from bounded wrist probes."""

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
from retrieval_cerebellum.active_view_probe import (
    ActiveViewProbeConfig,
    ActiveViewTransition,
    SensorOnlyActiveViewProbe,
    save_active_view_transitions,
)
from retrieval_cerebellum.assembly_kinematics import apply_bimanual_wrist_twists
from retrieval_cerebellum.ego_visual_state_estimation import EgoSpatialPredictor
from retrieval_cerebellum.sim_sensor_adapter import SimCerebellumSensorAdapter


DEFAULT_ZARR = Path("/mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-input-dir", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--settle-steps", type=int, default=3)
    parser.add_argument("--maximum-probes", type=int, default=8)
    parser.add_argument("--translation-step-mm", type=float, default=2.0)
    parser.add_argument("--rotation-step-deg", type=float, default=1.0)
    parser.add_argument("--maximum-wrist-force-n", type=float, default=8.0)
    parser.add_argument("--maximum-wrist-torque-nm", type=float, default=1.0)
    parser.add_argument("--minimum-stable-fingertips", type=int, default=2)
    parser.add_argument("--fingertip-force-threshold-n", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _step(raw_env, action44: np.ndarray) -> None:
    raw_env.step(policy_dual_arm_to_raw(rotvec_dual_arm_to_policy(action44)))


def _policy_observation(raw_env) -> dict[str, object]:
    observation = raw_env._compute_observation()
    state = observation["state"]
    state46 = np.concatenate(
        [
            np.asarray(state["tcp_pose"]).ravel(),
            np.asarray(state["gripper_pose"]).ravel(),
        ]
    )
    payload: dict[str, object] = {"state": state46}
    images = observation.get("images", {})
    payload.update({str(name): value for name, value in images.items()})
    return payload


def _restore_branch(
    actions44: np.ndarray,
    initial_state: np.ndarray,
    *,
    episode: int,
    frame: int,
):
    env = make_assembly_env(seed=episode, randomize=False)
    raw_env = env.unwrapped
    env.reset()
    restore_initial_state(
        env,
        "bimanual_assembly",
        CONFIG_MAPPING["bimanual_assembly"](),
        initial_state,
    )
    for action44 in actions44[: frame + 1]:
        _step(raw_env, action44)
    return env, raw_env


def _encode(
    predictor: EgoSpatialPredictor,
    sensor,
) -> tuple[np.ndarray, float]:
    if "ego" not in sensor.images:
        raise ValueError("sensor observation lacks ego RGB image")
    features, reliability, _ = predictor.encode([sensor.images["ego"]], batch_size=1)
    return features[0], float(reliability[0])


def _collect_branch(
    *,
    actions44: np.ndarray,
    initial_state: np.ndarray,
    episode: int,
    frame: int,
    control12: np.ndarray,
    settle_steps: int,
    predictor: EgoSpatialPredictor,
    probe: SensorOnlyActiveViewProbe,
) -> tuple[ActiveViewTransition | None, str | None]:
    env, raw_env = _restore_branch(
        actions44,
        initial_state,
        episode=episode,
        frame=frame,
    )
    adapter = SimCerebellumSensorAdapter(raw_env)
    base_action = np.asarray(actions44[frame], dtype=np.float64)
    try:
        before = adapter.capture(
            _policy_observation(raw_env),
            previous_action44=base_action,
        )
        if not probe.sensor_gate(before):
            return None, "baseline_sensor_gate"
        feature_before, reliability_before = _encode(predictor, before)
        candidate_action = apply_bimanual_wrist_twists(
            base_action,
            control12[:6],
            control12[6:],
            finger_reference44=base_action,
        )
        after = before
        for _ in range(settle_steps):
            _step(raw_env, candidate_action)
            after = adapter.capture(
                _policy_observation(raw_env),
                previous_action44=candidate_action,
            )
            if not probe.sensor_gate(after):
                return None, "post_probe_sensor_gate"
        feature_after, reliability_after = _encode(predictor, after)
        return (
            ActiveViewTransition(
                feature_before=feature_before,
                feature_after=feature_after,
                reliability_before=reliability_before,
                reliability_after=reliability_after,
                control12=control12,
                wrist_wrench_before=before.wrist_wrench_world,
                wrist_wrench_after=after.wrist_wrench_world,
            ),
            None,
        )
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    if args.settle_steps <= 0 or args.maximum_probes <= 0:
        raise ValueError("settle-steps and maximum-probes must be positive")
    zarr_paths = discover_zarr_demos(args.zarr_input_dir)
    if not 0 <= args.episode < len(zarr_paths):
        raise IndexError(f"episode {args.episode} is outside available demonstrations")
    actions44, initial_state = load_zarr_episode(zarr_paths[args.episode])
    if initial_state is None:
        raise ValueError("accurate replay requires initial_state")
    if not 0 <= args.frame < len(actions44):
        raise IndexError(f"frame {args.frame} is outside episode action rows")
    probe = SensorOnlyActiveViewProbe(
        ActiveViewProbeConfig(
            translation_step_m=args.translation_step_mm / 1000.0,
            rotation_step_rad=np.deg2rad(args.rotation_step_deg),
            maximum_wrist_force_n=args.maximum_wrist_force_n,
            maximum_wrist_torque_nm=args.maximum_wrist_torque_nm,
            minimum_stable_fingertips_per_hand=args.minimum_stable_fingertips,
            fingertip_force_threshold_n=args.fingertip_force_threshold_n,
        )
    )
    predictor = EgoSpatialPredictor.load(args.checkpoint, device=args.device)

    baseline_env, baseline_raw_env = _restore_branch(
        actions44,
        initial_state,
        episode=args.episode,
        frame=args.frame,
    )
    try:
        baseline_adapter = SimCerebellumSensorAdapter(baseline_raw_env)
        baseline = baseline_adapter.capture(
            _policy_observation(baseline_raw_env),
            previous_action44=actions44[args.frame],
        )
        controls = probe.bimanual_controls(baseline)[: args.maximum_probes]
    finally:
        baseline_env.close()
    if not controls:
        raise RuntimeError("baseline sensor gate rejected active view collection")

    transitions = []
    rejected = []
    for probe_index, control in enumerate(controls):
        transition, reason = _collect_branch(
            actions44=actions44,
            initial_state=initial_state,
            episode=args.episode,
            frame=args.frame,
            control12=control,
            settle_steps=args.settle_steps,
            predictor=predictor,
            probe=probe,
        )
        if transition is None:
            rejected.append({"probe_index": probe_index, "reason": reason})
        else:
            transitions.append(transition)
    if not transitions:
        raise RuntimeError("all active view probes were rejected by sensor gates")
    save_active_view_transitions(
        args.output,
        tuple(transitions),
        episode_index=args.episode,
        frame_index=args.frame,
    )
    reliability_delta = np.asarray(
        [item.reliability_after - item.reliability_before for item in transitions]
    )
    summary = {
        "stage": "sensor-only active ego-view transition collection",
        "output": str(args.output),
        "episode": args.episode,
        "frame": args.frame,
        "num_accepted": len(transitions),
        "num_rejected": len(rejected),
        "rejected": rejected,
        "reliability_delta": reliability_delta.tolist(),
        "runtime_inputs": [
            "ego RGB",
            "robot state",
            "fingertip forces",
            "wrist force torque",
            "previous action",
        ],
        "contains_geometry_labels": False,
        "contains_task_success_labels": False,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
