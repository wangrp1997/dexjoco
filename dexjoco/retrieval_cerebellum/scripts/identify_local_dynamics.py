"""Identify one-step contact dynamics at replayed RC-HB-SQP handoff states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state
from dexquery.data.action_utils import policy_dual_arm_to_raw, rotvec_dual_arm_to_policy
from dexquery.data.episode_replay import make_assembly_env
from dexquery.data.finger_contact_forces import FingerForceLabeler
from dexquery.data.zarr_io import discover_zarr_demos, load_zarr_episode
from retrieval_cerebellum.assembly_kinematics import apply_bimanual_wrist_twists
from retrieval_cerebellum.geometry_labels import PrivilegedGeometryLabeler
from retrieval_cerebellum.learning_data import state46_to_action44
from retrieval_cerebellum.local_dynamics import (
    MujocoIntegrationSnapshot,
    identify_one_step_dynamics,
)
from retrieval_cerebellum.short_horizon_execution import (
    check_bimanual_action_kinematics,
)
from retrieval_cerebellum.skill_prototype import SuccessfulSkillMemory


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")
DEFAULT_ZARR = Path("/mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly")
DEFAULT_PLAN_DIR = Path(
    "outputs/retrieval_cerebellum/belief_space_sqp_ep17_stable_handoff"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--zarr-input-dir", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/local_dynamics"),
    )
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--handoff-row", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--translation-step-m", type=float, default=2.5e-4)
    parser.add_argument("--rotation-step-rad", type=float, default=2.5e-3)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _state5(frame) -> np.ndarray:
    return np.asarray(
        [
            frame.peg_tip_in_hole_position[0],
            frame.peg_tip_in_hole_position[1],
            frame.peg_in_hole_rotvec[0],
            frame.peg_in_hole_rotvec[1],
            -frame.peg_tip_in_hole_position[2],
        ],
        dtype=np.float64,
    )


def _step_action(raw_env, action44: np.ndarray) -> None:
    action46 = rotvec_dual_arm_to_policy(np.asarray(action44, dtype=np.float32))
    raw_env.step(policy_dual_arm_to_raw(action46))


def _observed_action44(raw_env) -> np.ndarray:
    observation = raw_env._compute_observation()["state"]
    state46 = np.concatenate(
        [
            np.asarray(observation["tcp_pose"], dtype=np.float64).ravel(),
            np.asarray(observation["gripper_pose"], dtype=np.float64).ravel(),
        ]
    )
    return state46_to_action44(state46)


def _identify_episode(
    *,
    episode_index: int,
    record: dict,
    dataset_root: Path,
    zarr_path: Path,
    action_limits,
    translation_step_m: float,
    rotation_step_rad: float,
    rollout_steps: int,
    seed: int,
    handoff_row_override: int | None,
) -> tuple[dict, dict[str, np.ndarray]]:
    import pyarrow.parquet as parquet

    learning_path = (
        dataset_root
        / "retrieval_cerebellum_learning"
        / "episodes"
        / f"episode_{episode_index:06d}.parquet"
    )
    table = parquet.read_table(learning_path, columns=["frame_index"])
    frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    handoff_row = (
        int(record["handoff_row"])
        if handoff_row_override is None
        else handoff_row_override
    )
    if not 0 <= handoff_row < len(frame_index):
        raise IndexError(f"handoff row {handoff_row} is outside episode rows")
    handoff_frame = int(frame_index[handoff_row])
    replay_actions, initial_state = load_zarr_episode(zarr_path)
    if initial_state is None:
        raise ValueError(f"episode {episode_index} has no recorded initial_state")

    env = make_assembly_env(seed=seed, randomize=False, render_mode=None)
    raw_env = env.unwrapped
    geometry_labeler = PrivilegedGeometryLabeler(raw_env)
    force_labeler = FingerForceLabeler(raw_env)
    config = CONFIG_MAPPING["bimanual_assembly"]()
    try:
        env.reset()
        if not has_restorer("bimanual_assembly"):
            raise RuntimeError("bimanual_assembly initial-state restorer is unavailable")
        restore_initial_state(env, "bimanual_assembly", config, initial_state)
        geometry_labeler.reset_reference(raw_env)
        force_labeler.reset_reference(raw_env)
        for action in replay_actions[: handoff_frame + 1]:
            _step_action(raw_env, action)

        baseline_geometry = geometry_labeler.compute(raw_env)
        baseline_state = _state5(baseline_geometry)
        current_command = np.asarray(
            replay_actions[handoff_frame], dtype=np.float64
        ).reshape(44)
        snapshot = MujocoIntegrationSnapshot.capture(raw_env)
        preflight_conditions = []
        hold_wrenches: dict[str, np.ndarray] = {}

        def rollout_delta(right_twist: np.ndarray, left_twist: np.ndarray) -> np.ndarray:
            snapshot.restore(raw_env)
            for side, twist, position_limit, rotation_limit in (
                (
                    "right",
                    right_twist,
                    action_limits.right_position_step_m,
                    action_limits.right_rotation_step_rad,
                ),
                (
                    "left",
                    left_twist,
                    action_limits.left_position_step_m,
                    action_limits.left_rotation_step_rad,
                ),
            ):
                if np.linalg.norm(twist[:3]) > position_limit + 1e-12:
                    raise RuntimeError(f"{side} dynamics perturbation exceeds position limit")
                if np.linalg.norm(twist[3:]) > rotation_limit + 1e-12:
                    raise RuntimeError(f"{side} dynamics perturbation exceeds rotation limit")
            target_action = apply_bimanual_wrist_twists(
                current_command,
                right_twist,
                left_twist,
                finger_reference44=current_command,
            )
            preflight = check_bimanual_action_kinematics(
                raw_env,
                current_command,
                target_action,
                action_limits,
            )
            preflight_conditions.append(
                max(
                    preflight.right.jacobian_condition,
                    preflight.left.jacobian_condition,
                )
            )
            for _ in range(rollout_steps):
                _step_action(raw_env, target_action)
            force = force_labeler.compute(raw_env)
            if not np.any(right_twist) and not np.any(left_twist):
                hold_wrenches["right"] = np.asarray(
                    force.wrist_ft_right, dtype=np.float64
                )
                hold_wrenches["left"] = np.asarray(
                    force.wrist_ft_left, dtype=np.float64
                )
            return _state5(geometry_labeler.compute(raw_env)) - baseline_state

        linearization = identify_one_step_dynamics(
            rollout_delta,
            translation_step_m=translation_step_m,
            rotation_step_rad=rotation_step_rad,
            rollout_steps=rollout_steps,
        )
        snapshot.restore(raw_env)
        replay_error = (
            baseline_state - np.asarray(record["handoff_state5"], dtype=np.float64)
            if handoff_row_override is None
            else np.zeros(5, dtype=np.float64)
        )
        result = {
            "episode_index": episode_index,
            "split": record["split"],
            "handoff_row": handoff_row,
            "handoff_frame": handoff_frame,
            "handoff_state5": baseline_state.tolist(),
            "handoff_replay_error5": replay_error.tolist(),
            "handoff_replay_error_l2": float(np.linalg.norm(replay_error)),
            "peg_ok": bool(baseline_geometry.peg_ok),
            "tray_ok": bool(baseline_geometry.tray_ok),
            "maximum_preflight_condition": float(max(preflight_conditions)),
            "hold_wrist_ft_right": hold_wrenches["right"].tolist(),
            "hold_wrist_ft_left": hold_wrenches["left"].tolist(),
            **linearization.to_dict(),
        }
        arrays = {
            "drift": linearization.drift,
            "right_state_jacobian": linearization.right_state_jacobian,
            "left_state_jacobian": linearization.left_state_jacobian,
            "right_even_residual": linearization.right_even_residual,
            "left_even_residual": linearization.left_even_residual,
            "handoff_state5": baseline_state,
            "rollout_steps": np.asarray(linearization.rollout_steps, dtype=np.int64),
            "handoff_row": np.asarray(handoff_row, dtype=np.int64),
            "hold_wrist_ft_right": hold_wrenches["right"],
            "hold_wrist_ft_left": hold_wrenches["left"],
        }
        return result, arrays
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if args.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if args.handoff_row is not None and args.handoff_row < 0:
        raise ValueError("handoff_row must be non-negative")
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plan_summary = json.loads((args.plan_dir / "summary.json").read_text())
    records = plan_summary["records"]
    records = [item for item in records if item.get("handoff_row") is not None]
    if args.episodes is not None:
        requested = set(args.episodes)
        records = [item for item in records if int(item["episode_index"]) in requested]
    if args.max_episodes is not None:
        records = records[: args.max_episodes]
    if not records:
        raise ValueError("no plan episodes selected")
    if args.handoff_row is not None and len(records) != 1:
        raise ValueError("handoff_row override requires exactly one episode")

    learning_dir = args.dataset_root / "retrieval_cerebellum_learning"
    action_limits = SuccessfulSkillMemory.load(learning_dir).limits
    zarr_paths = discover_zarr_demos(args.zarr_input_dir)
    results = []
    for record in records:
        episode_index = int(record["episode_index"])
        result, arrays = _identify_episode(
            episode_index=episode_index,
            record=record,
            dataset_root=args.dataset_root,
            zarr_path=zarr_paths[episode_index],
            action_limits=action_limits,
            translation_step_m=args.translation_step_m,
            rotation_step_rad=args.rotation_step_rad,
            rollout_steps=args.rollout_steps,
            seed=args.seed_base + episode_index,
            handoff_row_override=args.handoff_row,
        )
        episode_stem = args.output_dir / "episodes" / f"episode_{episode_index:06d}"
        _write_json(episode_stem.with_suffix(".json"), result)
        np.savez_compressed(episode_stem.with_suffix(".npz"), **arrays)
        results.append(result)
        print(
            f"episode={episode_index} drift={np.linalg.norm(arrays['drift']):.6g} "
            f"cond={result['condition_number']:.3g} "
            f"even={result['maximum_even_residual']:.3g}",
            flush=True,
        )

    summary = {
        "stage": "P5 paired-rollout local dynamics identification",
        "plan_dir": str(args.plan_dir),
        "num_episodes": len(results),
        "translation_step_m": args.translation_step_m,
        "rotation_step_rad": args.rotation_step_rad,
        "rollout_steps": args.rollout_steps,
        "records": results,
        "limitations": [
            "privileged geometry is used to evaluate the local dynamics response",
            "the identified affine dynamics are valid only near the replayed handoff state",
            "wrench response Jacobians are not yet identified",
        ],
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
