"""Execute one or two RC-HB-SQP actions from accurately replayed handoff states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state
from dexquery.data.action_utils import policy_dual_arm_to_raw, rotvec_dual_arm_to_policy
from dexquery.data.episode_replay import make_assembly_env
from dexquery.data.finger_contact_forces import FingerForceLabeler
from dexquery.data.zarr_io import discover_zarr_demos, load_zarr_episode
from retrieval_cerebellum.geometry_labels import PrivilegedGeometryLabeler
from retrieval_cerebellum.learning_data import state46_to_action44
from retrieval_cerebellum.short_horizon_execution import (
    ExecutionSafetyLimits,
    check_bimanual_action_kinematics,
    command_increment_safety_reasons,
    execution_safety_reasons,
)
from retrieval_cerebellum.skill_prototype import SuccessfulSkillMemory
from retrieval_cerebellum.sqp_skill_adapter import SuccessfulSkillSQPAdapter


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")
DEFAULT_ZARR = Path("/mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly")
DEFAULT_PLAN_DIR = Path(
    "outputs/retrieval_cerebellum/belief_space_sqp_real_jacobian_validation10"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--zarr-input-dir", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/sqp_short_execution"),
    )
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--execute-steps", type=int, default=1)
    parser.add_argument("--execution-fraction", type=float, default=0.25)
    parser.add_argument(
        "--hold-fingers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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


def _right_attachment6(frame) -> np.ndarray:
    return np.concatenate(
        [frame.peg_in_right_palm_position, frame.peg_in_right_palm_rotvec]
    ).astype(np.float64)


def _left_attachment6(frame) -> np.ndarray:
    return np.concatenate(
        [frame.tray_in_left_palm_position, frame.tray_in_left_palm_rotvec]
    ).astype(np.float64)


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


def _fractional_action44(
    current_action44: np.ndarray,
    planned_action44: np.ndarray,
    fraction: float,
) -> np.ndarray:
    current = np.asarray(current_action44, dtype=np.float64).reshape(44)
    planned = np.asarray(planned_action44, dtype=np.float64).reshape(44)
    command = current + fraction * (planned - current)
    for rotation_slice in (slice(3, 6), slice(25, 28)):
        current_rotation = Rotation.from_rotvec(current[rotation_slice])
        planned_rotation = Rotation.from_rotvec(planned[rotation_slice])
        delta = planned_rotation * current_rotation.inv()
        command[rotation_slice] = (
            Rotation.from_rotvec(fraction * delta.as_rotvec()) * current_rotation
        ).as_rotvec()
    return command.astype(np.float32)


def _attachment_delta(before: np.ndarray, after: np.ndarray) -> dict:
    before_pose = np.asarray(before, dtype=np.float64).reshape(6)
    after_pose = np.asarray(after, dtype=np.float64).reshape(6)
    rotation_delta = (
        Rotation.from_rotvec(after_pose[3:])
        * Rotation.from_rotvec(before_pose[3:]).inv()
    )
    return {
        "translation_m": float(np.linalg.norm(after_pose[:3] - before_pose[:3])),
        "rotation_rad": float(rotation_delta.magnitude()),
    }


def _execute_episode(
    *,
    episode_index: int,
    record: dict,
    dataset_root: Path,
    zarr_path: Path,
    action_limits,
    safety_limits: ExecutionSafetyLimits,
    execute_steps: int,
    execution_fraction: float,
    hold_fingers: bool,
    seed: int,
) -> dict:
    import pyarrow.parquet as parquet

    learning_path = (
        dataset_root
        / "retrieval_cerebellum_learning"
        / "episodes"
        / f"episode_{episode_index:06d}.parquet"
    )
    table = parquet.read_table(learning_path, columns=["frame_index"])
    frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    handoff_row = int(record["handoff_row"])
    handoff_frame = int(frame_index[handoff_row])
    replay_actions, initial_state = load_zarr_episode(zarr_path)
    if initial_state is None:
        raise ValueError(f"episode {episode_index} has no recorded initial_state")
    if handoff_frame >= len(replay_actions):
        raise IndexError(
            f"handoff frame {handoff_frame} exceeds episode length {len(replay_actions)}"
        )
    plan_path = Path(record["action_plan_path"])
    plan = np.load(plan_path)
    action_plan = np.asarray(plan["action44"], dtype=np.float64)
    planned_linear_first_state = np.asarray(
        plan["linear_first_state5"],
        dtype=np.float64,
    )
    control_repeats = int(plan["control_repeats"]) if "control_repeats" in plan else 1
    steps_requested = min(execute_steps, len(action_plan))

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

        before_geometry = geometry_labeler.compute(raw_env)
        before_state = _state5(before_geometry)
        handoff_state = before_state.copy()
        before_right_attachment = _right_attachment6(before_geometry)
        before_left_attachment = _left_attachment6(before_geometry)
        expected_state = np.asarray(record["handoff_state5"], dtype=np.float64)
        replay_error = before_state - expected_state
        observed_action = _observed_action44(raw_env)
        current_command = np.asarray(
            replay_actions[handoff_frame], dtype=np.float64
        ).reshape(44)
        steps = []
        stop_reasons = []
        for step_index, planned_action in enumerate(action_plan[:steps_requested]):
            target_action = _fractional_action44(
                current_command,
                planned_action,
                execution_fraction,
            )
            if hold_fingers:
                target_action[6:22] = current_command[6:22]
                target_action[28:44] = current_command[28:44]
            preflight = check_bimanual_action_kinematics(
                raw_env,
                current_command,
                target_action,
                action_limits,
            )
            command_reasons = command_increment_safety_reasons(
                current_command,
                target_action,
                action_limits,
            )
            structural_reasons = tuple(
                reason
                for reason in preflight.reasons
                if "jacobian_condition_exceeds_limit" in reason
                or "estimated_joint_position_near_limit" in reason
            )
            blocking_preflight_reasons = command_reasons + structural_reasons
            step_record = {
                "step_index": step_index,
                "hold_fingers": hold_fingers,
                "preflight": preflight.to_dict(),
                "command_increment_reasons": list(command_reasons),
                "blocking_preflight_reasons": list(blocking_preflight_reasons),
            }
            if blocking_preflight_reasons:
                stop_reasons.extend(blocking_preflight_reasons)
                step_record["executed"] = False
                steps.append(step_record)
                break
            for _ in range(control_repeats):
                _step_action(raw_env, target_action)
            geometry = geometry_labeler.compute(raw_env)
            force = force_labeler.compute(raw_env)
            after_state = _state5(geometry)
            right_attachment_after = _right_attachment6(geometry)
            left_attachment_after = _left_attachment6(geometry)
            safety_reasons = execution_safety_reasons(
                before_state5=before_state,
                after_state5=after_state,
                before_peg_ok=before_geometry.peg_ok,
                after_peg_ok=geometry.peg_ok,
                before_tray_ok=before_geometry.tray_ok,
                after_tray_ok=geometry.tray_ok,
                before_right_attachment6=before_right_attachment,
                after_right_attachment6=right_attachment_after,
                before_left_attachment6=before_left_attachment,
                after_left_attachment6=left_attachment_after,
                wrist_ft_right=force.wrist_ft_right,
                wrist_ft_left=force.wrist_ft_left,
                limits=safety_limits,
            )
            step_record.update(
                {
                    "executed": True,
                    "before_state5": before_state.tolist(),
                    "after_state5": after_state.tolist(),
                    "state_delta5": (after_state - before_state).tolist(),
                    "execution_fraction": execution_fraction,
                    "control_repeats": control_repeats,
                    "scaled_linear_prediction_state5": (
                        expected_state
                        + execution_fraction
                        * (planned_linear_first_state - expected_state)
                    ).tolist(),
                    "scaled_linear_prediction_error5": (
                        after_state
                        - expected_state
                        - execution_fraction
                        * (planned_linear_first_state - expected_state)
                    ).tolist(),
                    "right_attachment_delta": _attachment_delta(
                        before_right_attachment,
                        right_attachment_after,
                    ),
                    "left_attachment_delta": _attachment_delta(
                        before_left_attachment,
                        left_attachment_after,
                    ),
                    "before_peg_ok": bool(before_geometry.peg_ok),
                    "before_tray_ok": bool(before_geometry.tray_ok),
                    "peg_ok": bool(geometry.peg_ok),
                    "tray_ok": bool(geometry.tray_ok),
                    "insert_ok": bool(geometry.insert_ok),
                    "peg_contact_count": int(geometry.peg_contact_count),
                    "tray_contact_count": int(geometry.tray_contact_count),
                    "wrist_ft_right": force.wrist_ft_right.tolist(),
                    "wrist_ft_left": force.wrist_ft_left.tolist(),
                    "safety_reasons": list(safety_reasons),
                }
            )
            steps.append(step_record)
            if safety_reasons:
                stop_reasons.extend(safety_reasons)
                break
            before_state = after_state
            before_geometry = geometry
            before_right_attachment = _right_attachment6(geometry)
            before_left_attachment = _left_attachment6(geometry)
            current_command = target_action
            observed_action = _observed_action44(raw_env)
        executed_steps = sum(bool(item.get("executed")) for item in steps)
        return {
            "episode_index": episode_index,
            "split": record["split"],
            "handoff_row": handoff_row,
            "handoff_frame": handoff_frame,
            "replay_state5": handoff_state.tolist(),
            "expected_handoff_state5": expected_state.tolist(),
            "handoff_replay_error5": replay_error.tolist(),
            "handoff_replay_error_l2": float(np.linalg.norm(replay_error)),
            "steps_requested": steps_requested,
            "steps_executed": executed_steps,
            "safe": bool(executed_steps == steps_requested and not stop_reasons),
            "stop_reasons": list(dict.fromkeys(stop_reasons)),
            "steps": steps,
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    if args.execute_steps <= 0:
        raise ValueError("execute_steps must be positive")
    if not 0.0 <= args.execution_fraction <= 1.0:
        raise ValueError("execution_fraction must be in [0, 1]")
    if args.max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plan_summary = json.loads((args.plan_dir / "summary.json").read_text())
    records = plan_summary["records"]
    records = [
        item
        for item in records
        if item.get("success") and item.get("action_plan_path") is not None
    ]
    if args.episodes is not None:
        requested = set(args.episodes)
        records = [item for item in records if int(item["episode_index"]) in requested]
    records = records[: args.max_episodes]
    if not records:
        raise ValueError("no plan episodes selected")

    learning_dir = args.dataset_root / "retrieval_cerebellum_learning"
    estimation_dir = args.dataset_root / "retrieval_cerebellum_estimation"
    skill_memory = SuccessfulSkillMemory.load(learning_dir)
    adapter = SuccessfulSkillSQPAdapter.load(learning_dir, estimation_dir)
    safety_limits = ExecutionSafetyLimits.from_successful_records(
        list(adapter.records.values())
    )
    zarr_paths = discover_zarr_demos(args.zarr_input_dir)
    results = []
    for record in records:
        episode_index = int(record["episode_index"])
        if episode_index >= len(zarr_paths):
            raise IndexError(f"episode {episode_index} has no matching zarr replay")
        result = _execute_episode(
            episode_index=episode_index,
            record=record,
            dataset_root=args.dataset_root,
            zarr_path=zarr_paths[episode_index],
            action_limits=skill_memory.limits,
            safety_limits=safety_limits,
            execute_steps=args.execute_steps,
            execution_fraction=args.execution_fraction,
            hold_fingers=args.hold_fingers,
            seed=args.seed_base + episode_index,
        )
        results.append(result)
        _write_json(
            args.output_dir / "episodes" / f"episode_{episode_index:06d}.json",
            result,
        )
        print(
            f"episode={episode_index} safe={result['safe']} "
            f"executed={result['steps_executed']}/{result['steps_requested']} "
            f"stop={result['stop_reasons']}",
            flush=True,
        )

    summary = {
        "stage": "P5 MuJoCo short-horizon execution",
        "plan_dir": str(args.plan_dir),
        "execute_steps": args.execute_steps,
        "execution_fraction": args.execution_fraction,
        "hold_fingers": args.hold_fingers,
        "num_episodes": len(results),
        "num_safe": sum(bool(item["safe"]) for item in results),
        "action_limits": {
            "right_position_step_m": skill_memory.limits.right_position_step_m,
            "left_position_step_m": skill_memory.limits.left_position_step_m,
            "right_rotation_step_rad": skill_memory.limits.right_rotation_step_rad,
            "left_rotation_step_rad": skill_memory.limits.left_rotation_step_rad,
        },
        "safety_limits": safety_limits.to_dict(),
        "records": results,
        "limitations": [
            "plans still start from privileged oracle assembly beliefs",
            "only one or two open-loop actions are executed before replanning exists",
            "fingers are held at the measured handoff posture until finger-force optimization is implemented",
            "privileged geometry and force labels are used only for evaluation and stopping",
        ],
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
