"""Run the RC-HB-SQP core on held-out successful insertion states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from dexjoco.sim.envs.assembly_geometry import (
    XMLS_DIR,
    arena_xml_path,
    names_for_family,
)
from retrieval_cerebellum.assembly_kinematics import (
    BimanualAssemblyKinematics,
    align_and_project_finger_references,
    bimanual_controls_to_action44,
    palm_pose_from_action44,
    pose_matrix,
)
from retrieval_cerebellum.handoff_selection import (
    StableHandoffConfig,
    final_entry_row,
    select_stable_handoff_row,
)
from retrieval_cerebellum.short_horizon_execution import ExecutionSafetyLimits
from retrieval_cerebellum.belief_space_sqp import (
    BeliefSpaceSQPConfig,
    BimanualInsertionBelief,
    InsertionGeometry,
    LocalBimanualInsertionModel,
    RetrievalConditionedBeliefSpaceSQP,
)
from retrieval_cerebellum.sqp_skill_adapter import (
    SuccessfulSkillSQPAdapter,
    assembly_state_from_belief18,
)


DEFAULT_DATASET = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--learning-dir", type=Path, default=None)
    parser.add_argument("--estimation-dir", type=Path, default=None)
    parser.add_argument(
        "--dynamics-dir",
        type=Path,
        default=None,
        help="Optional directory of paired-rollout episode_XXXXXX.npz models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retrieval_cerebellum/belief_space_sqp_oracle"),
    )
    parser.add_argument("--family-id", default="round_8mm")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--handoff-policy",
        choices=("stable", "latest_preentry"),
        default="stable",
    )
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--handoff-row", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _xml_geom(path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    for geom in root.iter("geom"):
        if geom.attrib.get("name") == name:
            position = np.fromstring(geom.attrib.get("pos", "0 0 0"), sep=" ")
            size = np.fromstring(geom.attrib["size"], sep=" ")
            return position, size
    raise KeyError(f"geometry {name!r} not found in {path}")


def _round_radial_clearance_m(family_id: str) -> float:
    names = names_for_family(family_id)
    if names.section != "round":
        raise ValueError("the current analytic radial envelope supports round pegs only")
    _, peg_size = _xml_geom(XMLS_DIR / names.peg_asset_xml, names.peg_collision)
    wall_name = f"{names.socket_body}_wall_pos_x"
    wall_position, wall_size = _xml_geom(
        XMLS_DIR / names.socket_asset_xml,
        wall_name,
    )
    socket_inner_radius = float(abs(wall_position[0]) - wall_size[0])
    clearance = socket_inner_radius - float(peg_size[0])
    if clearance <= 0.0:
        raise ValueError(f"non-positive radial clearance derived for {family_id}")
    return clearance


def _peg_tip_in_peg_m(family_id: str) -> np.ndarray:
    import mujoco

    names = names_for_family(family_id)
    model = mujoco.MjModel.from_xml_path(str(arena_xml_path(family_id)))
    peg_body_id = int(model.body(names.peg_body).id)
    peg_geom_id = int(model.geom(names.peg_collision).id)
    geom_type = int(model.geom_type[peg_geom_id])
    if geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        half_length = float(model.geom_size[peg_geom_id, 1])
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        half_length = float(model.geom_size[peg_geom_id, 2])
    else:
        raise ValueError(
            f"unsupported insertion collision geometry type {geom_type} for {family_id}"
        )
    geom_rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(geom_rotation, model.geom_quat[peg_geom_id])
    insertion_end = np.asarray(model.geom_pos[peg_geom_id], dtype=np.float64).copy()
    insertion_end += geom_rotation.reshape(3, 3) @ np.asarray(
        [0.0, 0.0, -half_length],
        dtype=np.float64,
    )
    if int(model.geom_bodyid[peg_geom_id]) != peg_body_id:
        raise ValueError("peg collision geometry is not attached to the peg body")
    return insertion_end


def _entry_handoff_row(features: np.ndarray) -> int:
    return final_entry_row(features) - 1


def _held_out_paths(learning_dir: Path, requested: set[int] | None) -> list[Path]:
    import pyarrow.parquet as parquet

    paths = []
    for path in sorted((learning_dir / "episodes").glob("episode_*.parquet")):
        episode_index = int(path.stem.rsplit("_", 1)[-1])
        if requested is not None and episode_index not in requested:
            continue
        table = parquet.read_table(path, columns=["split"])
        if str(table["split"][0].as_py()) != "train":
            paths.append(path)
    return paths


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _load_completed(output_dir: Path) -> dict[int, dict]:
    episode_dir = output_dir / "episodes"
    completed = {}
    for path in sorted(episode_dir.glob("episode_*.json")):
        record = json.loads(path.read_text())
        completed[int(record["episode_index"])] = record
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        for record in summary.get("records", []):
            episode_index = int(record["episode_index"])
            if episode_index not in completed:
                completed[episode_index] = record
                _write_json(
                    episode_dir / f"episode_{episode_index:06d}.json",
                    record,
                )
    return completed


def _summary(
    records: list[dict],
    *,
    family_id: str,
    clearance: float,
    target_depth: float,
    top_k: int,
    horizon: int,
    handoff_policy: str,
) -> dict:
    return {
        "stage": "RC-HB-SQP oracle shadow",
        "belief_source": "privileged geometry at the selected handoff frame",
        "gallery_split": "train",
        "family_id": family_id,
        "radial_clearance_m": clearance,
        "target_depth_m": target_depth,
        "top_k": top_k,
        "horizon": horizon,
        "handoff_policy": handoff_policy,
        "num_episodes": len(records),
        "num_success": sum(bool(item["success"]) for item in records),
        "records": records,
        "limitations": [
            "oracle assembly belief is used only to isolate the optimizer upper bound",
            "geometric Jacobians are used only for attachment uncertainty propagation",
            "paired-rollout state dynamics are used when a dynamics directory is supplied",
            "wrench capacities are empirical successful-data envelopes",
            "wrench response Jacobians remain zero; hold wrench is gated before planning",
            "MuJoCo execution results are stored by the separate short-execution runner",
        ],
    }


def main() -> None:
    import pyarrow.parquet as parquet

    args = parse_args()
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = _load_completed(args.output_dir) if args.resume else {}
    learning_dir = args.learning_dir or args.dataset_root / "retrieval_cerebellum_learning"
    estimation_dir = (
        args.estimation_dir or args.dataset_root / "retrieval_cerebellum_estimation"
    )
    adapter = SuccessfulSkillSQPAdapter.load(learning_dir, estimation_dir)
    execution_limits = ExecutionSafetyLimits.from_successful_records(
        list(adapter.records.values())
    )
    target_depth, _ = adapter.family_constants[args.family_id]
    clearance = _round_radial_clearance_m(args.family_id)
    peg_tip_in_peg = _peg_tip_in_peg_m(args.family_id)

    requested = None if args.episodes is None else set(args.episodes)
    paths = _held_out_paths(learning_dir, requested)
    if args.max_episodes is not None:
        paths = paths[: args.max_episodes]
    if not paths:
        raise ValueError("no held-out episodes selected")
    if args.handoff_row is not None and len(paths) != 1:
        raise ValueError("handoff_row override requires exactly one episode")

    records = dict(completed)
    for path in paths:
        episode_index = int(path.stem.rsplit("_", 1)[-1])
        if episode_index in completed:
            print(f"episode={episode_index} skip=completed", flush=True)
            continue
        table = parquet.read_table(
            path,
            columns=[
                "split",
                "geometry_features",
                "proprio_action44",
                "demo_action44",
            ],
        )
        split = str(table["split"][0].as_py())
        features = np.asarray(table["geometry_features"].to_pylist(), dtype=np.float64)
        proprio_actions = np.asarray(
            table["proprio_action44"].to_pylist(),
            dtype=np.float64,
        )
        demo_actions = np.asarray(
            table["demo_action44"].to_pylist(),
            dtype=np.float64,
        )
        handoff_selection = None
        if args.handoff_row is not None:
            handoff_row = args.handoff_row
            if not 0 <= handoff_row < len(features):
                raise IndexError(f"handoff row {handoff_row} is outside episode rows")
        elif args.handoff_policy == "stable":
            try:
                handoff_selection = select_stable_handoff_row(
                    features,
                    maximum_depth_advance_m=execution_limits.maximum_depth_advance_m,
                    right_attachment_translation_step_m=(
                        execution_limits.right_attachment_translation_step_m
                    ),
                    left_attachment_translation_step_m=(
                        execution_limits.left_attachment_translation_step_m
                    ),
                    right_attachment_rotation_step_rad=(
                        execution_limits.right_attachment_rotation_step_rad
                    ),
                    left_attachment_rotation_step_rad=(
                        execution_limits.left_attachment_rotation_step_rad
                    ),
                    config=StableHandoffConfig(
                        maximum_lateral_error_m=2.0 * clearance,
                    ),
                )
            except ValueError as error:
                record = {
                    "episode_index": episode_index,
                    "split": split,
                    "handoff_row": None,
                    "entry_row": final_entry_row(features),
                    "handoff_selection": None,
                    "success": False,
                    "rejection_reason": f"handoff_infeasible: {error}",
                    "selected_skill_id": None,
                    "action_plan_path": None,
                    "candidate_results": [],
                }
                records[episode_index] = record
                _write_json(
                    args.output_dir
                    / "episodes"
                    / f"episode_{episode_index:06d}.json",
                    record,
                )
                print(
                    f"episode={episode_index} split={split} rejected=handoff_infeasible",
                    flush=True,
                )
                continue
            handoff_row = handoff_selection.row
        else:
            handoff_row = _entry_handoff_row(features)
        belief18 = features[handoff_row, :18]
        mean = assembly_state_from_belief18(belief18)
        observed_action44 = proprio_actions[handoff_row]
        current_command44 = demo_actions[handoff_row]
        right_palm_world = palm_pose_from_action44(observed_action44, side="right")
        left_palm_world = palm_pose_from_action44(observed_action44, side="left")
        kinematics = BimanualAssemblyKinematics.from_observation(
            right_palm_world=right_palm_world,
            left_palm_world=left_palm_world,
            peg_in_right_palm=pose_matrix(belief18[6:9], belief18[9:12]),
            peg_tip_in_hole_position=belief18[:3],
            peg_in_hole_rotvec=belief18[3:6],
            peg_tip_in_peg=peg_tip_in_peg,
        )
        geometric_right_jacobian, geometric_left_jacobian = kinematics.state_jacobians(
            right_palm_world,
            left_palm_world,
        )
        state_drift = np.zeros(5, dtype=np.float64)
        right_state_jacobian = geometric_right_jacobian
        left_state_jacobian = geometric_left_jacobian
        dynamics_source = "nonlinear SE(3) kinematics"
        control_repeats = 1
        dynamics_translation_step_m = None
        dynamics_rotation_step_rad = None
        baseline_wrench_reasons = []
        if args.dynamics_dir is not None:
            dynamics_path = (
                args.dynamics_dir
                / "episodes"
                / f"episode_{episode_index:06d}.npz"
            )
            if not dynamics_path.exists():
                raise FileNotFoundError(
                    f"missing local dynamics model for episode {episode_index}: "
                    f"{dynamics_path}"
                )
            dynamics = np.load(dynamics_path)
            state_drift = np.asarray(dynamics["drift"], dtype=np.float64)
            right_state_jacobian = np.asarray(
                dynamics["right_state_jacobian"], dtype=np.float64
            )
            left_state_jacobian = np.asarray(
                dynamics["left_state_jacobian"], dtype=np.float64
            )
            dynamics_source = str(dynamics_path)
            if "rollout_steps" in dynamics:
                control_repeats = int(dynamics["rollout_steps"])
            dynamics_metadata = json.loads(
                dynamics_path.with_suffix(".json").read_text()
            )
            dynamics_translation_step_m = float(
                dynamics_metadata["translation_step_m"]
            )
            dynamics_rotation_step_rad = float(
                dynamics_metadata["rotation_step_rad"]
            )
            if "hold_wrist_ft_right" in dynamics:
                hold_right = np.asarray(
                    dynamics["hold_wrist_ft_right"], dtype=np.float64
                )
                hold_left = np.asarray(
                    dynamics["hold_wrist_ft_left"], dtype=np.float64
                )
                if np.linalg.norm(hold_right[:3]) > execution_limits.right_force_norm_max_n:
                    baseline_wrench_reasons.append("right_hold_force_exceeds_envelope")
                if np.linalg.norm(hold_left[:3]) > execution_limits.left_force_norm_max_n:
                    baseline_wrench_reasons.append("left_hold_force_exceeds_envelope")
                if np.linalg.norm(hold_right[3:]) > execution_limits.right_torque_norm_max_nm:
                    baseline_wrench_reasons.append("right_hold_torque_exceeds_envelope")
                if np.linalg.norm(hold_left[3:]) > execution_limits.left_torque_norm_max_nm:
                    baseline_wrench_reasons.append("left_hold_torque_exceeds_envelope")
            if baseline_wrench_reasons:
                record = {
                    "episode_index": episode_index,
                    "split": split,
                    "handoff_row": handoff_row,
                    "entry_row": final_entry_row(features),
                    "handoff_selection": (
                        None
                        if handoff_selection is None
                        else handoff_selection.to_dict()
                    ),
                    "handoff_state5": mean.tolist(),
                    "dynamics_source": dynamics_source,
                    "success": False,
                    "rejection_reason": "baseline_wrench_infeasible",
                    "baseline_wrench_reasons": baseline_wrench_reasons,
                    "selected_skill_id": None,
                    "action_plan_path": None,
                    "candidate_results": [],
                }
                records[episode_index] = record
                _write_json(
                    args.output_dir
                    / "episodes"
                    / f"episode_{episode_index:06d}.json",
                    record,
                )
                print(
                    f"episode={episode_index} split={split} "
                    "rejected=baseline_wrench_infeasible",
                    flush=True,
                )
                continue
        model = LocalBimanualInsertionModel(
            right_state_jacobian=right_state_jacobian,
            left_state_jacobian=left_state_jacobian,
            right_wrench_jacobian=np.zeros((6, 6)),
            left_wrench_jacobian=np.zeros((6, 6)),
            state_drift=state_drift,
        )
        planning_horizon = 1 if args.dynamics_dir is not None else args.horizon
        if args.dynamics_dir is None:
            planning_terminal_depth = min(
                target_depth,
                float(mean[4])
                + planning_horizon * execution_limits.maximum_depth_advance_m,
            )
        else:
            planning_terminal_depth = min(
                target_depth,
                float(mean[4]) + max(0.0, float(state_drift[4])),
            )
        sqp_config = BeliefSpaceSQPConfig(
            use_candidate_transition_residual=args.dynamics_dir is None,
            max_total_control=(
                None
                if args.dynamics_dir is None
                else (
                    2.0 * dynamics_translation_step_m,
                    2.0 * dynamics_translation_step_m,
                    2.0 * dynamics_translation_step_m,
                    2.0 * dynamics_rotation_step_rad,
                    2.0 * dynamics_rotation_step_rad,
                    2.0 * dynamics_rotation_step_rad,
                )
            )
        )
        solver = RetrievalConditionedBeliefSpaceSQP(
            model,
            InsertionGeometry(
                radial_clearance_m=clearance,
                target_depth_m=target_depth,
                terminal_depth_m=planning_terminal_depth,
            ),
            sqp_config,
        )
        candidates = adapter.retrieve_candidates(
            belief18,
            family_id=args.family_id,
            top_k=args.top_k,
            horizon=planning_horizon,
            source_span_steps=(
                planning_horizon if args.dynamics_dir is not None else None
            ),
        )
        belief = BimanualInsertionBelief(
            mean=mean,
            covariance=np.diag([1e-10, 1e-10, 1e-8, 1e-8, 1e-10]),
            attachment_process_covariance=np.eye(12) * 1e-11,
            attachment_to_state=np.concatenate(
                [geometric_right_jacobian, geometric_left_jacobian],
                axis=1,
            ),
        )
        results = tuple(solver.solve_candidate(belief, item) for item in candidates)
        successful_feasible = [
            item for item in results if item.success and item.feasible
        ]
        selected = (
            None
            if not successful_feasible
            else min(
                successful_feasible,
                key=lambda item: (item.objective, item.skill_id),
            )
        )
        selected_candidate = None
        action_plan = None
        first_step_linearization_error = None
        if selected is not None and selected.success:
            selected_candidate = next(
                item for item in candidates if item.skill_id == selected.skill_id
            )
            finger_references = align_and_project_finger_references(
                current_command44,
                selected_candidate.nominal_actions44,
                adapter.finger_step_limit,
            )
            action_plan = bimanual_controls_to_action44(
                current_command44,
                selected.right_controls,
                selected.left_controls,
                finger_references44=finger_references,
            )
            linear_first_state = (
                mean
                + state_drift
                + right_state_jacobian @ selected.right_controls[0]
                + left_state_jacobian @ selected.left_controls[0]
            )
            if args.dynamics_dir is None:
                nonlinear_first_state = kinematics.assembly_state(
                    palm_pose_from_action44(action_plan[0], side="right"),
                    palm_pose_from_action44(action_plan[0], side="left"),
                )
                first_step_linearization_error = (
                    nonlinear_first_state - linear_first_state
                )
            else:
                nonlinear_first_state = linear_first_state.copy()
            (args.output_dir / "episodes").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.output_dir
                / "episodes"
                / f"episode_{episode_index:06d}_action_plan.npz",
                action44=action_plan,
                right_controls=selected.right_controls,
                left_controls=selected.left_controls,
                right_state_jacobian=right_state_jacobian,
                left_state_jacobian=left_state_jacobian,
                geometric_right_state_jacobian=geometric_right_jacobian,
                geometric_left_state_jacobian=geometric_left_jacobian,
                state_drift5=state_drift,
                control_repeats=np.asarray(control_repeats, dtype=np.int64),
                nonlinear_first_state5=nonlinear_first_state,
                linear_first_state5=linear_first_state,
            )
        record = {
            "episode_index": episode_index,
            "split": split,
            "handoff_row": handoff_row,
            "entry_row": final_entry_row(features),
            "handoff_selection": (
                None if handoff_selection is None else handoff_selection.to_dict()
            ),
            "handoff_state5": mean.tolist(),
            "planning_terminal_depth_m": planning_terminal_depth,
            "planning_horizon": planning_horizon,
            "dynamics_source": dynamics_source,
            "state_drift5": state_drift.tolist(),
            "control_repeats": control_repeats,
            "max_total_control": (
                None
                if sqp_config.max_total_control is None
                else list(sqp_config.max_total_control)
            ),
            "use_candidate_transition_residual": (
                sqp_config.use_candidate_transition_residual
            ),
            "success": bool(selected is not None and selected.success),
            "selected_skill_id": None if selected is None else selected.skill_id,
            "selected_objective": None if selected is None else selected.objective,
            "selected_min_margin": (
                None if selected is None else selected.min_constraint_margin
            ),
            "state_jacobian_condition": float(
                np.linalg.cond(
                    np.concatenate(
                        [right_state_jacobian, left_state_jacobian],
                        axis=1,
                    )
                )
            ),
            "geometric_state_jacobian_condition": float(
                np.linalg.cond(
                    np.concatenate(
                        [geometric_right_jacobian, geometric_left_jacobian],
                        axis=1,
                    )
                )
            ),
            "first_step_linearization_error5": (
                None
                if first_step_linearization_error is None
                else first_step_linearization_error.tolist()
            ),
            "action_plan_path": (
                None
                if action_plan is None
                else str(
                    args.output_dir
                    / "episodes"
                    / f"episode_{episode_index:06d}_action_plan.npz"
                )
            ),
            "terminal_state5": None if selected is None else selected.states[-1].tolist(),
            "candidate_results": [
                {
                    "skill_id": item.skill_id,
                    "success": item.success,
                    "feasible": item.feasible,
                    "objective": item.objective,
                    "min_constraint_margin": item.min_constraint_margin,
                    "message": item.message,
                }
                for item in results
            ],
        }
        records[episode_index] = record
        _write_json(
            args.output_dir / "episodes" / f"episode_{episode_index:06d}.json",
            record,
        )
        current_records = [records[key] for key in sorted(records)]
        _write_json(
            args.output_dir / "summary.json",
            _summary(
                current_records,
                family_id=args.family_id,
                clearance=clearance,
                target_depth=target_depth,
                top_k=args.top_k,
                horizon=args.horizon,
                handoff_policy=args.handoff_policy,
            ),
        )
        print(
            f"episode={episode_index} split={split} success={record['success']} "
            f"selected={record['selected_skill_id']}",
            flush=True,
        )

    final_records = [records[key] for key in sorted(records)]
    summary = _summary(
        final_records,
        family_id=args.family_id,
        clearance=clearance,
        target_depth=target_depth,
        top_k=args.top_k,
        horizon=args.horizon,
        handoff_policy=args.handoff_policy,
    )
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
