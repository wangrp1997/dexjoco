#!/usr/bin/env python3
"""Run ECHO-Insert after a demo/privileged diagnostic handoff fixture."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEXJOCO_ROOT = _REPO_ROOT / "dexjoco"
for _path in (_REPO_ROOT, _DEXJOCO_ROOT, _REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv
from dexjoco_openpi_client.eval_dexjoco_openpi import _append_video_frames
from dexjoco.sim.envs.assembly_geometry import names_from_raw
from echo_insert.audit import audit_action_path, audit_sim_boundary
from echo_insert.controller import EchoConfig, EchoController
from echo_insert.kinematic_estimator import (
    UnobservableTaskFrame,
    estimate_task_frame,
)
from echo_insert.optimizer import OptimizerConfig
from echo_insert.public_io import (
    PublicObservation,
    apply_right_tip_pivot_action,
    state46_to_action44,
)
from echo_insert.sim_depth import read_ego_depth_m
from echo_insert.sim_wrist import read_wrist_wrench_local
from interaction_retarget.constants import default_sidecar_dir
from interaction_retarget.grasp.repair import _hand_joint_bounds
from interaction_retarget.sim.settle import read_arm_action
from interaction_retarget.skill_replay.insert import (
    _insert_geometry,
    dual_arm23_to_action44,
)
from hybrid_insert.geometry import (
    axis_parallel_error_rad,
    body_z_axis,
    height_along_axis,
    lateral_error,
    line_align_target_axis,
    pbvs_tip_feature_error,
    rotation_world_from_to,
)
from hybrid_insert.config import HybridInsertConfig
from scripts.eval_openpi_demo_handoff_insert import (
    CAMERA_MAPPING,
    PROMPT,
    _manifest_entries,
    _prepare_handoff,
    _raw_env,
)


COMPLIANCE_HEADER = (
    "合规状态: 不合规\n"
    "符合约束的完整成功: 0/0 (未完成合规评估)"
)
PROTOCOL = "insertion_only / privileged_diagnostic"
ENVIRONMENT_STEP_LIMIT = 1500
PRIVILEGED_CONTROLLER_STEP_LIMIT = 3000
PRIVILEGED_LEFT_GRIP_DELTA_RAD = -0.012
PRIVILEGED_PRECONTACT_SURFACE_ALONG_M = 0.100
POLICY_OBSERVATION = (
    "46-D state + previous 44-D action + 2x6 wrist F/T + "
    "one reset task frame (RGB-D or explicitly privileged MuJoCo geometry)"
)
TRAINING_FEEDBACK = "none"
FORBIDDEN_SOURCES = (
    "none claimed for this privileged diagnostic; demo handoff and optional "
    "simulator task geometry make it ineligible"
)
FULL_SUCCESS = 'native DexJoCo/OpenPI info["succeed"]'


def _csv_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one episode index")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=_csv_ints, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        default=default_sidecar_dir("bimanual_assembly"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-controller-steps", type=int)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--direct-demo-handoff", action="store_true")
    parser.add_argument("--privileged-task-frame", action="store_true")
    parser.add_argument("--power-sign", type=int, choices=(-1, 1), required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _public_observation(
    wrapper: DexJoCoOpenPIEnv,
    raw: Any,
    previous_action44: np.ndarray,
    *,
    ego_depth_m: np.ndarray | None = None,
) -> PublicObservation:
    state46 = np.asarray(wrapper.get_obs()["state"], dtype=np.float64)
    return PublicObservation(
        state46=state46,
        previous_action44=previous_action44,
        wrist_wrench_local=read_wrist_wrench_local(raw),
        fingertip_load=None,
        ego_depth_m=ego_depth_m,
    )


def _diagnostics_dict(diagnostics: Any) -> dict[str, Any]:
    if is_dataclass(diagnostics):
        values = asdict(diagnostics)
    elif isinstance(diagnostics, Mapping):
        values = dict(diagnostics)
    else:
        values = {
            name: getattr(diagnostics, name)
            for name in ("status", "selected_u5", "safety_reason")
            if hasattr(diagnostics, name)
        }
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in values.items()
    }


def _run_privileged_precontact(
    wrapper: DexJoCoOpenPIEnv,
    *,
    handoff_action44: np.ndarray,
    privileged_labeler: Any,
    max_steps: int,
    video_cb: Callable[[dict], None] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dynamically servo the true peg tip to a short precontact standoff."""
    raw = _raw_env(wrapper.env)
    cfg = HybridInsertConfig()
    echo_cfg = EchoConfig()
    names = names_from_raw(raw)
    peg_id = int(raw._model.body(names.peg_body).id)
    state46 = np.asarray(wrapper.get_obs()["state"], dtype=np.float64)
    frozen_action44 = state46_to_action44(state46).copy()
    frozen_action44[6:22] = handoff_action44[6:22]
    frozen_action44[28:44] = handoff_action44[28:44]
    previous_action44 = np.asarray(handoff_action44, dtype=np.float64).copy()
    target_along_m = PRIVILEGED_PRECONTACT_SURFACE_ALONG_M
    settle_frames = 0
    steps = 0
    peg_lost_streak = 0
    tray_lost_streak = 0
    insert_contact_streak = 0
    max_insert_contact_streak = 0
    ever_insert_contact = False
    status_counts: Counter[str] = Counter()
    trace: list[dict[str, Any]] = []
    stop_reason = ""
    monitor = {
        "minimum_tip_socket_distance_m": float("inf"),
        "minimum_abs_surface_along_m": float("inf"),
        "minimum_lateral_error_m": float("inf"),
        "minimum_axis_error_rad": float("inf"),
    }

    while steps < max_steps and not wrapper.is_done and not wrapper.is_success:
        state46 = np.asarray(wrapper.get_obs()["state"], dtype=np.float64)
        tip, socket, hole_axis, tip_socket_distance = _insert_geometry(raw)
        peg_axis = body_z_axis(raw._data.xmat[peg_id])
        insertion_axis = -np.asarray(hole_axis, dtype=np.float64)
        target_axis = line_align_target_axis(peg_axis, insertion_axis)
        axis_error = axis_parallel_error_rad(peg_axis, insertion_axis)
        feat = pbvs_tip_feature_error(
            tip,
            socket,
            hole_axis,
            peg_axis,
            target_along_m=target_along_m,
        )
        lateral = float(feat["lat_err"])
        along = float(feat["along"])
        reached = (
            axis_error <= 0.08
            and lateral <= cfg.pos_tol_m
            and abs(along - target_along_m) <= cfg.pos_tol_m
        )
        settle_frames = settle_frames + 1 if reached else 0
        if settle_frames >= 5:
            break

        correction = rotation_world_from_to(peg_axis, target_axis).as_rotvec()
        correction_norm = float(np.linalg.norm(correction))
        if correction_norm > echo_cfg.alignment_step_rad:
            correction *= echo_cfg.alignment_step_rad / correction_norm

        if axis_error > 2.0 * cfg.angle_tol_rad:
            tip_delta = np.zeros(3, dtype=np.float64)
            left_delta = np.zeros(3, dtype=np.float64)
            status = "privileged_align"
        else:
            near_surface = along <= 0.070
            lat_vec = np.asarray(feat["lat_vec"], dtype=np.float64)
            if along > 0.160:
                tip_delta = -cfg.pbvs_lambda_xy * lat_vec
                left_delta = np.zeros(3, dtype=np.float64)
            else:
                tip_delta = np.zeros(3, dtype=np.float64)
                left_delta = cfg.pbvs_lambda_xy * lat_vec
            allow_z = (
                along > 0.070
                or (
                    lateral <= cfg.axis_align_max_lat_m
                    and axis_error <= cfg.angle_tol_rad
                )
            )
            if allow_z:
                axis = np.asarray(hole_axis, dtype=np.float64)
                axis /= np.linalg.norm(axis)
                along_delta = cfg.pbvs_lambda_z * float(feat["e_along"]) * axis
                if along <= 0.160:
                    tip_delta -= (1.0 - cfg.left_share_xy) * along_delta
                    left_delta += cfg.left_share_xy * along_delta
                else:
                    tip_delta -= along_delta
            max_step = (
                cfg.max_insert_z_step_m
                if near_surface
                else echo_cfg.centering_step_m
            )
            tip_norm = float(np.linalg.norm(tip_delta))
            if tip_norm > max_step:
                tip_delta *= max_step / tip_norm
            left_norm = float(np.linalg.norm(left_delta))
            if left_norm > max_step:
                left_delta *= max_step / left_norm
            status = "privileged_standoff" if near_surface else "privileged_center"

        current_action44 = state46_to_action44(state46)
        left_target_xyz = current_action44[22:25] + left_delta
        frozen_action44[22:25] = left_target_xyz

        action44 = apply_right_tip_pivot_action(
            state46,
            frozen_action44,
            correction,
            tip,
            target_pivot_world=tip + tip_delta,
        )
        wrapper.step(np.asarray(action44, dtype=np.float32))
        if video_cb is not None:
            video_cb(wrapper.get_raw_images())
        previous_action44 = np.asarray(action44, dtype=np.float64).copy()
        steps += 1
        status_counts[status] += 1

        outcome = privileged_labeler.compute(raw)
        insert_contact_streak = insert_contact_streak + 1 if outcome.insert_ok else 0
        max_insert_contact_streak = max(
            max_insert_contact_streak,
            insert_contact_streak,
        )
        ever_insert_contact = ever_insert_contact or bool(outcome.insert_ok)
        peg_lost_streak = 0 if outcome.peg_ok or outcome.insert_ok else peg_lost_streak + 1
        tray_lost_streak = 0 if outcome.tray_ok else tray_lost_streak + 1
        monitor.update(
            {
                "minimum_tip_socket_distance_m": min(
                    monitor["minimum_tip_socket_distance_m"], tip_socket_distance
                ),
                "minimum_abs_surface_along_m": min(
                    monitor["minimum_abs_surface_along_m"], abs(along)
                ),
                "minimum_lateral_error_m": min(
                    monitor["minimum_lateral_error_m"], lateral
                ),
                "minimum_axis_error_rad": min(
                    monitor["minimum_axis_error_rad"], axis_error
                ),
                "final_tip_socket_distance_m": tip_socket_distance,
                "final_surface_along_m": along,
                "final_lateral_error_m": lateral,
                "final_axis_error_rad": axis_error,
            }
        )
        if steps == 1 or steps % 25 == 0 or reached:
            trace.append(
                {
                    "step": steps,
                    "status": status,
                    "tip_socket_distance_m": tip_socket_distance,
                    "surface_along_m": along,
                    "lateral_error_m": lateral,
                    "axis_error_rad": axis_error,
                    "observed_left_xyz": state46[7:10].tolist(),
                    "commanded_left_xyz": left_target_xyz.tolist(),
                }
            )
        if peg_lost_streak >= 10:
            stop_reason = "privileged_peg_lost"
            break
        if tray_lost_streak >= 10:
            stop_reason = "privileged_tray_lost"
            break

    complete = settle_frames >= 5
    if not complete and not stop_reason:
        stop_reason = (
            "environment_done" if wrapper.is_done else "privileged_precontact_step_budget"
        )
    return previous_action44, {
        "complete": complete,
        "steps": steps,
        "stop_reason": stop_reason,
        "status_counts": dict(status_counts),
        "trace": trace,
        "geometry_monitor": monitor,
        "ever_insert_contact": ever_insert_contact,
        "max_insert_contact_streak": max_insert_contact_streak,
        "insert_contact_streak": insert_contact_streak,
    }


def _run_controller(
    wrapper: DexJoCoOpenPIEnv,
    *,
    handoff_action44: np.ndarray,
    privileged_labeler: Any,
    privileged_task_frame: bool,
    privileged_precontact: bool,
    power_sign: float,
    max_controller_steps: int,
    video_cb: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    raw = _raw_env(wrapper.env)
    state46 = np.asarray(wrapper.get_obs()["state"], dtype=np.float64)
    previous_action44 = np.asarray(handoff_action44, dtype=np.float64).copy()
    if previous_action44.shape != (44,):
        raise ValueError("handoff_action44 must have shape (44,)")
    precontact_result: dict[str, Any] = {
        "complete": True,
        "steps": 0,
        "stop_reason": "",
        "status_counts": {},
        "trace": [],
        "geometry_monitor": {},
        "ever_insert_contact": False,
        "max_insert_contact_streak": 0,
        "insert_contact_streak": 0,
    }
    if privileged_task_frame:
        left_hand_lo, left_hand_hi = _hand_joint_bounds(raw._model, "left")
        previous_action44[28:44] = np.clip(
            previous_action44[28:44] + PRIVILEGED_LEFT_GRIP_DELTA_RAD,
            left_hand_lo,
            left_hand_hi,
        )
        raw.env_step = ENVIRONMENT_STEP_LIMIT - max_controller_steps
        if privileged_precontact:
            previous_action44, precontact_result = _run_privileged_precontact(
                wrapper,
                handoff_action44=previous_action44,
                privileged_labeler=privileged_labeler,
                max_steps=max_controller_steps,
                video_cb=video_cb,
            )
        state46 = np.asarray(wrapper.get_obs()["state"], dtype=np.float64)
        tip, socket, hole_axis, approach_distance = _insert_geometry(raw)
        names = names_from_raw(raw)
        peg_id = int(raw._model.body(names.peg_body).id)
        socket_id = int(raw._model.site(names.socket_site).id)
        physical_peg_axis = body_z_axis(raw._data.xmat[peg_id])
        insertion_axis = -np.asarray(hole_axis, dtype=np.float64)
        peg_axis = (
            physical_peg_axis
            if float(physical_peg_axis @ insertion_axis) >= 0.0
            else -physical_peg_axis
        )
        tangent = np.asarray(
            raw._data.site_xmat[socket_id],
            dtype=np.float64,
        ).reshape(3, 3)[:, 0]
        x_axis = tangent - insertion_axis * float(insertion_axis @ tangent)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(insertion_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        basis_world = np.column_stack([x_axis, y_axis, insertion_axis])
        approach = socket - tip
        surface_distance = float(approach @ insertion_axis)
        lateral_distance = float(
            np.linalg.norm(approach - surface_distance * insertion_axis)
        )
        maximum_advance_offset_m = surface_distance + 0.0675
        task_frame_summary = {
            "source": "privileged_mujoco_task_frame_after_dynamic_precontact",
            "basis_world": basis_world.tolist(),
            "peg_axis_proxy_world": peg_axis.tolist(),
            "physical_peg_axis_world": physical_peg_axis.tolist(),
            "peg_insert_end_world": tip.tolist(),
            "tray_entry_center_world": socket.tolist(),
            "approach_distance_m": float(approach_distance),
            "surface_distance_m": surface_distance,
            "lateral_centering_distance_m": lateral_distance,
            "maximum_advance_offset_m": maximum_advance_offset_m,
        }
    initial_observation = _public_observation(wrapper, raw, previous_action44)
    if not privileged_task_frame:
        initial_observation = _public_observation(
            wrapper,
            raw,
            previous_action44,
            ego_depth_m=read_ego_depth_m(raw),
        )
        assert initial_observation.ego_depth_m is not None
        task_frame = estimate_task_frame(
            initial_observation.state46,
            initial_observation.ego_depth_m,
        )
        basis_world = task_frame.basis_world
        peg_axis = task_frame.peg_axis_proxy_world
        tip = task_frame.peg_insert_end_world
        socket = task_frame.tray_entry_center_world
        maximum_advance_offset_m = task_frame.maximum_advance_offset_m
        task_frame_summary = task_frame.summary_record()

    controller = EchoController(
        task_basis_world=basis_world,
        peg_axis_world=peg_axis,
        peg_insert_end_world=tip,
        tray_entry_center_world=socket,
        config=EchoConfig(
            baseline_steps=9,
            optimizer=OptimizerConfig(
                power_sign=power_sign,
                maximum_advance_offset_m=maximum_advance_offset_m,
            ),
        ),
    )
    controller.reset(initial_observation)

    steps = 0
    precontact_steps = int(precontact_result["steps"])
    echo_step_budget = max_controller_steps - precontact_steps
    controller_terminal_reason = (
        "" if precontact_result["complete"] else str(precontact_result["stop_reason"])
    )
    status_counts: Counter[str] = Counter(precontact_result["status_counts"])
    last_diagnostics: dict[str, Any] = {}
    progress_trace: list[dict[str, Any]] = list(precontact_result["trace"])
    last_status = ""
    peg_lost_streak = 0
    tray_lost_streak = 0
    insert_contact_streak = int(precontact_result["insert_contact_streak"])
    max_insert_contact_streak = int(precontact_result["max_insert_contact_streak"])
    ever_insert_contact = bool(precontact_result["ever_insert_contact"])
    ever_surface_contact = False
    monitor_peg_id = int(raw._model.body(names_from_raw(raw).peg_body).id)
    privileged_geometry_monitor = {
        "minimum_tip_socket_distance_m": float("inf"),
        "minimum_abs_surface_along_m": float("inf"),
        "minimum_lateral_error_m": float("inf"),
        "minimum_axis_error_rad": float("inf"),
        "maximum_peg_tray_contact_count": 0,
    }
    while (
        precontact_result["complete"]
        and steps < echo_step_budget
        and not wrapper.is_done
        and not wrapper.is_success
    ):
        observation = _public_observation(wrapper, raw, previous_action44)
        action44, diagnostics = controller.step(observation)
        wrapper.step(np.asarray(action44, dtype=np.float32))
        if video_cb is not None:
            video_cb(wrapper.get_raw_images())
        next_state46 = np.asarray(wrapper.get_obs()["state"], dtype=np.float64)
        privileged_outcome = privileged_labeler.compute(raw)
        monitor_tip, monitor_socket, monitor_hole_axis, monitor_distance = (
            _insert_geometry(raw)
        )
        monitor_along = height_along_axis(
            monitor_tip,
            monitor_socket,
            monitor_hole_axis,
        )
        monitor_lateral, _ = lateral_error(
            monitor_tip,
            monitor_socket,
            monitor_hole_axis,
        )
        monitor_axis_error = axis_parallel_error_rad(
            body_z_axis(raw._data.xmat[monitor_peg_id]),
            monitor_hole_axis,
        )
        privileged_geometry_monitor.update(
            {
                "minimum_tip_socket_distance_m": min(
                    privileged_geometry_monitor["minimum_tip_socket_distance_m"],
                    monitor_distance,
                ),
                "minimum_abs_surface_along_m": min(
                    privileged_geometry_monitor["minimum_abs_surface_along_m"],
                    abs(monitor_along),
                ),
                "minimum_lateral_error_m": min(
                    privileged_geometry_monitor["minimum_lateral_error_m"],
                    monitor_lateral,
                ),
                "minimum_axis_error_rad": min(
                    privileged_geometry_monitor["minimum_axis_error_rad"],
                    monitor_axis_error,
                ),
                "maximum_peg_tray_contact_count": max(
                    privileged_geometry_monitor[
                        "maximum_peg_tray_contact_count"
                    ],
                    privileged_outcome.peg_tray_contact_count,
                ),
                "final_tip_socket_distance_m": monitor_distance,
                "final_surface_along_m": monitor_along,
                "final_lateral_error_m": monitor_lateral,
                "final_axis_error_rad": monitor_axis_error,
            }
        )
        insert_contact_streak = (
            insert_contact_streak + 1 if privileged_outcome.insert_ok else 0
        )
        max_insert_contact_streak = max(
            max_insert_contact_streak,
            insert_contact_streak,
        )
        ever_insert_contact = ever_insert_contact or bool(privileged_outcome.insert_ok)
        ever_surface_contact = (
            ever_surface_contact
            or privileged_outcome.peg_tray_contact_count > 0
        )
        peg_lost_streak = (
            0
            if privileged_outcome.peg_ok or privileged_outcome.insert_ok
            else peg_lost_streak + 1
        )
        tray_lost_streak = 0 if privileged_outcome.tray_ok else tray_lost_streak + 1
        previous_action44 = np.asarray(action44, dtype=np.float64).copy()
        steps += 1
        last_diagnostics = _diagnostics_dict(diagnostics)
        status = str(last_diagnostics.get("status", "unknown"))
        status_counts[status] += 1
        total_step = precontact_steps + steps
        if steps == 1 or steps % 25 == 0 or status != last_status:
            progress_trace.append(
                {
                    "step": total_step,
                    "status": status,
                    "axis_error_rad": last_diagnostics.get("axis_error_rad"),
                    "lateral_error_m": last_diagnostics.get("lateral_error_m"),
                    "selected_candidate": last_diagnostics.get("selected_candidate"),
                    "selected_u5": last_diagnostics.get("selected_u5"),
                    "wrench5": last_diagnostics.get("wrench5"),
                    "rls_updates": last_diagnostics.get("rls_updates"),
                    "search_cells": last_diagnostics.get("search_cells"),
                    "frontier_cells_remaining": last_diagnostics.get(
                        "frontier_cells_remaining"
                    ),
                    "axial_probe_cells": last_diagnostics.get("axial_probe_cells"),
                    "entry_mode": last_diagnostics.get(
                        "entry_mode"
                    ),
                    "recovering_interaction": last_diagnostics.get(
                        "recovering_interaction"
                    ),
                    "selected_score": last_diagnostics.get("selected_score"),
                    "information_gain": last_diagnostics.get("information_gain"),
                    "command_offset5": last_diagnostics.get("command_offset5"),
                    "safety_reason": last_diagnostics.get("safety_reason"),
                    "observed_right_pose7": observation.state46[:7].tolist(),
                    "next_right_pose7": next_state46[:7].tolist(),
                    "commanded_right_pose6": action44[:6].tolist(),
                }
            )
        last_status = status
        safety_reason = str(last_diagnostics.get("safety_reason", ""))
        if last_diagnostics.get("status") == "terminal":
            controller_terminal_reason = safety_reason or "controller_terminal"
            break
        if peg_lost_streak >= 10:
            controller_terminal_reason = "privileged_peg_lost"
            break
        if tray_lost_streak >= 10:
            controller_terminal_reason = "privileged_tray_lost"
            break

    final_outcome = privileged_labeler.compute(raw)
    return {
        "kinematic_task_frame": task_frame_summary,
        "controller_steps": precontact_steps + steps,
        "privileged_precontact_complete": bool(precontact_result["complete"]),
        "privileged_precontact_steps": precontact_steps,
        "privileged_precontact_stop_reason": precontact_result["stop_reason"],
        "privileged_precontact_geometry_monitor": precontact_result[
            "geometry_monitor"
        ],
        "echo_controller_steps": steps,
        "stop_reason": controller_terminal_reason
        or ("environment_done" if wrapper.is_done else "controller_step_budget"),
        "diagnostic_native_success": bool(wrapper.is_success),
        "ever_insert_contact": ever_insert_contact,
        "ever_surface_contact": ever_surface_contact,
        "privileged_left_grip_delta_rad": (
            PRIVILEGED_LEFT_GRIP_DELTA_RAD if privileged_task_frame else 0.0
        ),
        "privileged_left_preload_m": 0.0,
        "max_insert_contact_streak": max_insert_contact_streak,
        "final_tray_ok": bool(final_outcome.tray_ok),
        "final_peg_ok": bool(final_outcome.peg_ok),
        "final_insert_ok": bool(final_outcome.insert_ok),
        "privileged_geometry_monitor": privileged_geometry_monitor,
        "controller_status_counts": dict(status_counts),
        "last_controller_diagnostics": last_diagnostics,
        "controller_progress_trace": progress_trace,
    }


def main() -> int:
    print(COMPLIANCE_HEADER, flush=True)
    args = _parse_args()
    if args.max_controller_steps is not None and args.max_controller_steps <= 0:
        raise ValueError("max-controller-steps must be positive")
    audit_findings = audit_action_path() + audit_sim_boundary()
    if audit_findings:
        raise RuntimeError("ECHO provenance audit failed: " + "; ".join(audit_findings))
    output = args.output.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    handoff_mode = (
        "direct_demo" if args.direct_demo_handoff else "privileged_approach"
    )
    (output / "REPORT.md").write_text(
        COMPLIANCE_HEADER
        + "\n"
        + f"Protocol: {PROTOCOL}\n"
        + f"Handoff mode: {handoff_mode}\n"
        + f"policy_observation: {POLICY_OBSERVATION}\n"
        + f"training_feedback: {TRAINING_FEEDBACK}\n"
        + f"forbidden_sources: {FORBIDDEN_SOURCES}\n"
        + f"full_success: {FULL_SUCCESS}\n",
        encoding="utf-8",
    )
    sidecar_dir = args.sidecar_dir.expanduser().resolve()
    entries = _manifest_entries(sidecar_dir, args.episodes)

    wrapper = DexJoCoOpenPIEnv(
        env_name="bimanual_assembly",
        camera_mapping=CAMERA_MAPPING,
        seed=args.seed,
        rand_full=False,
        randomize_dynamics=False,
        dual_arm=True,
        prompt=PROMPT,
        render_mode="rgb_array",
    )
    wrapper.start()
    video_writers: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    try:
        for index, entry in enumerate(entries, start=1):
            episode = int(entry["episode_index"])
            print(f"诊断回合 {index}/{len(entries)}: demo={episode} handoff setup", flush=True)
            episode_video_dir = output / f"ep{episode:02d}"
            video_paths = {"ego": episode_video_dir / "ego.mp4"}
            video_frame_count = [0]
            video_cb = None
            if args.record_video:
                episode_video_dir.mkdir()
                video_writers = {
                    camera_name: imageio.get_writer(video_path, fps=30)
                    for camera_name, video_path in video_paths.items()
                }

                def append_video_frame(observation: dict) -> None:
                    _append_video_frames(
                        video_writers,
                        observation,
                        video_frame_count,
                    )

                video_cb = append_video_frame
            prepared = _prepare_handoff(
                wrapper,
                entry,
                sidecar_dir,
                video_cb=video_cb,
                direct_demo_handoff=args.direct_demo_handoff,
            )
            echo_start_video_frame = video_frame_count[0]
            labeler = prepared.pop("labeler")
            handoff_action44 = dual_arm23_to_action44(
                read_arm_action(_raw_env(wrapper.env), "left"),
                read_arm_action(_raw_env(wrapper.env), "right"),
            )
            setup_failure = prepared.pop("setup_failure")
            row: dict[str, Any] = {
                "compliance_status": "non-compliant",
                "protocol": PROTOCOL,
                "eligible_for_nonprivileged_full_task": False,
                "episode_index": episode,
                **prepared,
            }
            if int(row["handoff_env_step"]) >= ENVIRONMENT_STEP_LIMIT:
                setup_failure = "handoff_reached_environment_step_limit"
            elif bool(row["initial_insert_ok"]):
                setup_failure = "handoff_already_inserted"
            if setup_failure or not bool(row["initial_peg_ok"]):
                row.update(
                    {
                        "setup_ok": False,
                        "setup_failure": setup_failure or "peg_lost_before_controller",
                        "controller_steps": 0,
                        "stop_reason": "setup_failure",
                        "diagnostic_native_success": False,
                    }
                )
            else:
                try:
                    controller_result = _run_controller(
                        wrapper,
                        handoff_action44=handoff_action44,
                        privileged_labeler=labeler,
                        privileged_task_frame=bool(args.privileged_task_frame),
                        privileged_precontact=bool(
                            args.privileged_task_frame and args.direct_demo_handoff
                        ),
                        power_sign=float(args.power_sign),
                        max_controller_steps=(
                            args.max_controller_steps
                            if args.max_controller_steps is not None
                            else (
                                PRIVILEGED_CONTROLLER_STEP_LIMIT
                                if args.privileged_task_frame
                                else ENVIRONMENT_STEP_LIMIT
                                - int(row["handoff_env_step"])
                            )
                        ),
                        video_cb=video_cb,
                    )
                except UnobservableTaskFrame as error:
                    row.update(
                        {
                            "setup_ok": False,
                            "setup_failure": f"kinematic_task_frame_unobservable: {error}",
                            "controller_steps": 0,
                            "stop_reason": "setup_failure",
                            "diagnostic_native_success": False,
                        }
                    )
                else:
                    row["setup_ok"] = True
                    row["setup_failure"] = ""
                    row.update(controller_result)
            row["handoff_mode"] = (
                "direct_demo" if args.direct_demo_handoff else "privileged_approach"
            )
            row["task_frame_source"] = (
                "privileged_mujoco_task_frame"
                if args.privileged_task_frame
                else "public_rgbd_fk"
            )
            if args.record_video:
                row.update(
                    {
                        "video_path": str(video_paths["ego"]),
                        "video_paths": {
                            camera_name: str(video_path)
                            for camera_name, video_path in video_paths.items()
                        },
                        "video_frames": video_frame_count[0],
                        "echo_start_video_frame": echo_start_video_frame,
                    }
                )
                for writer in video_writers.values():
                    writer.close()
                video_writers = None
            rows.append(row)
            _write_json(output / f"episode_{episode:02d}.json", row)
    finally:
        if video_writers is not None:
            for writer in video_writers.values():
                writer.close()
        wrapper.close()

    evaluable = [row for row in rows if row["setup_ok"]]
    diagnostic_successes = sum(bool(row["diagnostic_native_success"]) for row in evaluable)
    summary = {
        "compliance_status": "不合规",
        "eligible_full_success": "0/0 (未完成合规评估)",
        "protocol": PROTOCOL,
        "handoff_mode": handoff_mode,
        "task_frame_source": (
            "privileged_mujoco_task_frame"
            if args.privileged_task_frame
            else "public_rgbd_fk"
        ),
        "policy_observation": POLICY_OBSERVATION,
        "training_feedback": TRAINING_FEEDBACK,
        "forbidden_sources": FORBIDDEN_SOURCES,
        "full_success": FULL_SUCCESS,
        "diagnostic_native_success": f"{diagnostic_successes}/{len(evaluable)}",
        "diagnostic_native_successes": diagnostic_successes,
        "diagnostic_evaluable_episodes": len(evaluable),
        "episodes_requested": args.episodes,
        "setup_failures": len(rows) - len(evaluable),
        "seed": args.seed,
        "max_controller_steps": args.max_controller_steps,
        "baseline_steps": 9,
        "wrist_power_sign": args.power_sign,
        "episodes": rows,
    }
    _write_json(output / "summary.json", summary)
    (output / "REPORT.md").write_text(
        COMPLIANCE_HEADER
        + "\n"
        + f"Diagnostic native success: {diagnostic_successes}/{len(evaluable)}\n"
        + f"Protocol: {PROTOCOL}\n"
        + f"Handoff mode: {handoff_mode}\n"
        + f"policy_observation: {POLICY_OBSERVATION}\n"
        + f"training_feedback: {TRAINING_FEEDBACK}\n"
        + f"forbidden_sources: {FORBIDDEN_SOURCES}\n"
        + f"full_success: {FULL_SUCCESS}\n",
        encoding="utf-8",
    )
    print(f"Diagnostic native success: {diagnostic_successes}/{len(evaluable)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
