"""Reach / alignment metrics on MuJoCo geom (not mocap command)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from interaction_retarget.constants import LEFT_HAND_BODIES, PEG_BODY, RIGHT_HAND_BODIES, TRAY_BODY
from interaction_retarget.grasp.ik import interaction_metrics_obj_frame
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.hand_geom import hand_keypoints_world
from interaction_retarget.sim.settle import vec_to_arm_action
from interaction_retarget.transforms import world_to_object

ObjectName = Literal["tray", "peg"]
Side = Literal["left", "right"]


def _object_body(object_name: ObjectName) -> str:
    return TRAY_BODY if object_name == "tray" else PEG_BODY


def _hand_bodies(side: Side) -> tuple[str, ...]:
    return LEFT_HAND_BODIES if side == "left" else RIGHT_HAND_BODIES


def site_err_m(raw_env, side: Side, target_mocap_pos: np.ndarray) -> float:
    """Wrist site vs commanded mocap position (m)."""
    site_id = int(raw_env._site_left_id if side == "left" else raw_env._site_right_id)
    site_pos = np.asarray(raw_env._data.site_xpos[site_id], dtype=np.float64)
    target = np.asarray(target_mocap_pos, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(site_pos - target))


def hand_rmse_obj_m(raw_env, canonical: dict, *, side: Side, object_name: ObjectName) -> float:
    """Hand keypoint geom RMSE in object frame vs δ* hand_points_obj (m)."""
    obj_body = _object_body(object_name)
    model = raw_env._model
    data = raw_env._data
    obj_id = model.body(obj_body).id
    obj_pos = np.asarray(data.xpos[obj_id], dtype=np.float64)
    obj_quat = np.asarray(data.xquat[obj_id], dtype=np.float64)
    hand_w = hand_keypoints_world(model, data, _hand_bodies(side))
    hand_o = world_to_object(hand_w, obj_pos, obj_quat)
    target = np.asarray(canonical["hand_points_obj"], dtype=np.float64)
    return float(np.sqrt(np.mean((hand_o - target) ** 2)))


def laplacian_rmse_obj_m(raw_env, canonical: dict, *, side: Side, object_name: ObjectName) -> float:
    obj_body = _object_body(object_name)
    _, metrics = interaction_metrics_obj_frame(
        raw_env._model,
        raw_env._data,
        side=side,
        obj_body=obj_body,
        target_hand_obj=canonical["hand_points_obj"],
        target_obj_samples_obj=canonical["object_samples_obj"],
        target_laplacian=canonical["laplacian_coords"],
        adjacency=canonical["adjacency"],
    )
    return float(metrics["laplacian_rmse_m"])


@dataclass
class ReachMetrics:
    site_err_m: float
    hand_rmse_m: float
    laplacian_rmse_m: float
    contact_count: int
    home_to_target_m: float
    total_steps: int
    site_converged: bool
    hand_converged: bool
    laplacian_converged: bool


def measure_reach(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    canonical: dict,
    target23: np.ndarray,
    home23: np.ndarray,
    detector: AssemblyContactDetector | None,
    total_steps: int,
    site_err_gate_m: float,
    hand_rmse_gate_m: float,
    laplacian_gate_m: float | None = None,
) -> ReachMetrics:
    target23 = vec_to_arm_action(target23)
    home23 = vec_to_arm_action(home23)
    contact = 0
    if detector is not None:
        c = detector.compute(raw_env)
        contact = int(c.tray_contact_count if object_name == "tray" else c.peg_contact_count)
    site_e = site_err_m(raw_env, side, target23[:3])
    hand_e = hand_rmse_obj_m(raw_env, canonical, side=side, object_name=object_name)
    lap_e = laplacian_rmse_obj_m(raw_env, canonical, side=side, object_name=object_name)
    lap_gate = float(laplacian_gate_m if laplacian_gate_m is not None else hand_rmse_gate_m)
    return ReachMetrics(
        site_err_m=site_e,
        hand_rmse_m=hand_e,
        laplacian_rmse_m=lap_e,
        contact_count=contact,
        home_to_target_m=float(np.linalg.norm(target23[:3] - home23[:3])),
        total_steps=int(total_steps),
        site_converged=site_e <= site_err_gate_m,
        hand_converged=hand_e <= hand_rmse_gate_m,
        laplacian_converged=lap_e <= lap_gate,
    )
