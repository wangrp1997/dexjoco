"""Task topology constraints for TPSR (δ* drift + assembly hole clearance)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from interaction_retarget.constants import (
    FINGERTIP_KEYPOINT_INDICES,
    LEFT_HAND_BODIES,
    PEG_BODY,
    RIGHT_HAND_BODIES,
    TRAY_BODY,
)
from interaction_retarget.grasp.metrics import hand_rmse_obj_m, laplacian_rmse_obj_m
from interaction_retarget.sim.hand_geom import hand_keypoints_world

ObjectName = Literal["tray", "peg"]
Side = Literal["left", "right"]

_TRAY_SOCKET_SITE = "industreal_tray_insert_round_peg_8mm_socket_site"
_TRAY_BOTTOM_GEOM = "industreal_tray_insert_round_peg_8mm_bottom_contact"
_PEG_INSERT_END_BODY_OFFSET = np.array([0.0, 0.0, 0.0135], dtype=np.float64)


def _side_for_object(object_name: ObjectName) -> Side:
    return "left" if object_name == "tray" else "right"


def _hand_bodies(side: Side) -> tuple[str, ...]:
    return LEFT_HAND_BODIES if side == "left" else RIGHT_HAND_BODIES


def _body_z_axis(xmat: np.ndarray) -> np.ndarray:
    axis = np.asarray(xmat, dtype=np.float64).reshape(3, 3)[:, 2]
    n = float(np.linalg.norm(axis))
    return axis / n if n > 1e-8 else np.array([0.0, 0.0, 1.0])


def _fingertips_world(raw_env, side: Side) -> np.ndarray:
    bodies = _hand_bodies(side)
    hand_w = hand_keypoints_world(raw_env._model, raw_env._data, bodies)
    idx = list(FINGERTIP_KEYPOINT_INDICES)
    return np.asarray(hand_w[idx], dtype=np.float64)


def peg_insert_end_world(raw_env) -> tuple[np.ndarray, np.ndarray]:
    """Insert end position and body +Z axis (world)."""
    model = raw_env._model
    data = raw_env._data
    bid = int(model.body(PEG_BODY).id)
    pos = np.asarray(data.xpos[bid], dtype=np.float64)
    xmat = np.asarray(data.xmat[bid], dtype=np.float64)
    axis = _body_z_axis(xmat)
    end = pos + xmat.reshape(3, 3) @ _PEG_INSERT_END_BODY_OFFSET
    return end, axis


def tray_hole_axis_world(raw_env) -> tuple[np.ndarray, np.ndarray]:
    """Socket site position and hole opening axis (world, points out of hole)."""
    model = raw_env._model
    data = raw_env._data
    socket_id = int(model.site(_TRAY_SOCKET_SITE).id)
    socket_pos = np.asarray(data.site_xpos[socket_id], dtype=np.float64)
    socket_xmat = np.asarray(data.site_xmat[socket_id], dtype=np.float64)
    bottom_id = int(model.geom(_TRAY_BOTTOM_GEOM).id)
    bottom_pos = np.asarray(data.geom_xpos[bottom_id], dtype=np.float64)
    opening = socket_pos - bottom_pos
    n = float(np.linalg.norm(opening))
    axis = opening / n if n > 1e-8 else _body_z_axis(socket_xmat)
    return socket_pos, axis


def _cylinder_violation(
    points: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    *,
    radius_m: float,
    length_m: float,
) -> float:
    """Max lateral distance inside guard cylinder (0 = ok)."""
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    rel = np.asarray(points, dtype=np.float64).reshape(-1, 3) - origin.reshape(1, 3)
    along = rel @ axis
    lat = rel - np.outer(along, axis)
    lat_d = np.linalg.norm(lat, axis=1)
    mask = (along >= 0.0) & (along <= float(length_m))
    if not np.any(mask):
        return 0.0
    inside = lat_d[mask] - float(radius_m)
    return float(max(0.0, np.max(inside)))


def hole_clearance_violation_m(
    raw_env,
    *,
    object_name: ObjectName,
    side: Side,
    cfg_radius_m: float,
    cfg_length_m: float,
) -> float:
    if object_name == "tray":
        return 0.0
    tips = _fingertips_world(raw_env, side)
    origin, axis = peg_insert_end_world(raw_env)
    return _cylinder_violation(
        tips, origin, axis, radius_m=cfg_radius_m, length_m=cfg_length_m
    )


def topology_drift(
    raw_env,
    canonical: dict,
    *,
    object_name: ObjectName,
    baseline_lap_m: float | None,
    baseline_hand_m: float | None,
    max_lap_drift_m: float,
    max_hand_drift_m: float,
) -> tuple[bool, float, float]:
    side = _side_for_object(object_name)
    lap = laplacian_rmse_obj_m(raw_env, canonical, side=side, object_name=object_name)
    hand = hand_rmse_obj_m(raw_env, canonical, side=side, object_name=object_name)
    ok = True
    if baseline_lap_m is not None and lap > baseline_lap_m + max_lap_drift_m:
        ok = False
    if baseline_hand_m is not None and hand > baseline_hand_m + max_hand_drift_m:
        ok = False
    return ok, lap, hand


def candidate_acceptable(
    raw_env,
    canonical: dict,
    *,
    object_name: ObjectName,
    side: Side,
    baseline_lap_m: float | None,
    baseline_hand_m: float | None,
    max_lap_drift_m: float,
    max_hand_drift_m: float,
    hole_radius_m: float,
    hole_length_m: float,
) -> bool:
    topo_ok, _, _ = topology_drift(
        raw_env,
        canonical,
        object_name=object_name,
        baseline_lap_m=baseline_lap_m,
        baseline_hand_m=baseline_hand_m,
        max_lap_drift_m=max_lap_drift_m,
        max_hand_drift_m=max_hand_drift_m,
    )
    if not topo_ok:
        return False
    hole_v = hole_clearance_violation_m(
        raw_env,
        object_name=object_name,
        side=side,
        cfg_radius_m=hole_radius_m,
        cfg_length_m=hole_length_m,
    )
    return hole_v <= 1e-6
