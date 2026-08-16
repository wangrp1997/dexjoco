"""Matched handoff-state micro-perturbations for recoverability audit."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from embodied_grasp_insertion.physics.grasp_metrics import REFERENCE_BODY, object_in_hand_pose
from embodied_grasp_insertion.simulation.calibrated_interventions import RIGHT_FINGER_JOINT_NAMES
from interaction_retarget.skill_replay.insert import _insert_geometry

# Flexion-like joints within Allegro right 16 (exclude abduction 0/4/8/12).
_FLEX_LOCAL = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15)


def socket_basis(raw) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (socket_origin, e_lat_x, e_lat_y, hole_axis_unit)."""
    _tip, socket, hole, _ = _insert_geometry(raw)
    z = np.asarray(hole, dtype=np.float64).reshape(3)
    z = z / max(float(np.linalg.norm(z)), 1e-9)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, z))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x = np.cross(helper, z)
    x = x / max(float(np.linalg.norm(x)), 1e-9)
    y = np.cross(z, x)
    y = y / max(float(np.linalg.norm(y)), 1e-9)
    return np.asarray(socket, dtype=np.float64).copy(), x, y, z


def _zero_peg_vel(raw) -> None:
    adr = int(raw._peg_qvel_adr)
    raw._data.qvel[adr : adr + 6] = 0.0


def _read_peg7(raw) -> tuple[np.ndarray, np.ndarray]:
    adr = int(raw._peg_qpos_adr)
    q = np.asarray(raw._data.qpos[adr : adr + 7], dtype=np.float64).copy()
    return q[:3], q[3:7]


def _write_peg7(raw, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    raw._set_free_joint_pose(
        int(raw._peg_qpos_adr),
        int(raw._peg_qvel_adr),
        np.asarray(pos, dtype=np.float64),
        np.asarray(quat_wxyz, dtype=np.float64),
    )
    _zero_peg_vel(raw)
    mujoco.mj_forward(raw._model, raw._data)


def apply_tip_lat(raw, *, dx: float, dy: float) -> dict[str, Any]:
    _sock, ex, ey, _ez = socket_basis(raw)
    pos, quat = _read_peg7(raw)
    delta = float(dx) * ex + float(dy) * ey
    _write_peg7(raw, pos + delta, quat)
    return {"delta_world_m": delta.tolist(), "dx": float(dx), "dy": float(dy)}


def apply_tip_along(raw, *, dz: float) -> dict[str, Any]:
    _sock, _ex, _ey, ez = socket_basis(raw)
    pos, quat = _read_peg7(raw)
    delta = float(dz) * ez
    _write_peg7(raw, pos + delta, quat)
    return {"delta_world_m": delta.tolist(), "dz": float(dz)}


def apply_axis_tilt(raw, *, axis_xyz: np.ndarray, angle_rad: float) -> dict[str, Any]:
    """Rotate peg about world axis through peg origin (socket-lateral axes preferred)."""
    pos, quat = _read_peg7(raw)
    ax = np.asarray(axis_xyz, dtype=np.float64).reshape(3)
    # Map abstract [1,0,0]/[0,1,0] onto socket lateral basis.
    _sock, ex, ey, ez = socket_basis(raw)
    world_ax = float(ax[0]) * ex + float(ax[1]) * ey + float(ax[2]) * ez
    n = float(np.linalg.norm(world_ax))
    if n < 1e-9:
        world_ax = ex
    else:
        world_ax = world_ax / n
    R0 = R.from_quat(quat, scalar_first=True)
    R1 = R.from_rotvec(world_ax * float(angle_rad)) * R0
    _write_peg7(raw, pos, R1.as_quat(scalar_first=True))
    return {
        "world_axis": world_ax.tolist(),
        "angle_rad": float(angle_rad),
    }


def apply_o2h_shift(raw, *, axis_xyz: np.ndarray, dist_m: float) -> dict[str, Any]:
    """Translate peg in palm frame; keep relative orientation."""
    model, data = raw._model, raw._data
    palm_id = int(model.body(REFERENCE_BODY).id)
    palm_pos = np.asarray(data.xpos[palm_id], dtype=np.float64)
    palm_R = R.from_matrix(np.asarray(data.xmat[palm_id], dtype=np.float64).reshape(3, 3))
    o2h = object_in_hand_pose(raw)
    ax = np.asarray(axis_xyz, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(ax))
    ax = ax / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    local_t = np.asarray(o2h.translation, dtype=np.float64) + ax * float(dist_m)
    local_rv = np.asarray(o2h.rotvec, dtype=np.float64)
    peg_pos = palm_pos + palm_R.apply(local_t)
    peg_quat = (palm_R * R.from_rotvec(local_rv)).as_quat(scalar_first=True)
    _write_peg7(raw, peg_pos, peg_quat)
    return {
        "local_delta_m": (ax * float(dist_m)).tolist(),
        "local_t_after": local_t.tolist(),
    }


def apply_finger_close(raw, *, flex_rad: float) -> dict[str, Any]:
    """Add flexion to right finger joints (+ctrl); does not move peg freejoint."""
    model, data = raw._model, raw._data
    touched = []
    for li in _FLEX_LOCAL:
        jn = RIGHT_FINGER_JOINT_NAMES[li]
        jid = int(model.joint(jn).id)
        qadr = int(model.jnt_qposadr[jid])
        before = float(data.qpos[qadr])
        after = before + float(flex_rad)
        data.qpos[qadr] = after
        # Mirror into ctrl if actuator exists with same local index (assembly: 7..22).
        act_id = 7 + int(li)
        if 0 <= act_id < int(model.nu):
            lo, hi = np.asarray(model.actuator_ctrlrange[act_id], dtype=np.float64)
            data.ctrl[act_id] = float(np.clip(float(data.ctrl[act_id]) + float(flex_rad), lo, hi))
        touched.append({"joint": jn, "before": before, "after": after})
    _zero_peg_vel(raw)
    mujoco.mj_forward(model, data)
    return {"flex_rad": float(flex_rad), "joints": touched}


def apply_perturbation(
    raw,
    *,
    kind: str,
    scale: float,
    base: dict[str, float],
    pert_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Apply one matched micro-perturbation. scale=0 or kind=none → no-op."""
    s = float(scale)
    k = str(kind)
    meta: dict[str, Any] = {"kind": k, "scale": s}
    if k in ("none", "identity") or s == 0.0:
        meta["applied"] = False
        return meta
    meta["applied"] = True
    if k == "tip_lat":
        sign = np.asarray(pert_cfg.get("sign", [1.0, 0.0]), dtype=np.float64).reshape(-1)
        dx = s * float(base["tip_lat_m"]) * float(sign[0])
        dy = s * float(base["tip_lat_m"]) * float(sign[1] if len(sign) > 1 else 0.0)
        meta.update(apply_tip_lat(raw, dx=dx, dy=dy))
    elif k == "tip_along":
        sign = float(pert_cfg.get("sign", 1.0))
        meta.update(apply_tip_along(raw, dz=s * float(base["tip_along_m"]) * sign))
    elif k == "axis":
        ax = np.asarray(pert_cfg.get("axis", [1.0, 0.0, 0.0]), dtype=np.float64)
        meta.update(apply_axis_tilt(raw, axis_xyz=ax, angle_rad=s * float(base["axis_rad"])))
    elif k == "o2h":
        ax = np.asarray(pert_cfg.get("axis", [1.0, 0.0, 0.0]), dtype=np.float64)
        meta.update(apply_o2h_shift(raw, axis_xyz=ax, dist_m=s * float(base["o2h_trans_m"])))
    elif k == "finger":
        # mode close → positive flex
        meta.update(apply_finger_close(raw, flex_rad=s * float(base["finger_flex_rad"])))
    else:
        raise ValueError(f"unknown perturbation kind: {k}")
    return meta
