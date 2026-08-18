"""Hard and soft object-in-hand projection for the independent benchmark."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from embodied_grasp_insertion.physics.grasp_metrics import (
    ObjectInHandPose,
    REFERENCE_BODY,
    names_from_raw,
    object_in_hand_pose,
)


def capture(raw) -> ObjectInHandPose:
    return object_in_hand_pose(raw, reference_body=REFERENCE_BODY)


def project(raw, pose: ObjectInHandPose, strength: float = 1.0) -> None:
    """Project the peg toward the locked pose with a tunable strength."""
    model, data = raw._model, raw._data
    bid = int(model.body(pose.reference_body).id)
    palm_pos = np.asarray(data.xpos[bid], dtype=np.float64)
    palm_rot = R.from_quat(np.asarray(data.xquat[bid]), scalar_first=True)
    target_pos = palm_pos + palm_rot.apply(pose.translation)
    target_quat = (palm_rot * R.from_rotvec(pose.rotvec)).as_quat(scalar_first=True)
    peg_id = int(model.body(names_from_raw(raw).peg_body).id)
    current_pos = np.asarray(data.xpos[peg_id], dtype=np.float64)
    current_quat = np.asarray(data.xquat[peg_id], dtype=np.float64)
    alpha = float(np.clip(strength, 0.0, 1.0))
    pos = current_pos + alpha * (target_pos - current_pos)
    cur_rot = R.from_quat(current_quat, scalar_first=True)
    rot = cur_rot * (cur_rot.inv() * R.from_quat(target_quat, scalar_first=True)).__pow__(alpha)
    raw._set_free_joint_pose(int(raw._peg_qpos_adr), int(raw._peg_qvel_adr), pos, rot.as_quat(scalar_first=True))
    data.qvel[int(raw._peg_qvel_adr) : int(raw._peg_qvel_adr) + 6] = 0.0
    mujoco.mj_forward(model, data)


def pin_pose(raw, pos: np.ndarray, rot: R) -> None:
    """Set a finite peg free-joint pose and clear its spatial velocity."""
    position = np.asarray(pos, dtype=np.float64).reshape(3)
    if not np.isfinite(position).all():
        raise ValueError("non-finite peg position")
    raw._set_free_joint_pose(
        int(raw._peg_qpos_adr),
        int(raw._peg_qvel_adr),
        position,
        rot.as_quat(scalar_first=True),
    )
    raw._data.qvel[int(raw._peg_qvel_adr) : int(raw._peg_qvel_adr) + 6] = 0.0
    mujoco.mj_forward(raw._model, raw._data)


def aligned_seat_pose(raw, *, along_m: float = -0.030) -> tuple[np.ndarray, R]:
    """Solve the peg pose whose tip is centered and seated along the hole axis."""
    from interaction_retarget.skill_replay.insert import _insert_geometry

    tip, socket, hole, _ = _insert_geometry(raw)
    model, data = raw._model, raw._data
    peg_id = int(model.body(names_from_raw(raw).peg_body).id)
    peg_pos = np.asarray(data.xpos[peg_id], dtype=np.float64).copy()
    peg_rot = R.from_quat(
        np.asarray(data.xquat[peg_id], dtype=np.float64), scalar_first=True
    )
    hole_u = np.asarray(hole, dtype=np.float64)
    hole_u /= np.linalg.norm(hole_u) + 1e-8
    peg_axis = peg_rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    if float(np.dot(peg_axis, hole_u)) < 0.0:
        hole_u = -hole_u

    cross = np.cross(peg_axis, hole_u)
    sin_angle = float(np.linalg.norm(cross))
    cos_angle = float(np.clip(np.dot(peg_axis, hole_u), -1.0, 1.0))
    correction = (
        R.identity()
        if sin_angle < 1e-10
        else R.from_rotvec(
            cross / sin_angle * float(np.arctan2(sin_angle, cos_angle))
        )
    )
    target_rot = correction * peg_rot
    tip_local = peg_rot.inv().apply(np.asarray(tip, dtype=np.float64) - peg_pos)
    target_tip = np.asarray(socket, dtype=np.float64) + hole_u * float(along_m)
    target_pos = target_tip - target_rot.apply(tip_local)
    return target_pos, target_rot


def adaptive_strength(tip_dist: float, lat_err: float, axis_err: float) -> float:
    """Keep transport rigid, become compliant only inside the contact funnel."""
    if tip_dist > 0.10:
        return 1.0
    if tip_dist > 0.055 or lat_err > 0.014 or axis_err > 0.22:
        return 0.85
    if tip_dist > 0.038 or lat_err > 0.008 or axis_err > 0.14:
        return 0.45
    return 0.12
