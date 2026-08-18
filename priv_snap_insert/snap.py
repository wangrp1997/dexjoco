"""Hard o2h snap for privileged PBVS."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from embodied_grasp_insertion.physics.grasp_metrics import (
    REFERENCE_BODY,
    ObjectInHandPose,
    object_in_hand_pose,
)
from dexjoco.sim.envs.assembly_geometry import names_from_raw
from interaction_retarget.skill_replay.insert import _insert_geometry


@dataclass
class O2HLock:
    o2h: ObjectInHandPose
    tip_in_peg: np.ndarray


def capture_o2h(raw, *, reference_body: str = REFERENCE_BODY) -> ObjectInHandPose:
    return object_in_hand_pose(raw, reference_body=reference_body)


def capture_o2h_lock(raw, *, reference_body: str = REFERENCE_BODY) -> O2HLock:
    o2h = object_in_hand_pose(raw, reference_body=reference_body)
    model, data = raw._model, raw._data
    peg_id = int(model.body(names_from_raw(raw).peg_body).id)
    tip_w, _, _, _ = _insert_geometry(raw)
    peg_pos = np.asarray(data.xpos[peg_id], dtype=np.float64)
    peg_rot = R.from_quat(np.asarray(data.xquat[peg_id], dtype=np.float64), scalar_first=True)
    tip_in_peg = peg_rot.inv().apply(np.asarray(tip_w, dtype=np.float64) - peg_pos)
    return O2HLock(o2h=o2h, tip_in_peg=np.asarray(tip_in_peg, dtype=np.float64))


def locked_peg_pose(raw, o2h: ObjectInHandPose) -> tuple[np.ndarray, R]:
    model, data = raw._model, raw._data
    bid = int(model.body(o2h.reference_body).id)
    palm_pos = np.asarray(data.xpos[bid], dtype=np.float64)
    palm_rot = R.from_quat(np.asarray(data.xquat[bid], dtype=np.float64), scalar_first=True)
    peg_pos = palm_pos + palm_rot.apply(np.asarray(o2h.translation, dtype=np.float64))
    peg_rot = palm_rot * R.from_rotvec(np.asarray(o2h.rotvec, dtype=np.float64))
    return peg_pos, peg_rot


def virtual_insert_features(raw, lock: O2HLock, *, target_along_m: float = -0.040) -> dict:
    from hybrid_insert.geometry import (
        axis_parallel_error_rad,
        height_along_axis,
        lateral_error,
    )

    peg_pos, peg_rot = locked_peg_pose(raw, lock.o2h)
    tip = peg_pos + peg_rot.apply(np.asarray(lock.tip_in_peg, dtype=np.float64))
    peg_ax = peg_rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    _, socket, hole, _ = _insert_geometry(raw)
    socket = np.asarray(socket, dtype=np.float64)
    hole = np.asarray(hole, dtype=np.float64)
    lat_err, lat_vec = lateral_error(tip, socket, hole)
    along = height_along_axis(tip, socket, hole)
    return {
        "tip": tip,
        "socket": socket,
        "hole": hole,
        "peg_axis": peg_ax,
        "lat_err": float(lat_err),
        "lat_vec": np.asarray(lat_vec, dtype=np.float64),
        "along": float(along),
        "tip_dist": float(np.linalg.norm(tip - socket)),
        "axis_err": float(axis_parallel_error_rad(peg_ax, hole)),
        "e_along": float(along - target_along_m),
    }


def project_peg_to_o2h(
    raw,
    o2h: ObjectInHandPose,
    *,
    strength: float = 1.0,
    pos_strength: float | None = None,
    rot_strength: float | None = None,
    reference_body: str | None = None,
    free_along: bool = False,
) -> None:
    a_pos = float(np.clip(pos_strength if pos_strength is not None else strength, 0.0, 1.0))
    a_rot = float(np.clip(rot_strength if rot_strength is not None else strength, 0.0, 1.0))
    if a_pos <= 1e-8 and a_rot <= 1e-8:
        return
    ref = reference_body or o2h.reference_body
    model, data = raw._model, raw._data
    bid = int(model.body(ref).id)
    palm_pos = np.asarray(data.xpos[bid], dtype=np.float64)
    palm_rot = R.from_quat(np.asarray(data.xquat[bid], dtype=np.float64), scalar_first=True)
    target_pos = palm_pos + palm_rot.apply(np.asarray(o2h.translation, dtype=np.float64))
    target_rot = palm_rot * R.from_rotvec(np.asarray(o2h.rotvec, dtype=np.float64))
    peg_id = int(model.body(names_from_raw(raw).peg_body).id)
    cur_pos = np.asarray(data.xpos[peg_id], dtype=np.float64)
    cur_rot = R.from_quat(np.asarray(data.xquat[peg_id], dtype=np.float64), scalar_first=True)
    pos = target_pos if a_pos >= 1.0 - 1e-8 else cur_pos + a_pos * (target_pos - cur_pos)
    if free_along:
        peg_z = target_rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        pos = target_pos + peg_z * float(np.dot(cur_pos - target_pos, peg_z))
    if a_rot >= 1.0 - 1e-8:
        quat = target_rot.as_quat(scalar_first=True)
    else:
        delta = cur_rot.inv() * target_rot
        quat = (cur_rot * R.from_rotvec(delta.as_rotvec() * a_rot)).as_quat(scalar_first=True)
    raw._set_free_joint_pose(int(raw._peg_qpos_adr), int(raw._peg_qvel_adr), pos, quat)
    data.qvel[int(raw._peg_qvel_adr) : int(raw._peg_qvel_adr) + 6] = 0.0
    mujoco.mj_forward(model, data)


def snap_peg_to_o2h(raw, o2h: ObjectInHandPose, *, reference_body: str | None = None) -> None:
    project_peg_to_o2h(raw, o2h, strength=1.0, reference_body=reference_body)


def _freejoint_pose(raw, qpos_adr: int) -> tuple[np.ndarray, np.ndarray]:
    q = raw._data.qpos
    adr = int(qpos_adr)
    return (
        np.asarray(q[adr : adr + 3], dtype=np.float64).copy(),
        np.asarray(q[adr + 3 : adr + 7], dtype=np.float64).copy(),
    )


def pin_freejoint(
    raw,
    qpos_adr: int,
    qvel_adr: int,
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
) -> None:
    raw._set_free_joint_pose(
        int(qpos_adr),
        int(qvel_adr),
        np.asarray(pos, dtype=np.float64),
        np.asarray(quat_wxyz, dtype=np.float64),
    )
    raw._data.qvel[int(qvel_adr) : int(qvel_adr) + 6] = 0.0
    mujoco.mj_forward(raw._model, raw._data)


def capture_peg_socket_pins(raw) -> dict[str, np.ndarray]:
    """World poses to hold after release (peg stays in hole, tray/socket stays)."""
    peg_pos, peg_quat = _freejoint_pose(raw, int(raw._peg_qpos_adr))
    sock_pos, sock_quat = _freejoint_pose(raw, int(raw._socket_qpos_adr))
    return {
        "peg_pos": peg_pos,
        "peg_quat": peg_quat,
        "socket_pos": sock_pos,
        "socket_quat": sock_quat,
    }


def capture_socket_pin(raw) -> dict[str, np.ndarray]:
    sock_pos, sock_quat = _freejoint_pose(raw, int(raw._socket_qpos_adr))
    return {"socket_pos": sock_pos, "socket_quat": sock_quat}


def apply_socket_pin(raw, pins: dict[str, np.ndarray]) -> None:
    pin_freejoint(
        raw,
        int(raw._socket_qpos_adr),
        int(raw._socket_qvel_adr),
        pins["socket_pos"],
        pins["socket_quat"],
    )


def apply_peg_socket_pins(raw, pins: dict[str, np.ndarray]) -> None:
    pin_freejoint(
        raw,
        int(raw._peg_qpos_adr),
        int(raw._peg_qvel_adr),
        pins["peg_pos"],
        pins["peg_quat"],
    )
    apply_socket_pin(raw, pins)


LEFT_PALM = "allegro_palm_left"


def capture_tray_in_left(raw, *, reference_body: str = LEFT_PALM) -> ObjectInHandPose:
    """Tray/socket pose in left-palm frame (handoff lock)."""
    model, data = raw._model, raw._data
    tray_id = int(model.body(names_from_raw(raw).socket_body).id)
    ref_id = int(model.body(reference_body).id)
    tray_pos = np.asarray(data.xpos[tray_id], dtype=np.float64)
    ref_pos = np.asarray(data.xpos[ref_id], dtype=np.float64)
    tray_rot = R.from_quat(np.asarray(data.xquat[tray_id], dtype=np.float64), scalar_first=True)
    ref_rot = R.from_quat(np.asarray(data.xquat[ref_id], dtype=np.float64), scalar_first=True)
    rel_rot = ref_rot.inv() * tray_rot
    rel_t = ref_rot.inv().apply(tray_pos - ref_pos)
    return ObjectInHandPose(
        reference_body=reference_body,
        translation=np.asarray(rel_t, dtype=np.float64),
        rotvec=np.asarray(rel_rot.as_rotvec(), dtype=np.float64),
    )


def project_tray_to_left(
    raw,
    t2h: ObjectInHandPose,
    *,
    strength: float = 1.0,
) -> None:
    """Keep tray glued to left palm (privileged), same idea as peg o2h."""
    a = float(np.clip(strength, 0.0, 1.0))
    if a <= 1e-8:
        return
    model, data = raw._model, raw._data
    ref = t2h.reference_body
    bid = int(model.body(ref).id)
    palm_pos = np.asarray(data.xpos[bid], dtype=np.float64)
    palm_rot = R.from_quat(np.asarray(data.xquat[bid], dtype=np.float64), scalar_first=True)
    target_pos = palm_pos + palm_rot.apply(np.asarray(t2h.translation, dtype=np.float64))
    target_rot = palm_rot * R.from_rotvec(np.asarray(t2h.rotvec, dtype=np.float64))
    cur_pos, cur_quat = _freejoint_pose(raw, int(raw._socket_qpos_adr))
    cur_rot = R.from_quat(cur_quat, scalar_first=True)
    pos = target_pos if a >= 1.0 - 1e-8 else cur_pos + a * (target_pos - cur_pos)
    if a >= 1.0 - 1e-8:
        quat = target_rot.as_quat(scalar_first=True)
    else:
        delta = cur_rot.inv() * target_rot
        quat = (cur_rot * R.from_rotvec(delta.as_rotvec() * a)).as_quat(scalar_first=True)
    pin_freejoint(
        raw,
        int(raw._socket_qpos_adr),
        int(raw._socket_qvel_adr),
        pos,
        quat,
    )


def pin_peg_tip_seated(
    raw,
    tip_in_peg: np.ndarray | None = None,
    *,
    along_m: float = -0.025,
) -> dict[str, np.ndarray]:
    """Translate peg so tip sits in hole; keep current orientation. Breaks o2h glue."""
    from interaction_retarget.skill_replay.insert import _insert_geometry

    tip_w, socket, hole, _ = _insert_geometry(raw)
    hole_u = np.asarray(hole, dtype=np.float64)
    hole_u = hole_u / (np.linalg.norm(hole_u) + 1e-8)
    tip_tgt = np.asarray(socket, dtype=np.float64) + hole_u * float(along_m)
    peg_id = int(raw._model.body(names_from_raw(raw).peg_body).id)
    peg_pos = np.asarray(raw._data.xpos[peg_id], dtype=np.float64)
    peg_quat = np.asarray(raw._data.xquat[peg_id], dtype=np.float64).copy()
    _ = tip_in_peg
    new_pos = peg_pos + (tip_tgt - np.asarray(tip_w, dtype=np.float64))
    pin_freejoint(
        raw,
        int(raw._peg_qpos_adr),
        int(raw._peg_qvel_adr),
        new_pos,
        peg_quat,
    )
    return {
        "peg_pos": np.asarray(new_pos, dtype=np.float64),
        "peg_quat": peg_quat,
    }
