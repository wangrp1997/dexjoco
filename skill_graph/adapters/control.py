"""MuJoCo bimanual assembly control (opspace mocap + Allegro)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from dexjoco.sim.controllers import opspace

from skill_graph.constants import Side

if TYPE_CHECKING:
    from skill_graph.adapters.assembly import AssemblySim


def read_arm23(sim: "AssemblySim", side: Side) -> np.ndarray:
    raw = sim.raw
    mid = int(raw._mocap_right_id if side == "right" else raw._mocap_left_id)
    pos = np.asarray(raw._data.mocap_pos[mid], dtype=np.float64)
    quat = np.asarray(raw._data.mocap_quat[mid], dtype=np.float64)
    ctrl_ids = np.asarray(raw._allegro_ctrl_ids, dtype=int)
    if side == "right":
        hand = np.asarray(raw._data.ctrl[ctrl_ids[:16]], dtype=np.float64)
    else:
        hand = np.asarray(raw._data.ctrl[ctrl_ids[16:32]], dtype=np.float64)
    return np.concatenate([pos, quat, hand], axis=0)


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0x = q0[[1, 2, 3, 0]]
    q1x = q1[[1, 2, 3, 0]]
    out = R.from_quat([q0x, q1x]).mean([1.0 - t, t]).as_quat()
    return np.asarray([out[3], out[0], out[1], out[2]], dtype=np.float64)


def interpolate_arm23(a: np.ndarray, b: np.ndarray, t: float, *, hand: np.ndarray | None = None) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(23)
    b = np.asarray(b, dtype=np.float64).reshape(23)
    pos = (1.0 - t) * a[:3] + t * b[:3]
    quat = _slerp_quat(a[3:7], b[3:7], t)
    h = hand if hand is not None else (1.0 - t) * a[7:23] + t * b[7:23]
    return np.concatenate([pos, quat, h], axis=0)


def interpolate_arm_only(a: np.ndarray, b: np.ndarray, t: float, *, hand: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(23)
    b = np.asarray(b, dtype=np.float64).reshape(23)
    pos = (1.0 - t) * a[:3] + t * b[:3]
    quat = _slerp_quat(a[3:7], b[3:7], t)
    return np.concatenate([pos, quat, np.asarray(hand, dtype=np.float64).reshape(16)], axis=0)


def settle_bimanual(sim: "AssemblySim", right23: np.ndarray, left23: np.ndarray, *, substeps: int = 40) -> None:
    raw = sim.raw
    model, data = raw._model, raw._data
    mr, ml = int(raw._mocap_right_id), int(raw._mocap_left_id)
    right23 = np.asarray(right23, dtype=np.float64).reshape(23)
    left23 = np.asarray(left23, dtype=np.float64).reshape(23)
    data.mocap_pos[mr] = right23[:3]
    data.mocap_quat[mr] = right23[3:7]
    data.mocap_pos[ml] = left23[:3]
    data.mocap_quat[ml] = left23[3:7]
    allegro = np.concatenate([right23[7:23], left23[7:23]])
    home = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4), dtype=np.float64)
    for _ in range(int(substeps)):
        data.ctrl[raw._panda_right_ctrl_ids] = opspace(
            model=model,
            data=data,
            site_id=int(raw._site_right_id),
            dof_ids=raw._panda_right_dof_ids,
            pos=data.mocap_pos[mr],
            ori=data.mocap_quat[mr],
            joint=home,
            gravity_comp=True,
            pos_gains=(400.0, 400.0, 400.0),
            damping_ratio=4,
        )
        data.ctrl[raw._panda_left_ctrl_ids] = opspace(
            model=model,
            data=data,
            site_id=int(raw._site_left_id),
            dof_ids=raw._panda_left_dof_ids,
            pos=data.mocap_pos[ml],
            ori=data.mocap_quat[ml],
            joint=home,
            gravity_comp=True,
            pos_gains=(400.0, 400.0, 400.0),
            damping_ratio=4,
        )
        data.ctrl[raw._allegro_ctrl_ids] = allegro
        mujoco.mj_step(model, data)
    sim.on_physics_step()


def step_side(
    sim: "AssemblySim",
    *,
    side: Side,
    active23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
) -> None:
    active23 = np.asarray(active23, dtype=np.float64).reshape(23)
    hold_right = np.asarray(hold_right, dtype=np.float64).reshape(23)
    hold_left = np.asarray(hold_left, dtype=np.float64).reshape(23)
    n = max(int(getattr(sim.raw, "_n_substeps", 1)), 1)
    if side == "left":
        settle_bimanual(sim, hold_right, active23, substeps=n)
    else:
        settle_bimanual(sim, active23, hold_left, substeps=n)
