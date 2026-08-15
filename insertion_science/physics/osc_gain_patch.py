"""Runtime OSC gain override without editing dexjoco / reach / embodied sources."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np

Mode = Literal["iso", "task_aniso"]


@dataclass
class OscGains:
    mode: Mode
    pos_gains: tuple[float, float, float]
    ori_gains: tuple[float, float, float]
    damping_ratio: float
    k_axial: float | None = None
    k_lateral: float | None = None
    task_axis: np.ndarray | None = None  # unit world vector

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "pos_gains": list(self.pos_gains),
            "ori_gains": list(self.ori_gains),
            "damping_ratio": float(self.damping_ratio),
            "k_axial": self.k_axial,
            "k_lateral": self.k_lateral,
            "task_axis": None
            if self.task_axis is None
            else np.asarray(self.task_axis, dtype=np.float64).tolist(),
        }


_ACTIVE: OscGains | None = None
_PATCHED = False
_ORIG = None


def _task_basis(axis: np.ndarray) -> np.ndarray:
    """Return R with columns [e1, e2, axial] (orthonormal)."""
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        a = a / n
    # pick a helper not parallel to a
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, a))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = np.cross(a, helper)
    e1 /= float(np.linalg.norm(e1))
    e2 = np.cross(a, e1)
    e2 /= float(np.linalg.norm(e2))
    return np.stack([e1, e2, a], axis=1)


def _opspace_task_aniso(model, data, site_id, dof_ids, **kwargs):
    """opspace with position stiffness anisotropic in frozen task axis frame."""
    from dexjoco.sim.controllers.opspace import (
        pd_control,
        pd_control_orientation,
    )
    import mujoco
    from dm_robotics.transformations import transformations as tr

    assert _ACTIVE is not None and _ACTIVE.mode == "task_aniso"
    assert _ACTIVE.task_axis is not None
    assert _ACTIVE.k_axial is not None and _ACTIVE.k_lateral is not None

    pos = kwargs.get("pos")
    ori = kwargs.get("ori")
    joint = kwargs.get("joint")
    ori_gains = kwargs.get("ori_gains", _ACTIVE.ori_gains)
    damping_ratio = float(kwargs.get("damping_ratio", _ACTIVE.damping_ratio))
    nullspace_stiffness = float(kwargs.get("nullspace_stiffness", 0.5))
    max_pos_acceleration = kwargs.get("max_pos_acceleration")
    max_ori_acceleration = kwargs.get("max_ori_acceleration")
    gravity_comp = bool(kwargs.get("gravity_comp", True))

    if pos is None:
        x_des = data.site_xpos[site_id]
    else:
        x_des = np.asarray(pos)
    if ori is None:
        xmat = data.site_xmat[site_id].reshape((3, 3))
        quat_des = tr.mat_to_quat(xmat.reshape((3, 3)))
    else:
        ori = np.asarray(ori)
        if ori.shape == (3, 3):
            quat_des = tr.mat_to_quat(ori)
        else:
            quat_des = ori
    if joint is None:
        q_des = data.qpos[dof_ids]
    else:
        q_des = np.asarray(joint)

    R = _task_basis(_ACTIVE.task_axis)
    k_diag = np.array(
        [_ACTIVE.k_lateral, _ACTIVE.k_lateral, _ACTIVE.k_axial], dtype=np.float64
    )
    kd_diag = damping_ratio * 2.0 * np.sqrt(np.maximum(k_diag, 1e-8))

    kp_o = np.asarray(ori_gains, dtype=np.float64)
    kd_o = damping_ratio * 2.0 * np.sqrt(kp_o)
    kp_kv_ori = np.stack([kp_o, kd_o], axis=-1)

    kp_joint = np.full((len(dof_ids),), nullspace_stiffness)
    kd_joint = damping_ratio * 2.0 * np.sqrt(kp_joint)
    kp_kv_joint = np.stack([kp_joint, kd_joint], axis=-1)

    ddx_max = max_pos_acceleration if max_pos_acceleration is not None else 0.0
    dw_max = max_ori_acceleration if max_ori_acceleration is not None else 0.0

    q = data.qpos[dof_ids]
    dq = data.qvel[dof_ids]

    J_v = np.zeros((3, model.nv), dtype=np.float64)
    J_w = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, J_v, J_w, site_id)
    J_v = J_v[:, dof_ids]
    J_w = J_w[:, dof_ids]
    J = np.concatenate([J_v, J_w], axis=0)

    x = data.site_xpos[site_id]
    dx = J_v @ dq
    # task-frame anisotropic PD on position
    x_err = np.asarray(x - x_des, dtype=np.float64)
    dx_task = R.T @ np.asarray(dx, dtype=np.float64)
    x_err_task = R.T @ x_err
    ddx_task = -k_diag * x_err_task - kd_diag * dx_task
    if ddx_max > 0.0:
        n2 = float(np.sum(ddx_task**2))
        if n2 > ddx_max**2:
            ddx_task *= ddx_max / np.sqrt(n2)
    ddx = R @ ddx_task

    quat = tr.mat_to_quat(data.site_xmat[site_id].reshape((3, 3)))
    if quat @ quat_des < 0.0:
        quat *= -1.0
    w = J_w @ dq
    dw = pd_control_orientation(
        quat=quat,
        quat_des=quat_des,
        w=w,
        kp_kv=kp_kv_ori,
        dw_max=dw_max,
    )

    M = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, M, data.qM)
    M = M[dof_ids, :][:, dof_ids]
    M_inv = np.linalg.inv(M)
    Mx_inv = J @ M_inv @ J.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

    ddx_dw = np.concatenate([ddx, dw], axis=0)
    tau = J.T @ Mx @ ddx_dw
    ddq = pd_control(x=q, x_des=q_des, dx=dq, kp_kv=kp_kv_joint, ddx_max=0.0)
    Jnull = M_inv @ J.T @ Mx
    tau += (np.eye(len(q)) - J.T @ Jnull.T) @ ddq
    if gravity_comp:
        tau += data.qfrc_bias[dof_ids]
    return tau


def _wrapped_opspace(*args, **kwargs):
    global _ACTIVE, _ORIG
    if _ACTIVE is None:
        return _ORIG(*args, **kwargs)
    if _ACTIVE.mode == "task_aniso":
        return _opspace_task_aniso(*args, **kwargs)
    kwargs = dict(kwargs)
    kwargs["pos_gains"] = _ACTIVE.pos_gains
    kwargs["ori_gains"] = _ACTIVE.ori_gains
    kwargs["damping_ratio"] = _ACTIVE.damping_ratio
    return _ORIG(*args, **kwargs)


def ensure_patched() -> None:
    global _PATCHED, _ORIG
    if _PATCHED:
        return
    import dexjoco.sim.envs.panda_bimanual_assembly_env as assembly_mod

    _ORIG = assembly_mod.opspace
    assembly_mod.opspace = _wrapped_opspace
    _PATCHED = True


def set_task_axis(axis: np.ndarray | None) -> None:
    global _ACTIVE
    if _ACTIVE is None or _ACTIVE.mode != "task_aniso":
        return
    if axis is None:
        return
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    _ACTIVE.task_axis = a / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


@contextmanager
def osc_gains(
    *,
    pos_gains: tuple[float, float, float] | np.ndarray,
    ori_gains: tuple[float, float, float] | np.ndarray,
    damping_ratio: float,
    mode: Mode = "iso",
    k_axial: float | None = None,
    k_lateral: float | None = None,
    task_axis: np.ndarray | None = None,
) -> Iterator[OscGains]:
    global _ACTIVE
    ensure_patched()
    cfg = OscGains(
        mode=mode,
        pos_gains=tuple(float(x) for x in np.asarray(pos_gains).reshape(3)),
        ori_gains=tuple(float(x) for x in np.asarray(ori_gains).reshape(3)),
        damping_ratio=float(damping_ratio),
        k_axial=None if k_axial is None else float(k_axial),
        k_lateral=None if k_lateral is None else float(k_lateral),
        task_axis=None
        if task_axis is None
        else np.asarray(task_axis, dtype=np.float64).reshape(3).copy(),
    )
    prev = _ACTIVE
    _ACTIVE = cfg
    try:
        yield cfg
    finally:
        _ACTIVE = prev


def scale_gains(
    *,
    baseline_pos: tuple[float, float, float],
    baseline_ori: tuple[float, float, float],
    stiffness_scale: float,
    damping_ratio: float,
) -> OscGains:
    s = float(stiffness_scale)
    return OscGains(
        mode="iso",
        pos_gains=tuple(float(v) * s for v in baseline_pos),
        ori_gains=tuple(float(v) * s for v in baseline_ori),
        damping_ratio=float(damping_ratio),
    )
