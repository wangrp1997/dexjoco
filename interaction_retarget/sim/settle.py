"""Settle Panda mocap + Allegro targets in MuJoCo (opspace, no env sleep)."""

from __future__ import annotations

from typing import Literal

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from dexjoco.sim.controllers import opspace

Side = Literal["left", "right"]


def _arm_arrays(raw_env) -> dict[str, np.ndarray | int]:
    return {
        "panda_right_dof": raw_env._panda_right_dof_ids,
        "panda_left_dof": raw_env._panda_left_dof_ids,
        "panda_right_ctrl": raw_env._panda_right_ctrl_ids,
        "panda_left_ctrl": raw_env._panda_left_ctrl_ids,
        "site_right": raw_env._site_right_id,
        "site_left": raw_env._site_left_id,
        "mocap_right": raw_env._mocap_right_id,
        "mocap_left": raw_env._mocap_left_id,
        "allegro_ctrl": raw_env._allegro_ctrl_ids,
        "allegro_dof_right": raw_env._allegro_dof_right_ids,
        "allegro_dof_left": raw_env._allegro_dof_left_ids,
        "panda_home_right": np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4), dtype=np.float64),
        "panda_home_left": np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4), dtype=np.float64),
    }


def arm_action_to_vec(action23: np.ndarray) -> np.ndarray:
    action23 = np.asarray(action23, dtype=np.float64).reshape(23)
    return action23.copy()


def vec_to_arm_action(vec23: np.ndarray) -> np.ndarray:
    return np.asarray(vec23, dtype=np.float64).reshape(23).copy()


def read_arm_action(raw_env, side: Side) -> np.ndarray:
    arrays = _arm_arrays(raw_env)
    mocap_id = int(arrays["mocap_right" if side == "right" else "mocap_left"])
    hand_ids = arrays["allegro_dof_right" if side == "right" else "allegro_dof_left"]
    data = raw_env._data
    pos = np.asarray(data.mocap_pos[mocap_id], dtype=np.float64)
    quat = np.asarray(data.mocap_quat[mocap_id], dtype=np.float64)
    hand = np.asarray(data.qpos[hand_ids], dtype=np.float64)
    return np.concatenate([pos, quat, hand], axis=0)


def _slerp_quat_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0_xyzw = q0[[1, 2, 3, 0]]
    q1_xyzw = q1[[1, 2, 3, 0]]
    slerp = Slerp([0.0, 1.0], R.from_quat([q0_xyzw, q1_xyzw]))
    out_xyzw = slerp(float(np.clip(t, 0.0, 1.0))).as_quat()
    return np.asarray([out_xyzw[3], out_xyzw[0], out_xyzw[1], out_xyzw[2]], dtype=np.float64)


def interpolate_action23(a23: np.ndarray, b23: np.ndarray, t: float) -> np.ndarray:
    a23 = vec_to_arm_action(a23)
    b23 = vec_to_arm_action(b23)
    pos = (1.0 - t) * a23[0:3] + t * b23[0:3]
    quat = _slerp_quat_wxyz(a23[3:7], b23[3:7], t)
    hand = (1.0 - t) * a23[7:23] + t * b23[7:23]
    return np.concatenate([pos, quat, hand], axis=0)


def settle_bimanual_actions(
    raw_env,
    *,
    right23: np.ndarray | None = None,
    left23: np.ndarray | None = None,
    n_substeps: int = 40,
) -> None:
    """Apply 23-d arm actions (pos+quat+hand) and run opspace substeps."""
    model = raw_env._model
    data = raw_env._data
    arrays = _arm_arrays(raw_env)

    if right23 is None:
        right23 = read_arm_action(raw_env, "right")
    if left23 is None:
        left23 = read_arm_action(raw_env, "left")

    right23 = vec_to_arm_action(right23)
    left23 = vec_to_arm_action(left23)

    data.mocap_pos[int(arrays["mocap_right"])] = right23[0:3]
    data.mocap_quat[int(arrays["mocap_right"])] = right23[3:7]
    data.mocap_pos[int(arrays["mocap_left"])] = left23[0:3]
    data.mocap_quat[int(arrays["mocap_left"])] = left23[3:7]

    allegro = np.concatenate([right23[7:23], left23[7:23]], axis=0)

    for _ in range(int(n_substeps)):
        tau_right = opspace(
            model=model,
            data=data,
            site_id=int(arrays["site_right"]),
            dof_ids=arrays["panda_right_dof"],
            pos=data.mocap_pos[int(arrays["mocap_right"])],
            ori=data.mocap_quat[int(arrays["mocap_right"])],
            joint=arrays["panda_home_right"],
            gravity_comp=True,
            pos_gains=(400.0, 400.0, 400.0),
            damping_ratio=4,
        )
        data.ctrl[arrays["panda_right_ctrl"]] = tau_right

        tau_left = opspace(
            model=model,
            data=data,
            site_id=int(arrays["site_left"]),
            dof_ids=arrays["panda_left_dof"],
            pos=data.mocap_pos[int(arrays["mocap_left"])],
            ori=data.mocap_quat[int(arrays["mocap_left"])],
            joint=arrays["panda_home_left"],
            gravity_comp=True,
            pos_gains=(400.0, 400.0, 400.0),
            damping_ratio=4,
        )
        data.ctrl[arrays["panda_left_ctrl"]] = tau_left
        data.ctrl[arrays["allegro_ctrl"]] = allegro
        mujoco.mj_step(model, data)


def settle_side_actions(
    raw_env,
    *,
    side: Side,
    active23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    n_substeps: int = 40,
    ramp_from_hold: bool = True,
) -> None:
    """Settle one arm toward ``active23``; the other arm holds fixed commands (async agent)."""
    active23 = vec_to_arm_action(active23)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    n = max(int(n_substeps), 1)
    start = read_arm_action(raw_env, side)
    for i in range(n):
        t = (i + 1) / n if ramp_from_hold else 1.0
        cmd = interpolate_action23(start, active23, t) if ramp_from_hold else active23
        if side == "left":
            settle_bimanual_actions(
                raw_env,
                right23=hold_right,
                left23=cmd,
                n_substeps=1,
            )
        else:
            settle_bimanual_actions(
                raw_env,
                right23=cmd,
                left23=hold_left,
                n_substeps=1,
            )
