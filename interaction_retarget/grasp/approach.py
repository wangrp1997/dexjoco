"""Approach trajectory: home → GenHand pre-grasp → grasp (refs/GenHand/simulation/trajectory.py).

MVP uses linear mocap / finger interpolation via env.step (dexjoco opspace).
Full pyroki trajopt (refs/pyroki/examples/07_trajopt.py) can replace this later.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from interaction_retarget.grasp.repair import _step_bimanual, _step_side
from interaction_retarget.sim.settle import vec_to_arm_action

Side = Literal["left", "right"]


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


def interpolate_arm_only(a23: np.ndarray, b23: np.ndarray, t: float, *, hand: np.ndarray) -> np.ndarray:
    """GenHand grasp phase: move mocap only, fingers stay open (trajectory.py)."""
    a23 = vec_to_arm_action(a23)
    b23 = vec_to_arm_action(b23)
    hand = np.asarray(hand, dtype=np.float64).reshape(16)
    pos = (1.0 - t) * a23[0:3] + t * b23[0:3]
    quat = _slerp_quat_wxyz(a23[3:7], b23[3:7], t)
    return np.concatenate([pos, quat, hand], axis=0)


def interpolate_fingers_only(a23: np.ndarray, b23: np.ndarray, t: float) -> np.ndarray:
    """GenHand close phase: arm fixed, fingers initial_joint → grasp_joint."""
    a23 = vec_to_arm_action(a23)
    b23 = vec_to_arm_action(b23)
    hand = (1.0 - t) * a23[7:23] + t * b23[7:23]
    return np.concatenate([b23[0:7], hand], axis=0)


def reach_side_via_env(
    raw_env,
    *,
    side: Side,
    target23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    n_steps: int = 80,
    start23: np.ndarray | None = None,
) -> None:
    """Ramp active arm home → target with env.step (matches demo reach physics)."""
    from interaction_retarget.sim.settle import read_arm_action

    target23 = vec_to_arm_action(target23)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    start = read_arm_action(raw_env, side) if start23 is None else vec_to_arm_action(start23)
    n = max(int(n_steps), 1)
    for i in range(n):
        t = (i + 1) / n
        cmd = interpolate_action23(start, target23, t)
        _step_side(
            raw_env,
            side=side,
            active23=cmd,
            hold_right=hold_right,
            hold_left=hold_left,
        )


def execute_side_approach(
    raw_env,
    *,
    side: Side,
    home: np.ndarray,
    pre_grasp: np.ndarray,
    grasp: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    pre_steps: int = 18,
    grasp_steps: int = 12,
) -> None:
    """Async reach: only ``side`` moves; the other arm holds at hold_right/hold_left."""
    from interaction_retarget.sim.settle import read_arm_action

    pre_steps = max(int(pre_steps), 1)
    grasp_steps = max(int(grasp_steps), 1)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)

    for i in range(pre_steps):
        t = (i + 1) / pre_steps
        active = interpolate_action23(home, pre_grasp, t)
        _step_side(
            raw_env,
            side=side,
            active23=active,
            hold_right=hold_right,
            hold_left=hold_left,
        )

    pre_actual = read_arm_action(raw_env, side)
    open_hand = pre_grasp[7:23]
    arm_steps = max(int(round(grasp_steps * 0.72)), 1)
    close_steps = max(int(grasp_steps) - arm_steps, 1)
    for i in range(arm_steps):
        t = (i + 1) / arm_steps
        active = interpolate_arm_only(pre_actual, grasp, t, hand=open_hand)
        _step_side(
            raw_env,
            side=side,
            active23=active,
            hold_right=hold_right,
            hold_left=hold_left,
        )
    for i in range(close_steps):
        t = (i + 1) / close_steps
        active = interpolate_fingers_only(
            np.concatenate([grasp[0:7], open_hand], axis=0),
            grasp,
            t,
        )
        _step_side(
            raw_env,
            side=side,
            active23=active,
            hold_right=hold_right,
            hold_left=hold_left,
        )


def execute_bimanual_approach(
    raw_env,
    *,
    home_right: np.ndarray,
    home_left: np.ndarray,
    pre_grasp_right: np.ndarray,
    pre_grasp_left: np.ndarray,
    grasp_right: np.ndarray,
    grasp_left: np.ndarray,
    pre_steps: int = 18,
    grasp_steps: int = 12,
) -> None:
    """Two-phase reach: home → pre-grasp (open hand), pre-grasp → grasp (close)."""
    pre_steps = max(int(pre_steps), 1)
    grasp_steps = max(int(grasp_steps), 1)

    for i in range(pre_steps):
        t = (i + 1) / pre_steps
        r = interpolate_action23(home_right, pre_grasp_right, t)
        l = interpolate_action23(home_left, pre_grasp_left, t)
        _step_bimanual(raw_env, r, l)

    for i in range(grasp_steps):
        t = (i + 1) / grasp_steps
        r = interpolate_action23(pre_grasp_right, grasp_right, t)
        l = interpolate_action23(pre_grasp_left, grasp_left, t)
        _step_bimanual(raw_env, r, l)
