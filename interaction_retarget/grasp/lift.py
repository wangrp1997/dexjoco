"""Tray lift between left grasp and right peg grasp (demo: tray_lift_start after left_grasp)."""

from __future__ import annotations

import numpy as np

from interaction_retarget.grasp.approach import interpolate_action23
from interaction_retarget.grasp.repair import _step_side
from interaction_retarget.sim.settle import vec_to_arm_action

# Demo median tray z rise grasp→lift (~20 ep); use slightly larger for clearance.
DEFAULT_TRAY_LIFT_M = 0.05


def execute_tray_lift(
    raw_env,
    *,
    grasp_left: np.ndarray,
    hold_right: np.ndarray,
    lift_height_m: float = DEFAULT_TRAY_LIFT_M,
    steps: int = 20,
) -> np.ndarray:
    """Lift left arm + tray in world +Z; right arm frozen (async)."""
    grasp_left = vec_to_arm_action(grasp_left)
    hold_right = vec_to_arm_action(hold_right)
    lift_left = grasp_left.copy()
    lift_left[2] += float(lift_height_m)
    steps = max(int(steps), 1)

    for i in range(steps):
        t = (i + 1) / steps
        active = interpolate_action23(grasp_left, lift_left, t)
        _step_side(
            raw_env,
            side="left",
            active23=active,
            hold_right=hold_right,
            hold_left=grasp_left,
        )
    return lift_left.copy()
