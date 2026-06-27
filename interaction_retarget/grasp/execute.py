"""Execute grasp: GenHand pre→grasp (approach.py) + spider repair (repair.py)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from interaction_retarget.grasp.approach import execute_side_approach
from interaction_retarget.grasp.metrics import measure_reach
from interaction_retarget.grasp.pre_grasp import derive_pre_grasp_from_grasp
from interaction_retarget.grasp.repair import repair_side_grasp
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.settle import read_arm_action, vec_to_arm_action

Side = Literal["left", "right"]
ObjectName = Literal["tray", "peg"]


def execute_side_grasp(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    canonical: dict,
    grasp23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    detector: AssemblyContactDetector,
    pre_steps: int,
    grasp_steps: int,
    max_repair_iters: int,
    skip_approach: bool = False,
    repair_hold_steps: int = 5,
    skip_repair: bool = False,
    finger_repair_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    """GenHand: home→pre→grasp→close; spider: repair until contact.

    When ``skip_approach=True`` the sim already holds the Laplacian IK pose
    (plan with ``restore_env=False``); only contact repair runs.
    """
    grasp23 = vec_to_arm_action(grasp23)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    home23 = read_arm_action(raw_env, side)

    pre_n = max(int(pre_steps), 1)
    grasp_n = max(int(grasp_steps), 1)
    total_steps = 0
    if not skip_approach:
        pre23 = derive_pre_grasp_from_grasp(grasp23, side=side, offset_scale=0.55).action23
        execute_side_approach(
            raw_env,
            side=side,
            home=home23,
            pre_grasp=pre23,
            grasp=grasp23,
            hold_right=hold_right,
            hold_left=hold_left,
            pre_steps=pre_n,
            grasp_steps=grasp_n,
        )
        total_steps = pre_n + grasp_n

    achieved23 = read_arm_action(raw_env, side)
    if side == "left":
        action_right, action_left = hold_right, achieved23
    else:
        action_right, action_left = achieved23, hold_left

    repair_iters = 0
    if skip_repair:
        right_hold, left_hold = action_right, action_left
    else:
        right_hold, left_hold, repair_iters = repair_side_grasp(
            raw_env,
            side=side,
            object_name=object_name,
            action_right=action_right,
            action_left=action_left,
            hold_right=hold_right,
            hold_left=hold_left,
            detector=detector,
            max_iters=max_repair_iters,
            canonical=canonical,
            require_on_table=False,
            hold_steps=repair_hold_steps,
            finger_only=finger_repair_only,
        )

    m = measure_reach(
        raw_env,
        side=side,
        object_name=object_name,
        canonical=canonical,
        target23=grasp23,
        home23=home23,
        detector=detector,
        total_steps=total_steps,
        site_err_gate_m=1.0,
        hand_rmse_gate_m=0.085,
        laplacian_gate_m=0.035,
    )

    info = {
        "site_err_m": m.site_err_m,
        "hand_rmse_m": m.hand_rmse_m,
        "laplacian_rmse_m": m.laplacian_rmse_m,
        "contact_count": m.contact_count,
        "reach_steps": total_steps,
        "pre_steps": pre_n if not skip_approach else 0,
        "grasp_steps": grasp_n if not skip_approach else 0,
        "skip_approach": skip_approach,
        "lap_converged": m.laplacian_converged,
        "hand_converged": m.hand_converged,
    }
    return right_hold, left_hold, repair_iters, info
