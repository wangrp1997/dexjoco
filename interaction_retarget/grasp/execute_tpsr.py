"""Execute grasp with TPSR refine (DexGraspBench staged close + Dexonomy QP refine)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np

from interaction_retarget.grasp.metrics import measure_reach
from interaction_retarget.grasp.pre_grasp import derive_pre_grasp_from_grasp
from interaction_retarget.grasp.approach import execute_side_approach, reach_arm_then_close
from interaction_retarget.grasp.staged_grasp import derive_squeeze23, execute_grasp_to_squeeze
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.settle import read_arm_action, vec_to_arm_action
from interaction_retarget.tpsr.config import TpsrConfig
from interaction_retarget.tpsr.grasp_filter import GraspFilter, grasp_filter_cfg_from_tpsr
from interaction_retarget.tpsr.refine import refine_side_grasp

Side = Literal["left", "right"]
ObjectName = Literal["tray", "peg"]


def execute_side_grasp_tpsr(
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
    tpsr_cfg: TpsrConfig | None = None,
    skip_approach: bool = False,
    skip_repair: bool = False,
    finger_repair_only: bool = False,
    direct_reach_steps: int | None = None,
    repair_hold_steps: int = 5,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    """DexGraspBench pre→grasp→squeeze, then Dexonomy GraspFilter + sim_refine."""
    grasp23 = vec_to_arm_action(grasp23)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    cfg = tpsr_cfg or TpsrConfig()
    if finger_repair_only:
        cfg = replace(cfg, finger_only=True)

    home23 = read_arm_action(raw_env, side)
    pre_n = max(int(pre_steps), 1)
    grasp_n = max(int(grasp_steps), 1)
    squeeze_n = max(int(cfg.squeeze_steps), 1)
    total_steps = 0

    def _squeeze_at_grasp(grasp_pose: np.ndarray) -> np.ndarray:
        return derive_squeeze23(
            grasp_pose,
            canonical,
            raw_env=raw_env,
            side=side,
            object_name=object_name,
        )

    if direct_reach_steps is not None:
        reach_n = max(int(direct_reach_steps), 1)
        close_n = max(int(repair_hold_steps) * 3, 16)
        reach_arm_then_close(
            raw_env,
            side=side,
            target23=grasp23,
            hold_right=hold_right,
            hold_left=hold_left,
            reach_steps=reach_n,
            close_steps=close_n,
        )
        squeeze23 = _squeeze_at_grasp(grasp23)
        execute_grasp_to_squeeze(
            raw_env,
            side=side,
            grasp23=grasp23,
            squeeze23=squeeze23,
            hold_right=hold_right,
            hold_left=hold_left,
            squeeze_steps=squeeze_n,
        )
        total_steps = reach_n + close_n + squeeze_n
    elif not skip_approach:
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
        squeeze23 = _squeeze_at_grasp(grasp23)
        execute_grasp_to_squeeze(
            raw_env,
            side=side,
            grasp23=grasp23,
            squeeze23=squeeze23,
            hold_right=hold_right,
            hold_left=hold_left,
            squeeze_steps=squeeze_n,
        )
        total_steps = pre_n + grasp_n + squeeze_n
    else:
        # IK-ready: already at grasp; sim_refine does DexGraspBench squeeze + QP.
        total_steps = 0

    achieved23 = read_arm_action(raw_env, side)
    if side == "left":
        action_right, action_left = hold_right, achieved23
    else:
        action_right, action_left = achieved23, hold_left

    gf_cfg = grasp_filter_cfg_from_tpsr(cfg)
    gf = GraspFilter(gf_cfg).forward(raw_env, side=side, object_name=object_name)
    qp_ready = gf.ok or not cfg.require_qp_fc

    repair_iters = 0
    tpsr_info: dict[str, float] = {"qp_max_error": gf.max_qp_error}
    if skip_repair and qp_ready:
        right_hold, left_hold = action_right, action_left
    else:
        right_hold, left_hold, repair_iters, metrics = refine_side_grasp(
            raw_env,
            side=side,
            object_name=object_name,
            action_right=action_right,
            action_left=action_left,
            hold_right=hold_right,
            hold_left=hold_left,
            detector=detector,
            canonical=canonical,
            cfg=cfg,
            skip_greedy=False,
        )
        tpsr_info = {
            "laplacian_rmse_m": metrics.laplacian_rmse_m,
            "hand_rmse_m": metrics.hand_rmse_m,
            "hole_violation_m": metrics.hole_violation_m,
            "contact_count": float(metrics.contact_count),
            "qp_max_error": GraspFilter(gf_cfg)
            .forward(raw_env, side=side, object_name=object_name)
            .max_qp_error,
        }

    if side == "left":
        right_hold = vec_to_arm_action(hold_right)
    else:
        left_hold = vec_to_arm_action(hold_left)

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
        "pre_steps": pre_n if not skip_approach and direct_reach_steps is None else 0,
        "grasp_steps": grasp_n if not skip_approach and direct_reach_steps is None else 0,
        "squeeze_steps": squeeze_n,
        "direct_reach_steps": int(direct_reach_steps) if direct_reach_steps is not None else 0,
        "skip_approach": skip_approach,
        "lap_converged": m.laplacian_converged,
        "hand_converged": m.hand_converged,
        **tpsr_info,
    }
    return right_hold, left_hold, repair_iters, info
