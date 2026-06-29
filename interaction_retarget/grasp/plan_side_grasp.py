"""单臂抓取规划：拓扑 δ* IK → staged grasp → TPSR → ContactOpt → FC。"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from interaction_retarget.grasp.agent_tpsr import TpsrGraspArmAgent
from interaction_retarget.grasp.ik import GraspIkResult
from interaction_retarget.grasp.locked_hold import enforce_locked_passive
from interaction_retarget.grasp.pipeline import _ik_grasp_ready
from interaction_retarget.grasp.topo_fc_finalize import ensure_side_topo_fc
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.settle import read_arm_action, vec_to_arm_action
from interaction_retarget.skill_replay.grasp_finalize import GraspFinalizeReport
from interaction_retarget.tpsr.config import TpsrConfig


def playback_tpsr_cfg(cfg: TpsrConfig) -> TpsrConfig:
    """录像时关 sim mocap 搜索，保留 TPSR refine + ContactOpt。"""
    return replace(cfg, sim_search_iters=0)


def ik_grasp_target(ik: GraspIkResult, side: Literal["left", "right"], fallback: np.ndarray) -> np.ndarray:
    if _ik_grasp_ready(ik):
        return vec_to_arm_action(ik.action_left if side == "left" else ik.action_right)
    if ik.contact_count > 0:
        return vec_to_arm_action(ik.action_left if side == "left" else ik.action_right)
    return vec_to_arm_action(fallback)


def plan_side_grasp(
    agent: TpsrGraspArmAgent,
    raw_env,
    *,
    target23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    detector: AssemblyContactDetector,
    ik: GraspIkResult,
    tpsr_cfg: TpsrConfig,
    contact_refine_iters: int,
    direct_reach_steps: int | None = None,
    live_record: bool = False,
) -> tuple[np.ndarray, np.ndarray, GraspFinalizeReport, int, bool, str]:
    cfg = playback_tpsr_cfg(tpsr_cfg) if live_record else tpsr_cfg
    side: Literal["left", "right"] = agent.side
    grasp23 = ik_grasp_target(ik, side, target23)
    agent.tpsr_cfg = cfg

    right_hold, left_hold, repair_iters, _ = agent.execute(
        raw_env,
        target23=grasp23,
        hold_right=hold_right,
        hold_left=hold_left,
        detector=detector,
        ik=ik,
        skip_approach=False,
        skip_repair=False,
        direct_reach_steps=direct_reach_steps,
        side_cfg=agent.side_cfg,
    )
    right_hold, left_hold, fin, topo_ok, topo_msg = ensure_side_topo_fc(
        raw_env,
        side=side,
        object_name=agent.object_name,
        canonical=agent.canonical,
        hold_right=right_hold,
        hold_left=hold_left,
        detector=detector,
        tpsr_cfg=cfg,
        contact_refine_iters=contact_refine_iters,
    )
    passive_r = vec_to_arm_action(right_hold)
    passive_l = vec_to_arm_action(left_hold)
    if side == "left":
        enforce_locked_passive(
            raw_env,
            locked_left=vec_to_arm_action(read_arm_action(raw_env, "left")),
            locked_right=passive_r,
            n_substeps=4,
        )
    else:
        enforce_locked_passive(
            raw_env,
            locked_left=passive_l,
            locked_right=vec_to_arm_action(read_arm_action(raw_env, "right")),
            n_substeps=4,
        )
    return (
        right_hold,
        left_hold,
        fin,
        int(repair_iters) + int(fin.tpsr_iters),
        bool(topo_ok),
        str(topo_msg),
    )
