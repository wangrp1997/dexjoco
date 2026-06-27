"""Single-arm agent: holosoma/pyroki IK plan + GenHand execute + spider repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from interaction_retarget.grasp.execute import execute_side_grasp
from interaction_retarget.grasp.ik import GraspIkResult, solve_grasp_ik
from interaction_retarget.grasp.ik_config import GraspSideConfig, side_config
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.settle import vec_to_arm_action

Side = Literal["left", "right"]
ObjectName = Literal["tray", "peg"]


@dataclass
class GraspArmAgent:
    side: Side
    object_name: ObjectName
    canonical: dict
    side_cfg: GraspSideConfig | None = None

    def _cfg(self) -> GraspSideConfig:
        return self.side_cfg if self.side_cfg is not None else side_config(self.object_name)

    def plan(
        self,
        raw_env,
        *,
        hold_right: np.ndarray,
        hold_left: np.ndarray,
        side_cfg: GraspSideConfig | None = None,
        restore_env: bool = True,
        detector: AssemblyContactDetector | None = None,
    ) -> tuple[np.ndarray, GraspIkResult]:
        cfg = side_cfg or self._cfg()
        hold_right = vec_to_arm_action(hold_right)
        hold_left = vec_to_arm_action(hold_left)
        ik = solve_grasp_ik(
            raw_env,
            self.canonical,
            object_name=self.object_name,
            hold_right=hold_right,
            hold_left=hold_left,
            weights=cfg.weights,
            reach_steps=cfg.reach_steps,
            settle_steps_opt=cfg.settle_steps_opt,
            maxiter=cfg.maxiter,
            n_outer_iters=cfg.n_outer_iters,
            maxfun=cfg.maxfun,
            optimize=cfg.optimize,
            pos_bounds_m=cfg.pos_bounds_m,
            success_hand_rmse_m=cfg.success_hand_rmse_m,
            success_laplacian_rmse_m=cfg.success_laplacian_rmse_m,
            success_min_contact=cfg.success_min_contact,
            restore_env=restore_env,
            rollout_opt=cfg.rollout_opt,
            physics_hold_steps=cfg.physics_hold_steps,
            detector=detector,
        )
        target = vec_to_arm_action(ik.action_left if self.side == "left" else ik.action_right)
        return target, ik

    def execute(
        self,
        raw_env,
        *,
        target23: np.ndarray,
        hold_right: np.ndarray,
        hold_left: np.ndarray,
        detector: AssemblyContactDetector,
        ik: GraspIkResult | None = None,
        side_cfg: GraspSideConfig | None = None,
        skip_approach: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, int, dict]:
        cfg = side_cfg or self._cfg()
        return execute_side_grasp(
            raw_env,
            side=self.side,
            object_name=self.object_name,
            canonical=self.canonical,
            grasp23=target23,
            hold_right=hold_right,
            hold_left=hold_left,
            detector=detector,
            pre_steps=cfg.approach_pre_steps,
            grasp_steps=cfg.approach_grasp_steps,
            max_repair_iters=cfg.max_repair_iters,
            skip_approach=skip_approach,
            repair_hold_steps=cfg.repair_hold_steps,
            skip_repair=skip_approach,
        )


def make_tray_agent(canonical: dict) -> GraspArmAgent:
    from interaction_retarget.grasp.ik_config import TRAY_SIDE

    return GraspArmAgent(side="left", object_name="tray", canonical=canonical, side_cfg=TRAY_SIDE)


def make_peg_agent(canonical: dict) -> GraspArmAgent:
    from interaction_retarget.grasp.ik_config import PEG_SIDE

    return GraspArmAgent(side="right", object_name="peg", canonical=canonical, side_cfg=PEG_SIDE)
