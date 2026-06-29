"""Grasp agent with TPSR execute path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from interaction_retarget.grasp.agent import GraspArmAgent
from interaction_retarget.grasp.execute_tpsr import execute_side_grasp_tpsr
from interaction_retarget.grasp.ik_config import GraspSideConfig, side_config
from interaction_retarget.tpsr.config import TpsrConfig

Side = Literal["left", "right"]
ObjectName = Literal["tray", "peg"]


@dataclass
class TpsrGraspArmAgent(GraspArmAgent):
    tpsr_cfg: TpsrConfig | None = None

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
        skip_repair: bool = False,
        finger_repair_only: bool = False,
        direct_reach_steps: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int, dict]:
        cfg = side_cfg or self._cfg()
        return execute_side_grasp_tpsr(
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
            tpsr_cfg=self.tpsr_cfg,
            skip_approach=skip_approach,
            skip_repair=skip_repair,
            finger_repair_only=finger_repair_only,
            direct_reach_steps=direct_reach_steps,
            repair_hold_steps=cfg.repair_hold_steps,
        )


def make_tray_agent_tpsr(canonical: dict, *, fast: bool = False) -> TpsrGraspArmAgent:
    return TpsrGraspArmAgent(
        side="left",
        object_name="tray",
        canonical=canonical,
        side_cfg=side_config("tray", fast=fast),
    )


def make_peg_agent_tpsr(canonical: dict, *, fast: bool = False) -> TpsrGraspArmAgent:
    return TpsrGraspArmAgent(
        side="right",
        object_name="peg",
        canonical=canonical,
        side_cfg=side_config("peg", fast=fast),
    )
