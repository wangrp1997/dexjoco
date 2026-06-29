"""TPSR: topology + hole constraints wrapped around grasp repair."""

from __future__ import annotations

from typing import Literal

import numpy as np

from interaction_retarget.grasp.metrics import hand_rmse_obj_m
from interaction_retarget.grasp.repair import laplacian_rmse, repair_side_grasp
from interaction_retarget.sim.settle import vec_to_arm_action
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.tpsr.config import TpsrConfig
from interaction_retarget.tpsr.constraints import candidate_acceptable
from interaction_retarget.tpsr.grasp_filter import GraspFilter, grasp_filter_cfg_from_tpsr
from interaction_retarget.tpsr.metrics import TpsrMetrics, hole_params, tpsr_metrics
from interaction_retarget.tpsr.sim_refine import sim_refine_side_grasp

ObjectName = Literal["tray", "peg"]
Side = Literal["left", "right"]


def refine_side_grasp(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    action_right: np.ndarray,
    action_left: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    detector: AssemblyContactDetector,
    canonical: dict,
    cfg: TpsrConfig | None = None,
    skip_greedy: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, TpsrMetrics]:
    """Sim refine (Dexonomy-style) then greedy repair fallback."""
    cfg = cfg or TpsrConfig()
    gf_cfg = grasp_filter_cfg_from_tpsr(cfg)
    right23, left23, sim_iters, metrics = sim_refine_side_grasp(
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
    )
    if metrics.contact_count >= cfg.min_contact_count:
        if not cfg.require_qp_fc:
            return right23, left23, sim_iters, metrics
        qp_ok = GraspFilter(gf_cfg).forward(
            raw_env, side=side, object_name=object_name
        ).ok
        if qp_ok:
            return right23, left23, sim_iters, metrics

    if skip_greedy:
        return right23, left23, sim_iters, metrics

    radius_m, length_m = hole_params(cfg, object_name)
    baseline_lap = laplacian_rmse(raw_env, canonical, object_name=object_name)
    baseline_hand = hand_rmse_obj_m(raw_env, canonical, side=side, object_name=object_name)

    def _accept() -> bool:
        if not candidate_acceptable(
            raw_env,
            canonical,
            object_name=object_name,
            side=side,
            baseline_lap_m=baseline_lap,
            baseline_hand_m=baseline_hand,
            max_lap_drift_m=cfg.max_laplacian_drift_m,
            max_hand_drift_m=cfg.max_hand_drift_m,
            hole_radius_m=radius_m,
            hole_length_m=length_m,
        ):
            return False
        if cfg.require_qp_fc:
            return GraspFilter(gf_cfg).forward(
                raw_env, side=side, object_name=object_name
            ).ok
        return True

    right23, left23, iters = repair_side_grasp(
        raw_env,
        side=side,
        object_name=object_name,
        action_right=right23,
        action_left=left23,
        hold_right=hold_right,
        hold_left=hold_left,
        detector=detector,
        max_iters=cfg.max_iters,
        min_contact_count=cfg.min_contact_count,
        hold_steps=cfg.hold_steps,
        max_laplacian_drift_m=cfg.max_laplacian_drift_m,
        canonical=canonical,
        require_on_table=cfg.require_on_table,
        finger_only=cfg.finger_only,
        accept_fn=_accept,
    )
    if side == "left":
        right23 = vec_to_arm_action(hold_right)
    else:
        left23 = vec_to_arm_action(hold_left)
    metrics = tpsr_metrics(
        raw_env, canonical, object_name=object_name, side=side, detector=detector, cfg=cfg
    )
    return right23, left23, sim_iters + iters, metrics
