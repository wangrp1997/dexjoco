"""Contact/FC only after execute (no duplicate TPSR). execute_tpsr already runs TPSR+QP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from interaction_retarget.constants import MIN_GRASP_CONTACT_COUNT
from interaction_retarget.contact_refine import ContactRefineReport, refine_demo_contact
from interaction_retarget.grasp.repair import side_contact_count
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.settle import vec_to_arm_action
from interaction_retarget.tpsr.config import TpsrConfig
from interaction_retarget.tpsr.grasp_filter import GraspFilter, grasp_filter_cfg_from_tpsr
from interaction_retarget.tpsr.refine import refine_side_grasp

Side = Literal["left", "right"]
ObjectName = Literal["tray", "peg"]


@dataclass
class GraspFinalizeReport:
    contact_count: int
    qp_ok: bool
    qp_max_error: float
    tpsr_iters: int
    contact_refine: ContactRefineReport | None


def finalize_side_grasp(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    canonical: dict,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    detector: AssemblyContactDetector,
    tpsr_cfg: TpsrConfig,
    contact_refine_iters: int = 12,
) -> tuple[np.ndarray, np.ndarray, GraspFinalizeReport]:
    """接触不足时：TPSR 补一次 → ContactOpt（短）→ GraspFilter FC。已够接触则只验 FC。"""
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    tpsr_iters = 0
    c0 = int(side_contact_count(detector, raw_env, object_name=object_name))

    if c0 < int(tpsr_cfg.min_contact_count):
        hold_right, hold_left, tpsr_iters, _ = refine_side_grasp(
            raw_env,
            side=side,
            object_name=object_name,
            action_right=hold_right,
            action_left=hold_left,
            hold_right=hold_right,
            hold_left=hold_left,
            detector=detector,
            canonical=canonical,
            cfg=tpsr_cfg,
        )

    gf_cfg = grasp_filter_cfg_from_tpsr(tpsr_cfg, ho_collision_thre_m=-0.006)
    c1 = int(side_contact_count(detector, raw_env, object_name=object_name))
    gf_res = GraspFilter(gf_cfg).forward(raw_env, side=side, object_name=object_name)
    crefine: ContactRefineReport | None = None
    need_contactopt = c1 < MIN_GRASP_CONTACT_COUNT or (
        tpsr_cfg.require_qp_fc and not gf_res.ok
    )
    if need_contactopt:
        hold_right, hold_left, crefine = refine_demo_contact(
            raw_env,
            side=side,
            object_name=object_name,
            canonical=canonical,
            hold_right=hold_right,
            hold_left=hold_left,
            detector=detector,
            max_iters=contact_refine_iters,
            grasp_filter_cfg=gf_cfg,
        )
    else:
        crefine = ContactRefineReport(
            contact_count=c1,
            contact_site_rmse_m=0.0,
            contactopt_loss=0.0,
            grasptta_loss=0.0,
            penetration_mean=0.0,
            laplacian_rmse_m=0.0,
            hand_rmse_m=0.0,
            total_score=0.0,
            qp_ok=bool(gf_res.ok or not tpsr_cfg.require_qp_fc),
            qp_max_error=float(gf_res.max_qp_error),
            improved=False,
        )

    c_final = int(side_contact_count(detector, raw_env, object_name=object_name))
    report = GraspFinalizeReport(
        contact_count=c_final,
        qp_ok=bool(crefine.qp_ok),
        qp_max_error=float(crefine.qp_max_error),
        tpsr_iters=int(tpsr_iters),
        contact_refine=crefine,
    )
    return hold_right, hold_left, report
