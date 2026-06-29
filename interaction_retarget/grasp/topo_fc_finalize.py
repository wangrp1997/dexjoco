"""拓扑 δ* 保持下的 ContactOpt + TPSR + FC 收敛（Laplacian guard 全程有效）。"""

from __future__ import annotations

from typing import Literal

from interaction_retarget.constants import MIN_GRASP_CONTACT_COUNT
from interaction_retarget.contact_refine import refine_demo_contact
from interaction_retarget.contact_refine.optimize_pose import ContactOptPoseConfig
from interaction_retarget.grasp.metrics import hand_rmse_obj_m, laplacian_rmse_obj_m
from interaction_retarget.grasp.repair import side_contact_count
from interaction_retarget.grasp.staged_grasp import re_squeeze_fc
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.settle import read_arm_action, vec_to_arm_action
from interaction_retarget.skill_replay.grasp_finalize import GraspFinalizeReport, finalize_side_grasp
from interaction_retarget.tpsr.config import TpsrConfig
from interaction_retarget.tpsr.constraints import candidate_acceptable
from interaction_retarget.tpsr.metrics import hole_params
from interaction_retarget.tpsr.grasp_filter import GraspFilter, GraspFilterConfig, grasp_filter_cfg_from_tpsr
from interaction_retarget.tpsr.refine import refine_side_grasp

Side = Literal["left", "right"]
ObjectName = Literal["tray", "peg"]


def _grasp_filter_cfg(tpsr_cfg: TpsrConfig) -> GraspFilterConfig:
    return grasp_filter_cfg_from_tpsr(tpsr_cfg, ho_collision_thre_m=-0.006)


def topology_grasp_ok(
    raw_env,
    detector: AssemblyContactDetector,
    canonical: dict,
    *,
    object_name: ObjectName,
    tpsr_cfg: TpsrConfig,
    max_lap_rmse_m: float = 0.040,
    max_hand_rmse_m: float | None = None,
) -> tuple[bool, str, float, float, int]:
    side: Side = "left" if object_name == "tray" else "right"
    if max_hand_rmse_m is None:
        max_hand_rmse_m = 0.055 if object_name == "tray" else 0.095
    lap = laplacian_rmse_obj_m(raw_env, canonical, side=side, object_name=object_name)
    hand = hand_rmse_obj_m(raw_env, canonical, side=side, object_name=object_name)
    cc = int(side_contact_count(detector, raw_env, object_name=object_name))
    if hand > max_hand_rmse_m:
        return False, f"hand_rmse={hand * 1e3:.1f}mm", lap, hand, cc
    if lap > max_lap_rmse_m:
        return False, f"lap={lap * 1e3:.1f}mm", lap, hand, cc
    if cc < MIN_GRASP_CONTACT_COUNT:
        return False, f"contact={cc}", lap, hand, cc
    return True, "ok", lap, hand, cc


def _topo_acceptable(
    raw_env,
    canonical: dict,
    *,
    object_name: ObjectName,
    side: Side,
    tpsr_cfg: TpsrConfig,
    baseline_lap: float,
    baseline_hand: float,
) -> bool:
    radius_m, length_m = hole_params(tpsr_cfg, object_name)
    return candidate_acceptable(
        raw_env,
        canonical,
        object_name=object_name,
        side=side,
        baseline_lap_m=baseline_lap,
        baseline_hand_m=baseline_hand,
        max_lap_drift_m=tpsr_cfg.max_laplacian_drift_m,
        max_hand_drift_m=tpsr_cfg.max_hand_drift_m,
        hole_radius_m=radius_m,
        hole_length_m=length_m,
    )


def ensure_side_topo_fc(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    canonical: dict,
    hold_right,
    hold_left,
    detector: AssemblyContactDetector,
    tpsr_cfg: TpsrConfig,
    contact_refine_iters: int,
) -> tuple[object, object, GraspFinalizeReport, bool, str]:
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    refine_iters = max(min(int(contact_refine_iters), 20), 8)
    gf_cfg = _grasp_filter_cfg(tpsr_cfg)

    baseline_lap = laplacian_rmse_obj_m(raw_env, canonical, side=side, object_name=object_name)
    baseline_hand = hand_rmse_obj_m(raw_env, canonical, side=side, object_name=object_name)

    hold_right, hold_left, fin = finalize_side_grasp(
        raw_env,
        side=side,
        object_name=object_name,
        canonical=canonical,
        hold_right=hold_right,
        hold_left=hold_left,
        detector=detector,
        tpsr_cfg=tpsr_cfg,
        contact_refine_iters=refine_iters,
    )
    topo_ok, topo_msg, _, _, _ = topology_grasp_ok(
        raw_env, detector, canonical, object_name=object_name, tpsr_cfg=tpsr_cfg
    )

    if not fin.qp_ok and topo_ok:
        strong_pose = ContactOptPoseConfig(
            n_iter=max(refine_iters * 2, 16),
            maxfun=80,
            w_laplacian=140.0,
            max_laplacian_drift_m=float(tpsr_cfg.max_laplacian_drift_m),
            max_hand_drift_m=float(tpsr_cfg.max_hand_drift_m),
        )
        hold_right, hold_left, _ = refine_demo_contact(
            raw_env,
            side=side,
            object_name=object_name,
            canonical=canonical,
            hold_right=hold_right,
            hold_left=hold_left,
            detector=detector,
            max_iters=strong_pose.n_iter,
            pose_cfg=strong_pose,
            grasp_filter_cfg=gf_cfg,
        )
        if _topo_acceptable(
            raw_env,
            canonical,
            object_name=object_name,
            side=side,
            tpsr_cfg=tpsr_cfg,
            baseline_lap=baseline_lap,
            baseline_hand=baseline_hand,
        ):
            gf = GraspFilter(gf_cfg).forward(raw_env, side=side, object_name=object_name)
            fin = GraspFinalizeReport(
                contact_count=int(side_contact_count(detector, raw_env, object_name=object_name)),
                qp_ok=bool(gf.ok),
                qp_max_error=float(gf.max_qp_error),
                tpsr_iters=fin.tpsr_iters,
                contact_refine=fin.contact_refine,
            )
        topo_ok, topo_msg, _, _, _ = topology_grasp_ok(
            raw_env, detector, canonical, object_name=object_name, tpsr_cfg=tpsr_cfg
        )

    cc = int(side_contact_count(detector, raw_env, object_name=object_name))
    if not fin.qp_ok and topo_ok and cc >= MIN_GRASP_CONTACT_COUNT:
        hold_right, hold_left, tpsr_n, _ = refine_side_grasp(
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
        if _topo_acceptable(
            raw_env,
            canonical,
            object_name=object_name,
            side=side,
            tpsr_cfg=tpsr_cfg,
            baseline_lap=baseline_lap,
            baseline_hand=baseline_hand,
        ):
            gf = GraspFilter(gf_cfg).forward(raw_env, side=side, object_name=object_name)
            fin = GraspFinalizeReport(
                contact_count=int(side_contact_count(detector, raw_env, object_name=object_name)),
                qp_ok=bool(gf.ok),
                qp_max_error=float(gf.max_qp_error),
                tpsr_iters=int(fin.tpsr_iters) + int(tpsr_n),
                contact_refine=fin.contact_refine,
            )
        topo_ok, topo_msg, _, _, _ = topology_grasp_ok(
            raw_env, detector, canonical, object_name=object_name, tpsr_cfg=tpsr_cfg
        )

    cc = int(side_contact_count(detector, raw_env, object_name=object_name))
    if not fin.qp_ok and topo_ok and cc < MIN_GRASP_CONTACT_COUNT:
        hold_right, hold_left, tpsr_n, _ = refine_side_grasp(
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
        if _topo_acceptable(
            raw_env,
            canonical,
            object_name=object_name,
            side=side,
            tpsr_cfg=tpsr_cfg,
            baseline_lap=baseline_lap,
            baseline_hand=baseline_hand,
        ):
            hold_right, hold_left, fin2 = finalize_side_grasp(
                raw_env,
                side=side,
                object_name=object_name,
                canonical=canonical,
                hold_right=hold_right,
                hold_left=hold_left,
                detector=detector,
                tpsr_cfg=tpsr_cfg,
                contact_refine_iters=refine_iters,
            )
            fin = GraspFinalizeReport(
                contact_count=fin2.contact_count,
                qp_ok=fin2.qp_ok,
                qp_max_error=fin2.qp_max_error,
                tpsr_iters=int(fin.tpsr_iters) + int(tpsr_n) + int(fin2.tpsr_iters),
                contact_refine=fin2.contact_refine,
            )
        topo_ok, topo_msg, _, _, _ = topology_grasp_ok(
            raw_env, detector, canonical, object_name=object_name, tpsr_cfg=tpsr_cfg
        )

    cc = int(side_contact_count(detector, raw_env, object_name=object_name))
    if not fin.qp_ok and topo_ok and cc >= MIN_GRASP_CONTACT_COUNT:
        hold_right, hold_left, sq_ok, sq_qp = re_squeeze_fc(
            raw_env,
            side=side,
            object_name=object_name,
            canonical=canonical,
            hold_right=hold_right,
            hold_left=hold_left,
            tpsr_cfg=tpsr_cfg,
            max_rounds=3,
        )
        if _topo_acceptable(
            raw_env,
            canonical,
            object_name=object_name,
            side=side,
            tpsr_cfg=tpsr_cfg,
            baseline_lap=baseline_lap,
            baseline_hand=baseline_hand,
        ):
            gf = GraspFilter(gf_cfg).forward(raw_env, side=side, object_name=object_name)
            fin = GraspFinalizeReport(
                contact_count=int(side_contact_count(detector, raw_env, object_name=object_name)),
                qp_ok=bool(gf.ok or sq_ok),
                qp_max_error=float(min(sq_qp, gf.max_qp_error)),
                tpsr_iters=fin.tpsr_iters,
                contact_refine=fin.contact_refine,
            )
        if side == "left":
            hold_left = vec_to_arm_action(read_arm_action(raw_env, "left"))
        else:
            hold_right = vec_to_arm_action(read_arm_action(raw_env, "right"))

    topo_ok, topo_msg, _, _, _ = topology_grasp_ok(
        raw_env, detector, canonical, object_name=object_name, tpsr_cfg=tpsr_cfg
    )
    return hold_right, hold_left, fin, bool(topo_ok), str(topo_msg)
