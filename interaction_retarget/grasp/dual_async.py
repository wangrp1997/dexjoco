"""双智能体 L1：拓扑 δ* + ContactOpt + FC，全过才抬升。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from interaction_retarget.constants import MIN_GRASP_CONTACT_COUNT, PEG_BODY, TRAY_BODY
from interaction_retarget.contact_refine import refine_demo_contact
from interaction_retarget.grasp.agent_tpsr import TpsrGraspArmAgent
from interaction_retarget.grasp.ik import GraspIkResult
from interaction_retarget.grasp.lift import execute_peg_lift, execute_tray_lift, verify_lift_fc
from interaction_retarget.grasp.locked_hold import enforce_locked_passive
from interaction_retarget.grasp.pipeline import _ik_grasp_ready
from interaction_retarget.grasp.plan_side_grasp import ik_grasp_target, plan_side_grasp
from interaction_retarget.grasp.pipeline_tpsr import _canonical_arm_target, _skipped_peg_ik
from interaction_retarget.grasp.repair import side_contact_count
from interaction_retarget.grasp.staged_grasp import re_squeeze_fc
from interaction_retarget.grasp.track_lift import TrackLiftReport, strong_lift_tpsr_cfg
from interaction_retarget.skill_replay.demo_grasp import demo_lift_world_dz
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.settle import read_arm_action, vec_to_arm_action
from interaction_retarget.skill_replay.grasp_finalize import GraspFinalizeReport
from interaction_retarget.tpsr.config import TpsrConfig
from interaction_retarget.tpsr.grasp_filter import grasp_filter_cfg_from_tpsr


@dataclass
class DualAsyncResult:
    left_hold: np.ndarray
    right_hold: np.ndarray
    left_locked: np.ndarray
    tray_fin: GraspFinalizeReport
    peg_fin: GraspFinalizeReport
    tray_lift: TrackLiftReport
    peg_lift: TrackLiftReport
    tray_topo_ok: bool
    tray_topo_msg: str
    peg_topo_ok: bool
    peg_topo_msg: str
    tray_fc_ok: bool
    peg_fc_ok: bool
    tray_qp_err: float
    peg_qp_err: float
    tray_lifted: bool
    peg_lifted: bool
    tray_lift_m: float
    peg_lift_m: float
    repair_iters: int
    peg_ik: GraspIkResult


def _grasp_cfg(cfg: TpsrConfig, *, side: Literal["left", "right"]) -> TpsrConfig:
    extra = 16 if side == "left" else 20
    return replace(cfg, squeeze_steps=int(cfg.squeeze_steps) + extra, require_qp_fc=True)


def _object_dz(raw_env, body: str, z0: float) -> float:
    bid = int(raw_env._model.body(body).id)
    return float(raw_env._data.xpos[bid, 2]) - z0


def _gf_cfg(tpsr_cfg: TpsrConfig):
    return grasp_filter_cfg_from_tpsr(tpsr_cfg, ho_collision_thre_m=-0.006)


def _side_ready_for_lift(
    raw_env,
    detector: AssemblyContactDetector,
    *,
    object_name: Literal["tray", "peg"],
    fin: GraspFinalizeReport,
    fc_ok: bool,
    topo_ok: bool,
) -> bool:
    cc = int(side_contact_count(detector, raw_env, object_name=object_name))
    return bool(topo_ok and cc >= MIN_GRASP_CONTACT_COUNT and (fin.qp_ok or fc_ok))


def _ensure_fc_before_lift(
    raw_env,
    *,
    side: Literal["left", "right"],
    object_name: Literal["tray", "peg"],
    canonical: dict,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    detector: AssemblyContactDetector,
    tpsr_cfg: TpsrConfig,
    contact_refine_iters: int,
    topo_ok: bool,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    ok, qp = verify_lift_fc(
        raw_env,
        side=side,
        object_name=object_name,
        tpsr_cfg=tpsr_cfg,
        hold_right=hold_right,
        hold_left=hold_left,
        detector=detector,
    )
    if ok:
        return hold_right, hold_left, ok, qp
    if topo_ok:
        hold_right, hold_left, _ = refine_demo_contact(
            raw_env,
            side=side,
            object_name=object_name,
            canonical=canonical,
            hold_right=hold_right,
            hold_left=hold_left,
            detector=detector,
            max_iters=max(contact_refine_iters, 8),
            grasp_filter_cfg=_gf_cfg(tpsr_cfg),
        )
        ok, qp = verify_lift_fc(
            raw_env,
            side=side,
            object_name=object_name,
            tpsr_cfg=tpsr_cfg,
            hold_right=hold_right,
            hold_left=hold_left,
            detector=detector,
        )
        if ok:
            return hold_right, hold_left, ok, qp
    hold_right, hold_left, _, sq_qp = re_squeeze_fc(
        raw_env,
        side=side,
        object_name=object_name,
        canonical=canonical,
        hold_right=hold_right,
        hold_left=hold_left,
        tpsr_cfg=tpsr_cfg,
        max_rounds=2,
    )
    hold_right = vec_to_arm_action(read_arm_action(raw_env, "right"))
    hold_left = vec_to_arm_action(read_arm_action(raw_env, "left"))
    ok, qp = verify_lift_fc(
        raw_env,
        side=side,
        object_name=object_name,
        tpsr_cfg=tpsr_cfg,
        hold_right=hold_right,
        hold_left=hold_left,
        detector=detector,
    )
    return hold_right, hold_left, bool(ok), float(qp if ok else min(qp, sq_qp))


def run_l1_dual_async(
    raw_env,
    *,
    left_agent: TpsrGraspArmAgent,
    right_agent: TpsrGraspArmAgent,
    canonical_tray: dict,
    canonical_peg: dict,
    detector: AssemblyContactDetector,
    tpsr_cfg: TpsrConfig,
    lift_ref: dict[str, Any] | None,
    manifest_entry: dict[str, Any] | None,
    contact_refine_iters: int = 8,
    skip_peg_lift: bool = False,
    live_record: bool = False,
    topology_ok_fn=None,
) -> DualAsyncResult:
    _ = topology_ok_fn
    right_home = vec_to_arm_action(read_arm_action(raw_env, "right"))
    left_home = vec_to_arm_action(read_arm_action(raw_env, "left"))
    lift_steps = 24 if live_record else 32

    left_target = _canonical_arm_target(raw_env, canonical_tray, "tray")
    _, tray_ik = left_agent.plan(
        raw_env,
        hold_right=right_home,
        hold_left=left_home,
        detector=detector,
        restore_env=True,
        side_cfg=left_agent.side_cfg,
    )
    left_target = ik_grasp_target(tray_ik, "left", left_target)

    repair_iters = 0
    tray_cfg = _grasp_cfg(tpsr_cfg, side="left")
    peg_cfg = _grasp_cfg(tpsr_cfg, side="right")

    right_hold, left_hold, tray_fin, n, tray_topo_ok, tray_topo_msg = plan_side_grasp(
        left_agent,
        raw_env,
        target23=left_target,
        hold_right=right_home,
        hold_left=left_home,
        detector=detector,
        ik=tray_ik,
        tpsr_cfg=tray_cfg,
        contact_refine_iters=contact_refine_iters,
        live_record=live_record,
    )
    repair_iters += n
    left_locked = vec_to_arm_action(read_arm_action(raw_env, "left"))

    right_hold, left_locked, tray_fc_ok, tray_qp = _ensure_fc_before_lift(
        raw_env,
        side="left",
        object_name="tray",
        canonical=canonical_tray,
        hold_right=vec_to_arm_action(read_arm_action(raw_env, "right")),
        hold_left=left_locked,
        detector=detector,
        tpsr_cfg=tray_cfg,
        contact_refine_iters=contact_refine_iters,
        topo_ok=tray_topo_ok,
    )
    left_locked = vec_to_arm_action(read_arm_action(raw_env, "left"))
    passive_right = vec_to_arm_action(read_arm_action(raw_env, "right"))
    enforce_locked_passive(raw_env, locked_left=left_locked, locked_right=passive_right, n_substeps=8)

    tray_lift = TrackLiftReport("tray", 0.0, 0.0, 0, False, float(tray_qp), 0, False)
    peg_lift = TrackLiftReport("peg", 0.0, 0.0, 0, False, 1.0, 0, False)
    tray_lifted = False
    peg_lifted = False
    tray_lift_m = 0.0
    peg_lift_m = 0.0
    peg_fin = GraspFinalizeReport(0, False, 1.0, 0, None)
    peg_fc_ok = False
    peg_qp = 1.0
    peg_ik = _skipped_peg_ik()
    peg_topo_ok = False
    peg_topo_msg = "pending"

    if _side_ready_for_lift(
        raw_env, detector, object_name="tray", fin=tray_fin, fc_ok=tray_fc_ok, topo_ok=tray_topo_ok
    ):
        lift_cfg = strong_lift_tpsr_cfg(tray_cfg, extra_squeeze=12)
        tray_z0 = float(raw_env._data.xpos[int(raw_env._model.body(TRAY_BODY).id), 2])
        tray_dz_tgt = (
            demo_lift_world_dz(manifest_entry, lift_ref, "tray")
            if manifest_entry is not None and lift_ref is not None
            else 0.05
        )
        left_locked = execute_tray_lift(
            raw_env,
            grasp_left=left_locked,
            hold_right=passive_right,
            lift_ref=lift_ref,
            detector=detector,
            lift_height_m=tray_dz_tgt,
            steps=lift_steps,
            hold_steps=8,
            pre_lift_settle=12,
            lock_passive_arm=True,
            object_z_only=True,
            canonical=canonical_tray,
            tpsr_cfg=lift_cfg,
        )
        tray_lift_m = _object_dz(raw_env, TRAY_BODY, tray_z0)
        tray_lifted = tray_lift_m >= 0.030
        tray_lift = TrackLiftReport(
            object_name="tray",
            object_dz_m=tray_lift_m,
            target_dz_m=tray_dz_tgt,
            contact_min=int(side_contact_count(detector, raw_env, object_name="tray")),
            fc_ok=bool(tray_fc_ok),
            qp_max_error=float(tray_qp),
            steps_executed=lift_steps,
            success=tray_lifted,
        )
        left_locked = vec_to_arm_action(read_arm_action(raw_env, "left"))
        passive_right = vec_to_arm_action(read_arm_action(raw_env, "right"))
        enforce_locked_passive(raw_env, locked_left=left_locked, locked_right=passive_right, n_substeps=8)

    if tray_lifted and not skip_peg_lift:
        peg_canonical_target = _canonical_arm_target(raw_env, canonical_peg, "peg")
        right_target, peg_ik = right_agent.plan(
            raw_env,
            hold_right=right_home,
            hold_left=left_locked,
            detector=detector,
            restore_env=True,
            side_cfg=right_agent.side_cfg,
        )
        right_target = ik_grasp_target(peg_ik, "right", peg_canonical_target)
        peg_direct = None if _ik_grasp_ready(peg_ik) else (100 if peg_ik.contact_count <= 0 else 70)

        right_hold, _, peg_fin, n, peg_topo_ok, peg_topo_msg = plan_side_grasp(
            right_agent,
            raw_env,
            target23=right_target,
            hold_right=right_home,
            hold_left=left_locked,
            detector=detector,
            ik=peg_ik,
            tpsr_cfg=peg_cfg,
            contact_refine_iters=contact_refine_iters,
            live_record=live_record,
            direct_reach_steps=peg_direct,
        )
        repair_iters += n
        right_hold, _, peg_fc_ok, peg_qp = _ensure_fc_before_lift(
            raw_env,
            side="right",
            object_name="peg",
            canonical=canonical_peg,
            hold_right=right_hold,
            hold_left=left_locked,
            detector=detector,
            tpsr_cfg=peg_cfg,
            contact_refine_iters=contact_refine_iters,
            topo_ok=peg_topo_ok,
        )
        left_locked, _, _, _ = re_squeeze_fc(
            raw_env,
            side="left",
            object_name="tray",
            canonical=canonical_tray,
            hold_right=right_hold,
            hold_left=left_locked,
            tpsr_cfg=replace(tray_cfg, require_qp_fc=False),
            max_rounds=1,
        )
        left_locked = vec_to_arm_action(read_arm_action(raw_env, "left"))
        right_hold = vec_to_arm_action(read_arm_action(raw_env, "right"))
        enforce_locked_passive(raw_env, locked_left=left_locked, locked_right=right_hold, n_substeps=6)

        if _side_ready_for_lift(
            raw_env, detector, object_name="peg", fin=peg_fin, fc_ok=peg_fc_ok, topo_ok=peg_topo_ok
        ):
            peg_z0 = float(raw_env._data.xpos[int(raw_env._model.body(PEG_BODY).id), 2])
            peg_dz_tgt = (
                demo_lift_world_dz(manifest_entry, lift_ref, "peg")
                if manifest_entry is not None and lift_ref is not None
                else 0.05
            )
            right_hold = execute_peg_lift(
                raw_env,
                grasp_right=right_hold,
                hold_left=left_locked,
                lift_ref=lift_ref,
                detector=detector,
                steps=lift_steps,
                pre_lift_settle=12,
                object_z_only=True,
                canonical=canonical_peg,
                tpsr_cfg=strong_lift_tpsr_cfg(peg_cfg, extra_squeeze=16),
            )
            peg_lift_m = _object_dz(raw_env, PEG_BODY, peg_z0)
            peg_lifted = peg_lift_m >= 0.010
            peg_lift = TrackLiftReport(
                object_name="peg",
                object_dz_m=peg_lift_m,
                target_dz_m=peg_dz_tgt,
                contact_min=int(side_contact_count(detector, raw_env, object_name="peg")),
                fc_ok=bool(peg_fc_ok),
                qp_max_error=float(peg_qp),
                steps_executed=lift_steps,
                success=peg_lifted,
            )
            enforce_locked_passive(raw_env, locked_left=left_locked, locked_right=right_hold, n_substeps=4)
    elif not tray_lifted:
        peg_topo_msg = "skipped_tray_not_lifted"

    right_hold = vec_to_arm_action(read_arm_action(raw_env, "right"))
    left_locked = vec_to_arm_action(read_arm_action(raw_env, "left"))
    if not tray_fin.qp_ok and not tray_fc_ok:
        tray_topo_msg = f"{tray_topo_msg};fc:qp={tray_qp:.3f}"
    if peg_fin.qp_max_error < 1.0 and not peg_fin.qp_ok and not peg_fc_ok:
        peg_topo_msg = f"{peg_topo_msg};fc:qp={peg_qp:.3f}"

    return DualAsyncResult(
        left_hold=left_locked,
        right_hold=right_hold,
        left_locked=left_locked,
        tray_fin=tray_fin,
        peg_fin=peg_fin,
        tray_lift=tray_lift,
        peg_lift=peg_lift,
        tray_topo_ok=bool(tray_topo_ok),
        tray_topo_msg=str(tray_topo_msg),
        peg_topo_ok=bool(peg_topo_ok),
        peg_topo_msg=str(peg_topo_msg),
        tray_fc_ok=bool(tray_fc_ok or tray_fin.qp_ok),
        peg_fc_ok=bool(peg_fc_ok or peg_fin.qp_ok),
        tray_qp_err=float(tray_qp),
        peg_qp_err=float(peg_qp),
        tray_lifted=tray_lifted,
        peg_lifted=peg_lifted,
        tray_lift_m=tray_lift_m,
        peg_lift_m=peg_lift_m,
        repair_iters=repair_iters,
        peg_ik=peg_ik,
    )
