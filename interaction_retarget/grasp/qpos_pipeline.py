"""Phase-1: DITTO demo segments — approach → squeeze (fingers) → lift (demo warp)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from interaction_retarget.constants import MIN_GRASP_CONTACT_COUNT, PEG_BODY, TRAY_BODY
from interaction_retarget.grasp.demo_frame_replay import (
    DemoWarpTracks,
    _read_grasp_in_object_frame,
    extract_demo_warp_tracks,
    replay_warp_approach,
    replay_warp_lift,
    replay_warp_squeeze,
)
from interaction_retarget.grasp.lift import hold_tray_before_peg
from interaction_retarget.grasp.qpos_refine import QposRefineResult
from interaction_retarget.grasp.repair import GraspRepairResult, _side_metrics, side_contact_count
from interaction_retarget.sim.contact import AssemblyContactDetector, FrameContact
from interaction_retarget.sim.settle import vec_to_arm_action
from interaction_retarget.tpsr.config import TpsrConfig
from interaction_retarget.tpsr.grasp_filter import GraspFilter, grasp_filter_cfg_from_tpsr

ObjectName = Literal["tray", "peg"]

MIN_TRAY_LIFT_M = 0.030
MIN_PEG_LIFT_M = 0.010


@dataclass
class SideQposGraspResult:
    object_name: ObjectName
    apply23: np.ndarray
    refine: QposRefineResult
    squeeze_fc_ok: bool
    qp_error: float


@dataclass
class QposGraspPipelineReport:
    tray: SideQposGraspResult
    peg: SideQposGraspResult
    repair: GraspRepairResult
    tray_lift_m: float
    peg_lift_m: float
    lift_success: bool
    success: bool


def _measure_side(
    raw_env,
    detector: AssemblyContactDetector,
    *,
    object_name: ObjectName,
    apply23: np.ndarray,
    tpsr_cfg: TpsrConfig,
) -> QposRefineResult:
    side = "left" if object_name == "tray" else "right"
    gf = GraspFilter(grasp_filter_cfg_from_tpsr(tpsr_cfg)).forward(
        raw_env, side=side, object_name=object_name  # type: ignore[arg-type]
    )
    cc = side_contact_count(detector, raw_env, object_name=object_name)  # type: ignore[arg-type]
    return QposRefineResult(
        action23=apply23,
        contact_rmse_m=0.0,
        contact_count=int(cc),
        qp_error=float(gf.max_qp_error),
        fc_ok=bool(gf.ok),
        success=bool(cc >= MIN_GRASP_CONTACT_COUNT),
        nit=0,
        message="ditto_segment_replay",
    )


def run_bimanual_demo_warp_grasp(
    raw_env,
    entry: dict[str, Any],
    *,
    sidecar_dir: Path,
    detector: AssemblyContactDetector,
    tracks: DemoWarpTracks | None = None,
    tpsr_cfg: TpsrConfig | None = None,
    do_squeeze: bool = True,
    squeeze_steps: int = 24,
    do_lift: bool = True,
) -> QposGraspPipelineReport:
    del entry, sidecar_dir, do_squeeze, squeeze_steps
    tpsr_cfg = tpsr_cfg or TpsrConfig(require_qp_fc=False)
    if tracks is None:
        tracks = extract_demo_warp_tracks(entry)

    tray_z0 = float(raw_env._data.xpos[raw_env._model.body(TRAY_BODY).id, 2])
    peg_z0 = float(raw_env._data.xpos[raw_env._model.body(PEG_BODY).id, 2])

    # --- tray: approach → squeeze → lift (all from same demo, DITTO warp) ---
    hold_left, hold_right = replay_warp_approach(raw_env, tracks.tray_approach)
    if int(tracks.tray_squeeze.demo_frames.shape[0]) > 0:
        hold_left, hold_right = replay_warp_squeeze(
            raw_env, tracks.tray_squeeze, active="left"
        )
    _, _, live_tray_hand = _read_grasp_in_object_frame(raw_env, "left")
    tray_lift_hand = np.maximum(tracks.tray_squeeze.left_hand[-1], live_tray_hand)

    if do_lift and int(tracks.tray_lift.demo_frames.shape[0]) > 0:
        right_home = vec_to_arm_action(hold_right)
        hold_left, hold_right = replay_warp_lift(
            raw_env,
            tracks.tray_lift,
            active="left",
            hand_lock=tray_lift_hand,
            lock_right23=right_home,
        )
        hold_tray_before_peg(
            raw_env,
            left_hold=hold_left,
            right_home=right_home,
            detector=detector,
            lift_ref=None,
            hold_steps=8,
        )
        hold_right = right_home

    tray_res = SideQposGraspResult(
        object_name="tray",
        apply23=hold_left,
        refine=_measure_side(raw_env, detector, object_name="tray", apply23=hold_left, tpsr_cfg=tpsr_cfg),
        squeeze_fc_ok=True,
        qp_error=0.0,
    )

    # --- peg: left holds tray; approach → squeeze → lift ---
    hold_left = vec_to_arm_action(hold_left)
    hold_left, hold_right = replay_warp_approach(
        raw_env, tracks.peg_approach, lock_left23=hold_left
    )
    if int(tracks.peg_squeeze.demo_frames.shape[0]) > 0:
        hold_left, hold_right = replay_warp_squeeze(
            raw_env, tracks.peg_squeeze, active="right", lock_left23=hold_left
        )
    _, _, live_peg_hand = _read_grasp_in_object_frame(raw_env, "right")
    peg_lift_hand = np.maximum(tracks.peg_squeeze.right_hand[-1], live_peg_hand)

    if do_lift and int(tracks.peg_lift.demo_frames.shape[0]) > 0:
        hold_left, hold_right = replay_warp_lift(
            raw_env,
            tracks.peg_lift,
            active="right",
            hand_lock=peg_lift_hand,
            lock_left23=hold_left,
        )

    peg_res = SideQposGraspResult(
        object_name="peg",
        apply23=hold_right,
        refine=_measure_side(raw_env, detector, object_name="peg", apply23=hold_right, tpsr_cfg=tpsr_cfg),
        squeeze_fc_ok=True,
        qp_error=0.0,
    )

    contact: FrameContact = detector.compute(raw_env)
    tray_m = _side_metrics(raw_env, detector, contact, object_name="tray")
    peg_m = _side_metrics(raw_env, detector, contact, object_name="peg")
    tray_lift_m = float(raw_env._data.xpos[raw_env._model.body(TRAY_BODY).id, 2]) - tray_z0
    peg_lift_m = float(raw_env._data.xpos[raw_env._model.body(PEG_BODY).id, 2]) - peg_z0
    lift_success = bool(tray_lift_m >= MIN_TRAY_LIFT_M and peg_lift_m >= MIN_PEG_LIFT_M)

    repair = GraspRepairResult(
        action_right=hold_right,
        action_left=hold_left,
        tray=tray_m,
        peg=peg_m,
        repair_iters=0,
        hold_steps=0,
        stable_tray=tray_m.contact_count >= MIN_GRASP_CONTACT_COUNT,
        stable_peg=peg_m.contact_count >= MIN_GRASP_CONTACT_COUNT,
        success=lift_success,
    )
    return QposGraspPipelineReport(
        tray=tray_res,
        peg=peg_res,
        repair=repair,
        tray_lift_m=tray_lift_m,
        peg_lift_m=peg_lift_m,
        lift_success=lift_success,
        success=lift_success,
    )


run_bimanual_qpos_grasp = run_bimanual_demo_warp_grasp
