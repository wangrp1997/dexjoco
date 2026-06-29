"""Dexonomy-style sim local refine: contact forces via finger close + scored search."""

from __future__ import annotations

from typing import Literal

import numpy as np

from interaction_retarget.constants import MIN_GRASP_CONTACT_COUNT, PEG_BODY, TRAY_BODY
from interaction_retarget.grasp.ik import _action23_to_x22, _physics_hold_active
from interaction_retarget.grasp.metrics import hand_rmse_obj_m, laplacian_rmse_obj_m
from interaction_retarget.grasp.repair import (
    _close_fingers,
    _hand_joint_bounds,
    _nudge_mocap_toward_object,
    laplacian_rmse,
    side_contact_count,
)
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.settle import read_arm_action, settle_bimanual_actions, vec_to_arm_action
from interaction_retarget.sim.state import restore_sim, snapshot_sim
from interaction_retarget.tpsr.config import TpsrConfig
from interaction_retarget.tpsr.contact_physics import max_penetration_m
from interaction_retarget.tpsr.constraints import candidate_acceptable, hole_clearance_violation_m
from interaction_retarget.tpsr.grasp_filter import GraspFilter, grasp_filter_cfg_from_tpsr
from interaction_retarget.tpsr.metrics import TpsrMetrics, hole_params, tpsr_metrics

ObjectName = Literal["tray", "peg"]
Side = Literal["left", "right"]

_FINGER_DELTAS = (-0.08, -0.06, -0.04, -0.025, -0.015, -0.008, 0.008, 0.015)
_MOCAP_STEPS_M = (0.002, 0.005, 0.008, 0.012, 0.018, 0.025, 0.032)
_MOCAP_BOOTSTRAP_M = (0.035, 0.028, 0.022, 0.016, 0.012, 0.008, 0.005)
_CUMULATIVE_NUDGE_M = 0.04
_CUMULATIVE_NUDGE_MAX = 14
_CUMULATIVE_REACH_DIST_M = 0.12


def _filter_cfg(cfg: TpsrConfig):
    return grasp_filter_cfg_from_tpsr(cfg)


def _qp_ok(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    cfg: TpsrConfig,
) -> tuple[bool, float]:
    if not cfg.require_qp_fc:
        return True, 0.0
    res = GraspFilter(_filter_cfg(cfg)).forward(
        raw_env, side=side, object_name=object_name
    )
    return res.ok, res.max_qp_error


def _active23(side: Side, right23: np.ndarray, left23: np.ndarray) -> np.ndarray:
    return left23 if side == "left" else right23


def _set_active(side: Side, right23: np.ndarray, left23: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = vec_to_arm_action(active)
    if side == "left":
        return right23, active
    return active, left23


def _passive_from_hold(
    side: Side,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    right23: np.ndarray,
    left23: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Active side from refine; passive side keeps the hold command (no sim readback drift)."""
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    right23 = vec_to_arm_action(right23)
    left23 = vec_to_arm_action(left23)
    if side == "left":
        return hold_right, left23
    return right23, hold_left


def _object_body_name(object_name: ObjectName) -> str:
    return TRAY_BODY if object_name == "tray" else PEG_BODY


def _mocap_object_dist_m(raw_env, action23: np.ndarray, *, object_name: ObjectName) -> float:
    action23 = vec_to_arm_action(action23)
    bid = int(raw_env._model.body(_object_body_name(object_name)).id)
    obj_pos = np.asarray(raw_env._data.xpos[bid], dtype=np.float64)
    return float(np.linalg.norm(obj_pos - action23[0:3]))


def _cumulative_reach_object(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    right23: np.ndarray,
    left23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    detector: AssemblyContactDetector,
    cfg: TpsrConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk mocap toward object when δ* target is far (random seeds: peg often 30–50 cm off)."""
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    right23 = vec_to_arm_action(right23)
    left23 = vec_to_arm_action(left23)
    lo, hi = _hand_joint_bounds(raw_env._model, side)
    active = _active23(side, right23, left23)
    if _mocap_object_dist_m(raw_env, active, object_name=object_name) <= _CUMULATIVE_REACH_DIST_M:
        return _passive_from_hold(side, hold_right, hold_left, right23, left23)

    for _ in range(_CUMULATIVE_NUDGE_MAX):
        if side_contact_count(detector, raw_env, object_name=object_name) >= cfg.min_contact_count:
            break
        active = _nudge_mocap_toward_object(raw_env, active, object_name=object_name, step_m=_CUMULATIVE_NUDGE_M)
        nr, nl = _set_active(side, right23, left23, active)
        _apply_pose(
            raw_env,
            side=side,
            right23=nr,
            left23=nl,
            hold_right=hold_right,
            hold_left=hold_left,
            hold_steps=max(cfg.hold_steps, 3),
        )
        active = vec_to_arm_action(read_arm_action(raw_env, side))
        for delta in (-0.04, -0.025, -0.015):
            active2 = _close_fingers(active, delta, lo, hi)
            nr2, nl2 = _set_active(side, nr, nl, active2)
            _apply_pose(
                raw_env,
                side=side,
                right23=nr2,
                left23=nl2,
                hold_right=hold_right,
                hold_left=hold_left,
                hold_steps=max(cfg.hold_steps, 2),
            )
            if side_contact_count(detector, raw_env, object_name=object_name) >= cfg.min_contact_count:
                break

    right23 = vec_to_arm_action(read_arm_action(raw_env, "right"))
    left23 = vec_to_arm_action(read_arm_action(raw_env, "left"))
    return _passive_from_hold(side, hold_right, hold_left, right23, left23)


def _score(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    detector: AssemblyContactDetector,
    canonical: dict,
    baseline_lap: float,
    baseline_hand: float,
    cfg: TpsrConfig,
    bootstrap: bool = False,
) -> float:
    lap_drift = cfg.max_laplacian_drift_m * (cfg.bootstrap_lap_drift_scale if bootstrap else 1.0)
    hand_drift = cfg.max_hand_drift_m * (cfg.bootstrap_lap_drift_scale if bootstrap else 1.0)
    radius_m, length_m = hole_params(cfg, object_name)
    if not candidate_acceptable(
        raw_env,
        canonical,
        object_name=object_name,
        side=side,
        baseline_lap_m=baseline_lap,
        baseline_hand_m=baseline_hand,
        max_lap_drift_m=lap_drift,
        max_hand_drift_m=hand_drift,
        hole_radius_m=radius_m,
        hole_length_m=length_m,
    ):
        return -1e6
    contact = float(side_contact_count(detector, raw_env, object_name=object_name))
    pen = max_penetration_m(raw_env, side=side, object_name=object_name)
    lap = laplacian_rmse(raw_env, canonical, object_name=object_name)
    qp_pass, qp_err = _qp_ok(raw_env, side=side, object_name=object_name, cfg=cfg)
    if cfg.require_qp_fc and not qp_pass:
        return -1e6
    return (
        contact * 100.0
        - pen * 800.0
        - max(0.0, lap - baseline_lap) * (80.0 if bootstrap else 120.0)
        - qp_err * 500.0
    )


def _bootstrap_mocap(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    right23: np.ndarray,
    left23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    detector: AssemblyContactDetector,
    canonical: dict,
    baseline_lap: float,
    baseline_hand: float,
    cfg: TpsrConfig,
    sim0,
) -> tuple[np.ndarray, np.ndarray, float, object]:
    """Large mocap nudges toward object when IK gave zero contact."""
    best_r, best_l = right23, left23
    best_score = -1e9
    best_snap = sim0
    active = _active23(side, right23, left23)
    for step in _MOCAP_BOOTSTRAP_M:
        updated = _nudge_mocap_toward_object(raw_env, active, object_name=object_name, step_m=step)
        nr, nl = _set_active(side, right23, left23, updated)
        restore_sim(raw_env, sim0)
        _apply_pose(
            raw_env,
            side=side,
            right23=nr,
            left23=nl,
            hold_right=hold_right,
            hold_left=hold_left,
            hold_steps=max(cfg.hold_steps, 3),
        )
        lo, hi = _hand_joint_bounds(raw_env._model, side)
        for delta in (-0.04, -0.025, -0.015):
            active2 = _close_fingers(vec_to_arm_action(read_arm_action(raw_env, side)), delta, lo, hi)
            nr2, nl2 = _set_active(side, nr, nl, active2)
            restore_sim(raw_env, sim0)
            _apply_pose(
                raw_env,
                side=side,
                right23=nr2,
                left23=nl2,
                hold_right=hold_right,
                hold_left=hold_left,
                hold_steps=max(cfg.hold_steps, 3),
            )
            sc = _score(
                raw_env,
                side=side,
                object_name=object_name,
                detector=detector,
                canonical=canonical,
                baseline_lap=baseline_lap,
                baseline_hand=baseline_hand,
                cfg=cfg,
                bootstrap=True,
            )
            if sc > best_score:
                best_score = sc
                best_r = vec_to_arm_action(read_arm_action(raw_env, "right"))
                best_l = vec_to_arm_action(read_arm_action(raw_env, "left"))
                best_snap = snapshot_sim(raw_env)
            if side_contact_count(detector, raw_env, object_name=object_name) >= cfg.min_contact_count:
                return best_r, best_l, best_score, best_snap
    return best_r, best_l, best_score, best_snap


def _apply_pose(
    raw_env,
    *,
    side: Side,
    right23: np.ndarray,
    left23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    hold_steps: int,
) -> None:
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    active = _active23(side, right23, left23)
    for _ in range(max(int(hold_steps), 1)):
        if side == "left":
            settle_bimanual_actions(raw_env, right23=hold_right, left23=active, n_substeps=2)
        else:
            settle_bimanual_actions(raw_env, right23=active, left23=hold_left, n_substeps=2)


def _progressive_close(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    right23: np.ndarray,
    left23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    canonical: dict,
    detector: AssemblyContactDetector,
    baseline_lap: float,
    baseline_hand: float,
    cfg: TpsrConfig,
    bootstrap: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Dexonomy squeeze: ramp fingers toward demo median while sim settles."""
    lo, hi = _hand_joint_bounds(raw_env._model, side)
    target_hand = np.asarray(canonical.get("hand_joint_median", _active23(side, right23, left23)[7:23]), dtype=np.float64)
    if target_hand.size != 16:
        target_hand = _active23(side, right23, left23)[7:23]
    open_hand = np.zeros(16, dtype=np.float64)
    best_score = -1e9
    best_r, best_l = right23, left23
    n = max(int(cfg.sim_close_steps), 4)
    for i in range(1, n + 1):
        t = i / n
        hand = (1.0 - t) * open_hand + t * target_hand
        active = _active23(side, right23, left23).copy()
        active[7:23] = hand
        right23, left23 = _set_active(side, right23, left23, active)
        _apply_pose(
            raw_env,
            side=side,
            right23=right23,
            left23=left23,
            hold_right=hold_right,
            hold_left=hold_left,
            hold_steps=cfg.hold_steps,
        )
        right23 = vec_to_arm_action(read_arm_action(raw_env, "right"))
        left23 = vec_to_arm_action(read_arm_action(raw_env, "left"))
        sc = _score(
            raw_env,
            side=side,
            object_name=object_name,
            detector=detector,
            canonical=canonical,
            baseline_lap=baseline_lap,
            baseline_hand=baseline_hand,
            cfg=cfg,
            bootstrap=bootstrap,
        )
        if sc > best_score:
            best_score = sc
            best_r, best_l = right23, left23
    return best_r, best_l, best_score


def sim_refine_side_grasp(
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
) -> tuple[np.ndarray, np.ndarray, int, TpsrMetrics]:
    """Local sim refine anchored to δ* (Dexonomy stage-2 lite + DexGraspBench contact check)."""
    cfg = cfg or TpsrConfig()
    right23 = vec_to_arm_action(action_right)
    left23 = vec_to_arm_action(action_left)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    lo, hi = _hand_joint_bounds(raw_env._model, side)

    baseline_lap = laplacian_rmse(raw_env, canonical, object_name=object_name)
    baseline_hand = hand_rmse_obj_m(raw_env, canonical, side=side, object_name=object_name)
    sim0 = snapshot_sim(raw_env)
    sim_initial = sim0
    initial_contact = side_contact_count(detector, raw_env, object_name=object_name)
    initial_r, initial_l = right23, left23
    bootstrap = initial_contact < cfg.min_contact_count

    if bootstrap:
        right23, left23 = _cumulative_reach_object(
            raw_env,
            side=side,
            object_name=object_name,
            right23=right23,
            left23=left23,
            hold_right=hold_right,
            hold_left=hold_left,
            detector=detector,
            cfg=cfg,
        )
        sim0 = snapshot_sim(raw_env)
        initial_contact = side_contact_count(detector, raw_env, object_name=object_name)
        initial_r, initial_l = right23, left23

    best_r, best_l = right23, left23
    best_score = _score(
        raw_env,
        side=side,
        object_name=object_name,
        detector=detector,
        canonical=canonical,
        baseline_lap=baseline_lap,
        baseline_hand=baseline_hand,
        cfg=cfg,
        bootstrap=bootstrap,
    )

    if bootstrap:
        br, bl, bsc, bsnap = _bootstrap_mocap(
            raw_env,
            side=side,
            object_name=object_name,
            right23=right23,
            left23=left23,
            hold_right=hold_right,
            hold_left=hold_left,
            detector=detector,
            canonical=canonical,
            baseline_lap=baseline_lap,
            baseline_hand=baseline_hand,
            cfg=cfg,
            sim0=sim0,
        )
        if bsc > best_score:
            best_score = bsc
            best_r, best_l = br, bl
            sim0 = bsnap
        right23, left23 = best_r, best_l
        restore_sim(raw_env, sim0)

    # Physics hold from current arm pose (GenHand close in sim).
    active = _active23(side, right23, left23)
    x22 = _action23_to_x22(active)
    _physics_hold_active(
        raw_env,
        side=side,
        x22=x22,
        hold_right=hold_right,
        hold_left=hold_left,
        settle_steps=cfg.sim_settle_steps,
        hold_steps=cfg.sim_close_steps,
        detector=detector,
    )
    right23 = vec_to_arm_action(read_arm_action(raw_env, "right"))
    left23 = vec_to_arm_action(read_arm_action(raw_env, "left"))
    sc = _score(
        raw_env,
        side=side,
        object_name=object_name,
        detector=detector,
        canonical=canonical,
        baseline_lap=baseline_lap,
        baseline_hand=baseline_hand,
        cfg=cfg,
        bootstrap=bootstrap,
    )
    if sc > best_score:
        best_score = sc
        best_r, best_l = right23, left23
        sim0 = snapshot_sim(raw_env)

    right23, left23, sc_close = _progressive_close(
        raw_env,
        side=side,
        object_name=object_name,
        right23=right23,
        left23=left23,
        hold_right=hold_right,
        hold_left=hold_left,
        canonical=canonical,
        detector=detector,
        baseline_lap=baseline_lap,
        baseline_hand=baseline_hand,
        cfg=cfg,
        bootstrap=bootstrap,
    )
    if sc_close > best_score:
        best_score = sc_close
        best_r, best_l = right23, left23
        sim0 = snapshot_sim(raw_env)

    iters = 0
    for _ in range(int(cfg.sim_search_iters)):
        improved = False
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        active = _active23(side, right23, left23)
        for delta in _FINGER_DELTAS:
            updated = _close_fingers(active, delta, lo, hi)
            nr, nl = _set_active(side, right23, left23, updated)
            restore_sim(raw_env, sim0)
            _apply_pose(
                raw_env,
                side=side,
                right23=nr,
                left23=nl,
                hold_right=hold_right,
                hold_left=hold_left,
                hold_steps=cfg.hold_steps,
            )
            sc = _score(
                raw_env,
                side=side,
                object_name=object_name,
                detector=detector,
                canonical=canonical,
                baseline_lap=baseline_lap,
                baseline_hand=baseline_hand,
                cfg=cfg,
                bootstrap=bootstrap,
            )
            if sc > -1e5:
                candidates.append((sc, vec_to_arm_action(read_arm_action(raw_env, "right")), vec_to_arm_action(read_arm_action(raw_env, "left"))))
        if not cfg.finger_only:
            for step in _MOCAP_STEPS_M:
                updated = _nudge_mocap_toward_object(raw_env, active, object_name=object_name, step_m=step)
                nr, nl = _set_active(side, right23, left23, updated)
                restore_sim(raw_env, sim0)
                _apply_pose(
                    raw_env,
                    side=side,
                    right23=nr,
                    left23=nl,
                    hold_right=hold_right,
                    hold_left=hold_left,
                    hold_steps=cfg.hold_steps,
                )
                sc = _score(
                    raw_env,
                    side=side,
                    object_name=object_name,
                    detector=detector,
                    canonical=canonical,
                    baseline_lap=baseline_lap,
                    baseline_hand=baseline_hand,
                    cfg=cfg,
                    bootstrap=bootstrap,
                )
                if sc > -1e5:
                    candidates.append((sc, vec_to_arm_action(read_arm_action(raw_env, "right")), vec_to_arm_action(read_arm_action(raw_env, "left"))))
        if not candidates:
            break
        sc, nr, nl = max(candidates, key=lambda x: x[0])
        if sc > best_score + 0.5:
            best_score = sc
            best_r, best_l = nr, nl
            right23, left23 = nr, nl
            sim0 = snapshot_sim(raw_env)
            improved = True
            iters += 1
        if side_contact_count(detector, raw_env, object_name=object_name) >= cfg.min_contact_count and not improved:
            qp_pass, _ = _qp_ok(raw_env, side=side, object_name=object_name, cfg=cfg)
            if qp_pass or not cfg.require_qp_fc:
                break

    restore_sim(raw_env, sim0)
    _apply_pose(
        raw_env,
        side=side,
        right23=best_r,
        left23=best_l,
        hold_right=hold_right,
        hold_left=hold_left,
        hold_steps=max(cfg.hold_steps, 2),
    )
    metrics = tpsr_metrics(
        raw_env, canonical, object_name=object_name, side=side, detector=detector, cfg=cfg
    )
    if metrics.contact_count < initial_contact and initial_contact >= cfg.min_contact_count:
        restore_sim(raw_env, sim_initial)
        _apply_pose(
            raw_env,
            side=side,
            right23=initial_r,
            left23=initial_l,
            hold_right=hold_right,
            hold_left=hold_left,
            hold_steps=max(cfg.hold_steps, 2),
        )
        best_r, best_l = initial_r, initial_l
        metrics = tpsr_metrics(
            raw_env, canonical, object_name=object_name, side=side, detector=detector, cfg=cfg
        )
    best_r, best_l = _passive_from_hold(side, hold_right, hold_left, best_r, best_l)
    return best_r, best_l, iters, metrics
