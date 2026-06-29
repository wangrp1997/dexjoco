"""Random grasp pipeline with TPSR refine + bench verification."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from interaction_retarget.bench.config import BenchConfig
from interaction_retarget.bench.lift_verify import LiftVerifyConfig, TrayGraspLiftReport, verify_tray_grasp_lift
from interaction_retarget.bench.verify import BenchHoldReport, verify_side_hold
from interaction_retarget.constants import CONTACT_WINDOW, MIN_GRASP_CONTACT_COUNT, TRAY_BODY
from interaction_retarget.grasp.agent_tpsr import make_peg_agent_tpsr, make_tray_agent_tpsr
from interaction_retarget.grasp.distill import load_canonical_grasp
from interaction_retarget.grasp.ik import GraspIkResult, mocap_world_from_canonical
from interaction_retarget.grasp.lift import (
    DEFAULT_TRAY_LIFT_M,
    execute_peg_lift,
    execute_tray_lift,
    hold_tray_before_peg,
    load_lift_reference,
)
from interaction_retarget.grasp.locked_hold import enforce_locked_passive
from interaction_retarget.grasp.pipeline import _ik_grasp_ready
from interaction_retarget.grasp.repair import (
    GraspRepairResult,
    SideContactMetrics,
    laplacian_rmse,
    prepare_lift_grasp,
    repair_side_grasp,
    side_contact_count,
    verify_grasp_hold,
)
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.replay import make_assembly_env
from interaction_retarget.sim.settle import read_arm_action, settle_bimanual_actions, vec_to_arm_action
from interaction_retarget.sim.state import restore_sim, snapshot_sim
from interaction_retarget.tpsr.config import TpsrConfig
from interaction_retarget.tpsr.refine import refine_side_grasp

def _canonical_arm_target(raw_env, canonical: dict, object_name: Literal["tray", "peg"]) -> np.ndarray:
    pos_w, quat_w = mocap_world_from_canonical(raw_env, canonical, object_name=object_name)
    hand = np.asarray(
        canonical.get("hand_joint_median", np.zeros(16, dtype=np.float64)),
        dtype=np.float64,
    ).reshape(16)
    return np.concatenate([pos_w, quat_w, hand], axis=0)


def _bimanual_fail_reason(
    *,
    tray_contact: int,
    peg_contact: int,
    tray_min: int,
    hold_contact_min: int,
    bench_tray: BenchHoldReport | None,
    bench_peg: BenchHoldReport | None,
    tray_lift_hold_stable: bool | None,
    skip_tray_lift: bool,
) -> str:
    if not skip_tray_lift and tray_lift_hold_stable is False:
        return "tray_lift_hold_unstable"
    if not skip_tray_lift and hold_contact_min < tray_min and tray_contact < tray_min:
        return f"tray_contact={max(tray_contact, hold_contact_min)}<{tray_min}"
    if bench_tray is not None and not bench_tray.stable:
        return "bench_tray_unstable"
    if peg_contact < MIN_GRASP_CONTACT_COUNT:
        return f"peg_contact={peg_contact}<{MIN_GRASP_CONTACT_COUNT}"
    if bench_peg is not None and not bench_peg.stable:
        return "bench_peg_unstable"
    return "ok"


GraspSource = Literal["random_tpsr"]
ValidationStage = Literal["tray_lift", "bimanual_grasp"]


def _skipped_peg_ik() -> GraspIkResult:
    z = np.zeros(23, dtype=np.float64)
    return GraspIkResult(
        object_name="peg",
        active_side="right",
        action_right=z,
        action_left=z,
        cost=float("inf"),
        laplacian_rmse_m=float("nan"),
        hand_rmse_m=float("nan"),
        contact_count=0,
        contact_site_rmse_m=float("nan"),
        success=False,
    )


@dataclass
class TpsrGraspReport:
    seed: int
    tray_ik: GraspIkResult
    peg_ik: GraspIkResult
    repair: GraspRepairResult
    tray_laplacian_rmse_m: float
    peg_laplacian_rmse_m: float
    tray_reach: dict[str, Any] = field(default_factory=dict)
    peg_reach: dict[str, Any] = field(default_factory=dict)
    tray_lift_hold_stable: bool | None = None
    success: bool = False
    bench_tray: BenchHoldReport | None = None
    bench_peg: BenchHoldReport | None = None
    tray_grasp_lift: TrayGraspLiftReport | None = None
    source: GraspSource = "random_tpsr"
    fail_reason: str = ""
    stage: ValidationStage = "tray_lift"


def run_random_grasp_tpsr(
    *,
    sidecar_dir: Path,
    seed: int = 0,
    hold_steps: int = 20,
    tray_lift_height_m: float = DEFAULT_TRAY_LIFT_M,
    tray_lift_steps: int | None = None,
    skip_tray_lift: bool = False,
    skip_tray_hold: bool = False,
    skip_peg_lift: bool = False,
    peg_lift_steps: int | None = None,
    tray_hold_max_steps: int = 72,
    hold_warmup_steps: int = 10,
    fast: bool = False,
    tpsr_cfg: TpsrConfig | None = None,
    bench_cfg: BenchConfig | None = None,
    stage: ValidationStage = "tray_lift",
) -> TpsrGraspReport:
    """Phase-1 validation stages (no insert / hybrid_insert).

    - ``tray_lift`` (default): left tray grasp → lift → hold → verify tray only.
    - ``bimanual_grasp``: above + right peg grasp (+ optional peg lift); still no insert.
    """
    skip_peg_grasp = stage == "tray_lift"
    sidecar_dir = Path(sidecar_dir)
    tpsr_cfg = tpsr_cfg or TpsrConfig()
    bench_cfg = bench_cfg or BenchConfig(hold_steps=hold_steps, warmup_steps=hold_warmup_steps)
    canonical_tray = load_canonical_grasp(sidecar_dir / "canonical_tray_grasp.npz")
    canonical_peg = load_canonical_grasp(sidecar_dir / "canonical_peg_grasp.npz")
    lift_ref = load_lift_reference(sidecar_dir)
    left_agent = make_tray_agent_tpsr(canonical_tray, fast=fast)
    left_agent.tpsr_cfg = tpsr_cfg
    right_agent = make_peg_agent_tpsr(canonical_peg, fast=fast)
    right_agent.tpsr_cfg = tpsr_cfg

    if fast:
        tray_hold_max_steps = min(tray_hold_max_steps, 16)
        hold_steps = min(hold_steps, 8)
        hold_warmup_steps = min(hold_warmup_steps, 4)
        lift_exec_cap = 18
        lift_pre_settle = 4
    else:
        lift_exec_cap = None
        lift_pre_settle = 12

    env = make_assembly_env(seed=int(seed), randomize=False)
    raw = env.unwrapped
    detector = AssemblyContactDetector(raw)
    tray_lift_hold_stable: bool | None = None
    hold_contact_min = 0
    tray_grasp_lift: TrayGraspLiftReport | None = None
    bench_tray: BenchHoldReport | None = None
    try:
        env.reset()
        detector.reset_reference(raw)

        right_home = vec_to_arm_action(read_arm_action(raw, "right"))
        left_home = vec_to_arm_action(read_arm_action(raw, "left"))
        right_hold = right_home.copy()
        left_hold = left_home.copy()
        repair_iters = 0

        sim_pre_tray = snapshot_sim(raw)
        left_target, tray_ik = left_agent.plan(
            raw, hold_right=right_hold, hold_left=left_hold, detector=detector, restore_env=False
        )
        tray_ik_ready = _ik_grasp_ready(tray_ik)
        tray_skip_repair = tray_ik_ready
        tray_direct: int | None = None
        if not tray_ik_ready:
            restore_sim(raw, sim_pre_tray)
            left_target, tray_ik = left_agent.plan(
                raw,
                hold_right=right_hold,
                hold_left=left_hold,
                restore_env=True,
                detector=detector,
            )
            tray_ik_ready = _ik_grasp_ready(tray_ik)
        if not tray_ik_ready:
            if tray_ik.contact_count <= 0 or tray_ik.laplacian_rmse_m <= 0.045:
                tray_direct = 80 if tray_ik.contact_count <= 0 else 50
                if tray_ik.contact_count <= 0:
                    pos_w, quat_w = mocap_world_from_canonical(
                        raw, canonical_tray, object_name="tray"
                    )
                    hand = np.asarray(
                        canonical_tray.get("hand_joint_median", np.zeros(16, dtype=np.float64)),
                        dtype=np.float64,
                    ).reshape(16)
                    left_target = np.concatenate([pos_w, quat_w, hand], axis=0)
                    tray_skip_repair = False

        restore_sim(raw, sim_pre_tray)
        right_hold, left_hold, n, tray_reach = left_agent.execute(
            raw,
            target23=left_target,
            hold_right=right_hold,
            hold_left=left_hold,
            detector=detector,
            ik=tray_ik,
            skip_approach=tray_ik_ready,
            skip_repair=tray_skip_repair,
            direct_reach_steps=tray_direct,
        )
        repair_iters += n

        if side_contact_count(detector, raw, object_name="tray") < MIN_GRASP_CONTACT_COUNT:
            right_hold, left_hold, n, _ = refine_side_grasp(
                raw,
                side="left",
                object_name="tray",
                action_right=right_hold,
                action_left=left_hold,
                hold_right=right_hold,
                hold_left=left_hold,
                detector=detector,
                canonical=canonical_tray,
                cfg=tpsr_cfg,
            )
            repair_iters += n
            right_hold = vec_to_arm_action(read_arm_action(raw, "right"))
            left_hold = vec_to_arm_action(read_arm_action(raw, "left"))

        if not skip_tray_lift:
            left_hold = execute_tray_lift(
                raw,
                grasp_left=left_hold,
                hold_right=right_home,
                lift_ref=lift_ref,
                detector=detector,
                lift_height_m=tray_lift_height_m,
                steps=tray_lift_steps,
                hold_steps=4 if fast else 8,
                pre_lift_settle=lift_pre_settle,
                lift_exec_cap=lift_exec_cap,
            )
            if not skip_tray_hold:
                left_hold, tray_lift_hold_stable, hold_contact_min = hold_tray_before_peg(
                    raw,
                    left_hold=left_hold,
                    right_home=right_home,
                    detector=detector,
                    lift_ref=lift_ref,
                    max_hold_steps=tray_hold_max_steps,
                    warmup_steps=hold_warmup_steps,
                )
                if tray_lift_hold_stable is False:
                    left_hold = prepare_lift_grasp(
                        raw,
                        side="left",
                        grasp_arm=left_hold,
                        hold_other=right_home,
                        detector=detector,
                        object_name="tray",
                        settle_steps=8,
                    )
                    left_hold, tray_lift_hold_stable, hold_contact_min = hold_tray_before_peg(
                        raw,
                        left_hold=left_hold,
                        right_home=right_home,
                        detector=detector,
                        lift_ref=lift_ref,
                        hold_steps=max(int(CONTACT_WINDOW) + 4, 16),
                        max_hold_steps=tray_hold_max_steps,
                        warmup_steps=hold_warmup_steps,
                    )

        left_locked: np.ndarray | None = None
        if not skip_tray_lift and not skip_tray_hold:
            left_locked = vec_to_arm_action(left_hold)

        tray_grasp_lift: TrayGraspLiftReport | None = None
        if skip_peg_grasp and not skip_tray_lift:
            tray_grasp_lift = verify_tray_grasp_lift(
                raw,
                detector=detector,
                canonical_tray=canonical_tray,
                lift_ref=lift_ref,
                hold_contact_min=hold_contact_min,
                cfg=LiftVerifyConfig(),
                default_lift_height_m=tray_lift_height_m,
                hold_stable=tray_lift_hold_stable,
            )

        peg_ik = _skipped_peg_ik()
        peg_reach: dict[str, Any] = {}
        tray_contact_pre_peg: int | None = None
        if not skip_peg_grasp:
            peg_tpsr_cfg = replace(tpsr_cfg, require_qp_fc=False)
            right_agent.tpsr_cfg = peg_tpsr_cfg

            if left_locked is None:
                left_locked = vec_to_arm_action(read_arm_action(raw, "left"))
            right_hold = right_home.copy()
            left_hold = left_locked.copy()
            settle_bimanual_actions(raw, right23=right_hold, left23=left_hold, n_substeps=4)
            tray_contact_pre_peg = int(side_contact_count(detector, raw, object_name="tray"))
            sim_pre_peg = snapshot_sim(raw)

            peg_ik_ready = False
            peg_direct: int | None = None
            restore_sim(raw, sim_pre_peg)
            right_target, peg_ik = right_agent.plan(
                raw,
                hold_right=right_hold,
                hold_left=left_hold,
                restore_env=True,
                detector=detector,
            )
            peg_ik_ready = _ik_grasp_ready(peg_ik)
            if peg_ik_ready and peg_ik.contact_count >= MIN_GRASP_CONTACT_COUNT:
                peg_direct = None
            else:
                restore_sim(raw, sim_pre_peg)
                if peg_ik.contact_count > 0:
                    right_target = vec_to_arm_action(peg_ik.action_right)
                    peg_direct = 70
                else:
                    right_target = _canonical_arm_target(raw, canonical_peg, "peg")
                    peg_direct = 100
                peg_ik_ready = False

            right_hold, _, n, peg_reach = right_agent.execute(
                raw,
                target23=right_target,
                hold_right=right_hold,
                hold_left=left_locked,
                detector=detector,
                ik=peg_ik,
                skip_approach=peg_ik_ready,
                skip_repair=False,
                direct_reach_steps=peg_direct,
            )
            repair_iters += n
            left_hold = left_locked.copy()
            enforce_locked_passive(raw, locked_left=left_locked, locked_right=right_hold)

            if peg_reach.get("contact_count", 0) < MIN_GRASP_CONTACT_COUNT:
                right_hold, _, n, _ = refine_side_grasp(
                    raw,
                    side="right",
                    object_name="peg",
                    action_right=right_hold,
                    action_left=left_locked,
                    hold_right=right_hold,
                    hold_left=left_locked,
                    detector=detector,
                    canonical=canonical_peg,
                    cfg=peg_tpsr_cfg,
                )
                repair_iters += n
                left_hold = left_locked.copy()
                enforce_locked_passive(raw, locked_left=left_locked, locked_right=right_hold)

            if not skip_peg_lift:
                right_hold = execute_peg_lift(
                    raw,
                    grasp_right=right_hold,
                    hold_left=left_locked,
                    lift_ref=lift_ref,
                    detector=detector,
                    steps=peg_lift_steps,
                )
                left_hold = left_locked.copy()
                enforce_locked_passive(raw, locked_left=left_locked, locked_right=right_hold)

            right_agent.tpsr_cfg = tpsr_cfg

        if skip_peg_grasp and not skip_tray_lift and tray_grasp_lift is not None:
            tray_c = int(tray_grasp_lift.grasp_contact)
            tray_id = raw._model.body(TRAY_BODY).id
            tray_z = float(raw._data.xpos[tray_id, 2])
            rest_z = float(detector._tray_rest_z)  # noqa: SLF001
            repair = GraspRepairResult(
                action_right=right_hold,
                action_left=left_hold,
                tray=SideContactMetrics(
                    side="left",
                    object_name="tray",
                    contact_count=tray_c,
                    has_contact=tray_c >= MIN_GRASP_CONTACT_COUNT,
                    object_z=tray_z,
                    object_rest_z=rest_z,
                    on_table=tray_z >= rest_z - 0.015,
                ),
                peg=SideContactMetrics(
                    side="right",
                    object_name="peg",
                    contact_count=0,
                    has_contact=False,
                    object_z=0.0,
                    object_rest_z=0.0,
                    on_table=True,
                ),
                repair_iters=repair_iters,
                hold_steps=0,
                stable_tray=bool(tray_lift_hold_stable),
                stable_peg=False,
                success=bool(tray_grasp_lift.success),
            )
            bench_tray = None
        elif skip_peg_grasp:
            repair = verify_grasp_hold(
                raw,
                detector,
                action_right=right_hold,
                action_left=left_hold,
                hold_steps=hold_steps,
                warmup_steps=hold_warmup_steps,
                tray_lifted=not skip_tray_lift,
                adjust_right=False,
                require_peg=False,
            )
            repair.repair_iters = repair_iters
            bench_tray = verify_side_hold(
                raw,
                object_name="tray",
                action_right=right_hold,
                action_left=left_hold,
                detector=detector,
                bench_cfg=bench_cfg,
                tpsr_cfg=tpsr_cfg,
            )
        else:
            left_hold = left_locked.copy() if left_locked is not None else left_hold
            enforce_locked_passive(
                raw,
                locked_left=left_hold,
                locked_right=right_hold,
                n_substeps=12,
            )
            repair = verify_grasp_hold(
                raw,
                detector,
                action_right=right_hold,
                action_left=left_hold,
                hold_steps=hold_steps,
                warmup_steps=hold_warmup_steps,
                tray_lifted=not skip_tray_lift,
                adjust_left=False,
                adjust_right=False,
                require_tray=True,
                require_peg=True,
            )
            repair.repair_iters = repair_iters
            if bench_tray is None:
                bench_tray = verify_side_hold(
                    raw,
                    object_name="tray",
                    action_right=right_hold,
                    action_left=left_hold,
                    detector=detector,
                    bench_cfg=bench_cfg,
                    tpsr_cfg=tpsr_cfg,
                )

        bench_peg: BenchHoldReport | None = None
        if not skip_peg_grasp:
            bench_peg = verify_side_hold(
                raw,
                object_name="peg",
                action_right=right_hold,
                action_left=left_hold,
                detector=detector,
                bench_cfg=bench_cfg,
                tpsr_cfg=tpsr_cfg,
            )

        tray_min = max(2, MIN_GRASP_CONTACT_COUNT - 1) if not skip_tray_lift else MIN_GRASP_CONTACT_COUNT
        tray_contact = int(side_contact_count(detector, raw, object_name="tray"))
        peg_contact = int(side_contact_count(detector, raw, object_name="peg"))
        fail_reason = ""

        if skip_peg_grasp and tray_grasp_lift is not None:
            success = bool(tray_grasp_lift.success and tray_lift_hold_stable is not False)
            fail_reason = "ok" if success else "tray_lift_L0_fail"
        elif skip_peg_grasp:
            success = bool(repair.success)
            fail_reason = "ok" if success else "tray_hold_fail"
        else:
            tray_hold_ok = (
                tray_lift_hold_stable is not False
                if not skip_tray_lift and not skip_tray_hold
                else True
            )
            tray_contact_ok = (
                tray_contact >= tray_min
                if skip_tray_lift
                else (hold_contact_min >= tray_min or tray_contact_pre_peg >= tray_min)
            )
            success = bool(
                tray_hold_ok
                and tray_contact_ok
                and peg_contact >= MIN_GRASP_CONTACT_COUNT
                and bench_peg is not None
                and bench_peg.stable
            )
            fail_reason = _bimanual_fail_reason(
                tray_contact=tray_contact,
                peg_contact=peg_contact,
                tray_min=tray_min,
                hold_contact_min=hold_contact_min,
                bench_tray=bench_tray,
                bench_peg=bench_peg,
                tray_lift_hold_stable=tray_lift_hold_stable,
                skip_tray_lift=skip_tray_lift,
            )
            if success:
                fail_reason = "ok"

        return TpsrGraspReport(
            seed=int(seed),
            source="random_tpsr",
            stage=stage,
            fail_reason=fail_reason,
            tray_ik=tray_ik,
            peg_ik=peg_ik,
            repair=repair,
            tray_laplacian_rmse_m=laplacian_rmse(raw, canonical_tray, object_name="tray"),
            peg_laplacian_rmse_m=(
                laplacian_rmse(raw, canonical_peg, object_name="peg")
                if not skip_peg_grasp
                else float("nan")
            ),
            tray_reach=tray_reach,
            peg_reach=peg_reach,
            tray_lift_hold_stable=tray_lift_hold_stable,
            success=success,
            bench_tray=bench_tray,
            bench_peg=bench_peg,
            tray_grasp_lift=tray_grasp_lift,
        )
    finally:
        env.close()
