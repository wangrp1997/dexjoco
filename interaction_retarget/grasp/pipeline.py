"""Random object-pose grasp pipeline (phase-1 inference, no demo lookup)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from interaction_retarget.constants import CONTACT_WINDOW, MIN_GRASP_CONTACT_COUNT
from interaction_retarget.grasp.agent import make_peg_agent, make_tray_agent
from interaction_retarget.grasp.distill import load_canonical_grasp
from interaction_retarget.grasp.ik import GraspIkResult
from interaction_retarget.grasp.lift import (
    DEFAULT_TRAY_LIFT_M,
    execute_peg_lift,
    execute_tray_lift,
    hold_tray_before_peg,
    load_lift_reference,
)
from interaction_retarget.grasp.repair import (
    GraspRepairResult,
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

GraspSource = Literal["random"]


def _ik_grasp_ready(ik: GraspIkResult) -> bool:
    return bool(ik.success and ik.contact_count >= MIN_GRASP_CONTACT_COUNT)


@dataclass
class RandomGraspReport:
    seed: int
    source: GraspSource
    tray_ik: GraspIkResult
    peg_ik: GraspIkResult
    repair: GraspRepairResult
    tray_laplacian_rmse_m: float
    peg_laplacian_rmse_m: float
    tray_reach: dict[str, Any] = field(default_factory=dict)
    peg_reach: dict[str, Any] = field(default_factory=dict)
    tray_lift_hold_stable: bool | None = None
    success: bool = False


def run_random_grasp(
    *,
    sidecar_dir: Path,
    seed: int = 0,
    hold_steps: int = 16,
    tray_lift_height_m: float = DEFAULT_TRAY_LIFT_M,
    tray_lift_steps: int | None = None,
    skip_tray_lift: bool = False,
    skip_tray_hold: bool = False,
    skip_peg_lift: bool = False,
    peg_lift_steps: int | None = None,
    tray_hold_max_steps: int = 72,
    hold_warmup_steps: int = 10,
    fast: bool = False,
) -> RandomGraspReport:
    """Demo order: left/tray grasp → tray lift → hold → right/peg grasp → peg lift."""
    sidecar_dir = Path(sidecar_dir)
    canonical_tray = load_canonical_grasp(sidecar_dir / "canonical_tray_grasp.npz")
    canonical_peg = load_canonical_grasp(sidecar_dir / "canonical_peg_grasp.npz")
    lift_ref = load_lift_reference(sidecar_dir)
    left_agent = make_tray_agent(canonical_tray, fast=fast)
    right_agent = make_peg_agent(canonical_peg, fast=fast)
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
        right_hold, left_hold, n, tray_reach = left_agent.execute(
            raw,
            target23=left_target,
            hold_right=right_hold,
            hold_left=left_hold,
            detector=detector,
            ik=tray_ik,
            skip_approach=tray_ik_ready,
            skip_repair=tray_ik_ready,
            finger_repair_only=False,
        )
        repair_iters += n
        if tray_ik_ready and side_contact_count(detector, raw, object_name="tray") < MIN_GRASP_CONTACT_COUNT:
            right_hold, left_hold, n = repair_side_grasp(
                raw,
                side="left",
                object_name="tray",
                action_right=right_hold,
                action_left=left_hold,
                hold_right=right_hold,
                hold_left=left_hold,
                detector=detector,
                max_iters=8,
                canonical=canonical_tray,
                require_on_table=False,
                finger_only=False,
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
                left_hold, tray_lift_hold_stable, _ = hold_tray_before_peg(
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
                    left_hold, tray_lift_hold_stable, _ = hold_tray_before_peg(
                        raw,
                        left_hold=left_hold,
                        right_home=right_home,
                        detector=detector,
                        lift_ref=lift_ref,
                        hold_steps=max(int(CONTACT_WINDOW) + 4, 16),
                        max_hold_steps=tray_hold_max_steps,
                        warmup_steps=hold_warmup_steps,
                    )

        right_hold = right_home.copy()
        left_hold = vec_to_arm_action(read_arm_action(raw, "left"))
        peg_after_lift = not skip_tray_lift
        if peg_after_lift:
            settle_bimanual_actions(raw, right23=right_hold, left23=left_hold, n_substeps=4)
        sim_pre_peg = snapshot_sim(raw)
        right_target, peg_ik = right_agent.plan(
            raw,
            hold_right=right_hold,
            hold_left=left_hold,
            restore_env=False,
            detector=detector,
        )
        peg_ik_ready = _ik_grasp_ready(peg_ik)
        peg_direct: int | None = None
        if not peg_ik_ready:
            if peg_ik.laplacian_rmse_m <= 0.035:
                peg_direct = 60
            else:
                restore_sim(raw, sim_pre_peg)
                right_target, peg_ik = right_agent.plan(
                    raw,
                    hold_right=right_hold,
                    hold_left=left_hold,
                    restore_env=True,
                    detector=detector,
                )
                peg_ik_ready = _ik_grasp_ready(peg_ik)
                if not peg_ik_ready:
                    peg_direct = 80
        right_hold, left_hold, n, peg_reach = right_agent.execute(
            raw,
            target23=right_target,
            hold_right=right_hold,
            hold_left=left_hold,
            detector=detector,
            ik=peg_ik,
            skip_approach=peg_ik_ready,
            skip_repair=peg_ik_ready,
            direct_reach_steps=peg_direct,
        )
        repair_iters += n

        if not skip_tray_lift and peg_reach.get("contact_count", 0) < MIN_GRASP_CONTACT_COUNT:
            right_hold, left_hold, n = repair_side_grasp(
                raw,
                side="right",
                object_name="peg",
                action_right=right_hold,
                action_left=left_hold,
                hold_right=right_hold,
                hold_left=left_hold,
                detector=detector,
                max_iters=12,
                canonical=canonical_peg,
                require_on_table=False,
                finger_only=False,
            )
            repair_iters += n

        if not skip_peg_lift:
            right_hold = execute_peg_lift(
                raw,
                grasp_right=right_hold,
                hold_left=left_hold,
                lift_ref=lift_ref,
                detector=detector,
                steps=peg_lift_steps,
            )

        if not skip_tray_lift:
            left_hold = prepare_lift_grasp(
                raw,
                side="left",
                grasp_arm=left_hold,
                hold_other=right_hold,
                detector=detector,
                object_name="tray",
                finger_deltas=(-0.004, -0.008, -0.012),
                settle_steps=4,
            )
            left_hold = vec_to_arm_action(read_arm_action(raw, "left"))
            right_hold = vec_to_arm_action(read_arm_action(raw, "right"))

        repair = verify_grasp_hold(
            raw,
            detector,
            action_right=right_hold,
            action_left=left_hold,
            hold_steps=hold_steps,
            warmup_steps=hold_warmup_steps,
            tray_lifted=not skip_tray_lift,
        )
        repair.repair_iters = repair_iters

        return RandomGraspReport(
            seed=int(seed),
            source="random",
            tray_ik=tray_ik,
            peg_ik=peg_ik,
            repair=repair,
            tray_laplacian_rmse_m=laplacian_rmse(raw, canonical_tray, object_name="tray"),
            peg_laplacian_rmse_m=laplacian_rmse(raw, canonical_peg, object_name="peg"),
            tray_reach=tray_reach,
            peg_reach=peg_reach,
            tray_lift_hold_stable=tray_lift_hold_stable,
            success=repair.success,
        )
    finally:
        env.close()
