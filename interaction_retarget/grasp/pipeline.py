"""Random object-pose grasp pipeline (phase-1 inference, no demo lookup)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from interaction_retarget.grasp.agent import make_peg_agent, make_tray_agent
from interaction_retarget.grasp.distill import load_canonical_grasp
from interaction_retarget.grasp.ik import GraspIkResult
from interaction_retarget.grasp.lift import DEFAULT_TRAY_LIFT_M, execute_tray_lift
from interaction_retarget.grasp.repair import (
    GraspRepairResult,
    laplacian_rmse,
    verify_grasp_hold,
)
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.replay import make_assembly_env
from interaction_retarget.sim.settle import read_arm_action, vec_to_arm_action

GraspSource = Literal["random"]


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
    success: bool = False


def run_random_grasp(
    *,
    sidecar_dir: Path,
    seed: int = 0,
    hold_steps: int = 16,
    tray_lift_height_m: float = DEFAULT_TRAY_LIFT_M,
    tray_lift_steps: int = 20,
    skip_tray_lift: bool = False,
    hold_warmup_steps: int = 10,
) -> RandomGraspReport:
    """Async two-agent grasp: left/tray → [lift] → right/peg (per-arm side configs)."""
    sidecar_dir = Path(sidecar_dir)
    canonical_tray = load_canonical_grasp(sidecar_dir / "canonical_tray_grasp.npz")
    canonical_peg = load_canonical_grasp(sidecar_dir / "canonical_peg_grasp.npz")
    left_agent = make_tray_agent(canonical_tray)
    right_agent = make_peg_agent(canonical_peg)

    env = make_assembly_env(seed=int(seed), randomize=False)
    raw = env.unwrapped
    detector = AssemblyContactDetector(raw)
    try:
        env.reset()
        detector.reset_reference(raw)

        right_hold = vec_to_arm_action(read_arm_action(raw, "right"))
        left_hold = vec_to_arm_action(read_arm_action(raw, "left"))
        repair_iters = 0

        left_target, tray_ik = left_agent.plan(raw, hold_right=right_hold, hold_left=left_hold)
        right_hold, left_hold, n, tray_reach = left_agent.execute(
            raw,
            target23=left_target,
            hold_right=right_hold,
            hold_left=left_hold,
            detector=detector,
            ik=tray_ik,
        )
        repair_iters += n

        if not skip_tray_lift:
            left_hold = execute_tray_lift(
                raw,
                grasp_left=left_hold,
                hold_right=right_hold,
                lift_height_m=tray_lift_height_m,
                steps=tray_lift_steps,
            )

        right_hold = vec_to_arm_action(read_arm_action(raw, "right"))
        left_hold = vec_to_arm_action(read_arm_action(raw, "left"))
        right_target, peg_ik = right_agent.plan(
            raw,
            hold_right=right_hold,
            hold_left=left_hold,
            restore_env=False,
            detector=detector,
        )
        right_hold, left_hold, n, peg_reach = right_agent.execute(
            raw,
            target23=right_target,
            hold_right=right_hold,
            hold_left=left_hold,
            detector=detector,
            ik=peg_ik,
            skip_approach=True,
        )
        repair_iters += n

        repair = verify_grasp_hold(
            raw,
            detector,
            action_right=right_hold,
            action_left=left_hold,
            hold_steps=hold_steps,
            warmup_steps=hold_warmup_steps,
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
            success=repair.success,
        )
    finally:
        env.close()
