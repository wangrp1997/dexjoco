"""End-effector / contact audit before delivering skill_replay videos."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from interaction_retarget.constants import MIN_GRASP_CONTACT_COUNT, PEG_BODY, TRAY_BODY
from interaction_retarget.grasp.repair import side_contact_count
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.settle import read_arm_action

# dexquery eval: env_step >= 1500 → truncate; ego.mp4 ≈ 50.03s @ 30fps
EVAL_FPS = 30
EVAL_MAX_FRAMES = 1500
EVAL_MAX_DURATION_S = EVAL_MAX_FRAMES / EVAL_FPS


@dataclass
class EefSample:
    phase: str
    step: int
    left_mocap: np.ndarray
    right_mocap: np.ndarray
    tray_z: float
    peg_z: float
    tray_contact: int
    peg_contact: int


@dataclass
class TrajectoryAudit:
    sim_steps: int = 0
    video_frames: int = 0
    phases: list[EefSample] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def ok_for_delivery(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"sim_steps={self.sim_steps} video_frames={self.video_frames} "
            f"video_duration={self.video_frames / EVAL_FPS:.1f}s",
        ]
        for ph in self.phases:
            lz, rz = float(ph.left_mocap[2]), float(ph.right_mocap[2])
            lines.append(
                f"  [{ph.phase}] step={ph.step} "
                f"left_z={lz:.3f} right_z={rz:.3f} "
                f"tray_c={ph.tray_contact} peg_c={ph.peg_contact} "
                f"tray_obj_z={ph.tray_z:.3f} peg_obj_z={ph.peg_z:.3f}"
            )
        for err in self.errors:
            lines.append(f"  FAIL: {err}")
        return "\n".join(lines)


def _body_z(raw_env, name: str) -> float:
    bid = int(raw_env._model.body(name).id)
    return float(raw_env._data.xpos[bid, 2])


def snapshot_phase(
    raw_env,
    detector: AssemblyContactDetector,
    *,
    phase: str,
    step: int,
) -> EefSample:
    left = np.asarray(read_arm_action(raw_env, "left")[0:3], dtype=np.float64)
    right = np.asarray(read_arm_action(raw_env, "right")[0:3], dtype=np.float64)
    return EefSample(
        phase=str(phase),
        step=int(step),
        left_mocap=left,
        right_mocap=right,
        tray_z=_body_z(raw_env, TRAY_BODY),
        peg_z=_body_z(raw_env, PEG_BODY),
        tray_contact=int(side_contact_count(detector, raw_env, object_name="tray")),
        peg_contact=int(side_contact_count(detector, raw_env, object_name="peg")),
    )


MIN_TRAY_LIFT_M = 0.030
MIN_PEG_LIFT_M = 0.010


def validate_delivery(
    audit: TrajectoryAudit,
    *,
    table_z: float,
    peg_rest_z: float | None = None,
    min_table_clearance_m: float = 0.04,
    min_tray_lift_m: float = MIN_TRAY_LIFT_M,
    min_peg_lift_m: float = MIN_PEG_LIFT_M,
    require_tray_contact: int = MIN_GRASP_CONTACT_COUNT,
    require_peg_contact: int = MIN_GRASP_CONTACT_COUNT,
    require_both_lifts: bool = True,
) -> TrajectoryAudit:
    if audit.video_frames > EVAL_MAX_FRAMES:
        audit.errors.append(
            f"video_too_long: {audit.video_frames} frames "
            f"({audit.video_frames / EVAL_FPS:.1f}s > {EVAL_MAX_DURATION_S:.1f}s)"
        )
    if audit.sim_steps > EVAL_MAX_FRAMES:
        audit.errors.append(
            f"sim_steps_too_many: {audit.sim_steps} > {EVAL_MAX_FRAMES} "
            f"(dexquery cap)"
        )

    for ph in audit.phases:
        if ph.phase.endswith("_grasp") or ph.phase.endswith("_lift"):
            side = "left" if "tray" in ph.phase else "right"
            mocap_z = float(ph.left_mocap[2] if side == "left" else ph.right_mocap[2])
            if mocap_z < table_z + min_table_clearance_m:
                audit.errors.append(
                    f"{ph.phase}: eef_z={mocap_z:.3f} < table+{min_table_clearance_m:.2f}"
                )
        if ph.phase == "tray_grasp_done" and ph.tray_contact < require_tray_contact:
            audit.errors.append(
                f"tray_grasp_no_contact: {ph.tray_contact} < {require_tray_contact}"
            )
        if ph.phase == "peg_grasp_done" and ph.peg_contact < require_peg_contact:
            audit.errors.append(
                f"peg_grasp_no_contact: {ph.peg_contact} < {require_peg_contact}"
            )
        if ph.phase == "tray_lift_done":
            tray_lift = float(ph.tray_z) - float(table_z)
            if tray_lift < min_tray_lift_m:
                audit.errors.append(
                    f"tray_not_lifted: dz={tray_lift:.3f}m < {min_tray_lift_m:.3f}m"
                )
        if ph.phase == "peg_lift_done" and peg_rest_z is not None:
            peg_lift = float(ph.peg_z) - float(peg_rest_z)
            if peg_lift < min_peg_lift_m:
                audit.errors.append(
                    f"peg_not_lifted: dz={peg_lift:.3f}m < {min_peg_lift_m:.3f}m"
                )

    if require_both_lifts:
        phases = {ph.phase for ph in audit.phases}
        for required in ("tray_lift_done", "peg_lift_done"):
            if required not in phases:
                audit.errors.append(f"missing_phase: {required}")
    return audit
