"""DITTO-style demo approach warp (object-frame waypoints)."""

from __future__ import annotations

import numpy as np

from skill_graph.adapters.assembly import AssemblySim
from skill_graph.adapters.control import interpolate_arm23, interpolate_arm_only, read_arm23, step_side
from skill_graph.constants import APPROACH_LOOKBACK, ObjectName, Side
from skill_graph.math.se3 import arm23_from_object_frame
from skill_graph.skills.approach.base import ApproachBackend, ApproachResult
from skill_graph.skills.templates.schema import GraspTemplate

TABLE_Z_MIN = 0.012
BLEND_STEPS = 12
GRASP_SNAP_STEPS = 8
SQUEEZE_STEPS = 24
SUBSTEPS_PER_WAYPOINT_MIN = 3


def _substeps_for_segment(prev23: np.ndarray, tgt23: np.ndarray, *, n_wp: int) -> int:
    """Match demo pacing: ~APPROACH_LOOKBACK micro-steps spread across waypoints."""
    dist_m = float(np.linalg.norm(tgt23[:3] - prev23[:3]))
    from_dist = max(int(dist_m / 0.004), 1)  # ~4 mm per micro-step
    from_budget = max(APPROACH_LOOKBACK // max(n_wp, 1), SUBSTEPS_PER_WAYPOINT_MIN)
    return max(from_dist, from_budget, SUBSTEPS_PER_WAYPOINT_MIN)


def _blend_to_target(
    sim: AssemblySim,
    *,
    side: Side,
    prev23: np.ndarray,
    tgt23: np.ndarray,
    hand: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    blend_steps: int,
) -> tuple[int, np.ndarray]:
    steps = 0
    prev = np.asarray(prev23, dtype=np.float64).reshape(23)
    tgt = np.asarray(tgt23, dtype=np.float64).reshape(23)
    n = max(int(blend_steps), 1)
    for bi in range(n):
        alpha = (bi + 1) / n
        cmd = interpolate_arm_only(prev, tgt, alpha, hand=hand)
        step_side(sim, side=side, active23=cmd, hold_right=hold_right, hold_left=hold_left)
        steps += 1
    return steps, read_arm23(sim, side)


class DemoWarpApproach(ApproachBackend):
    name = "demo_warp"

    def execute(
        self,
        sim: AssemblySim,
        template: GraspTemplate,
        *,
        side: Side,
        object_name: ObjectName,
        hold_right: np.ndarray,
        hold_left: np.ndarray,
    ) -> ApproachResult:
        hold_right = np.asarray(hold_right, dtype=np.float64).reshape(23)
        hold_left = np.asarray(hold_left, dtype=np.float64).reshape(23)
        steps = 0
        n_wp = int(template.approach_mocap_pos_obj.shape[0])
        open_hand = template.approach_hand[0].copy()
        prev = read_arm23(sim, side)

        if n_wp >= 1:
            obj_pos, obj_quat = sim.object_pose(object_name)
            first = arm23_from_object_frame(
                template.approach_mocap_pos_obj[0],
                template.approach_mocap_quat_obj[0],
                open_hand,
                live_obj_pos=obj_pos,
                live_obj_quat=obj_quat,
            )
            blend_n = max(BLEND_STEPS, _substeps_for_segment(prev, first, n_wp=n_wp))
            n_blend, prev = _blend_to_target(
                sim,
                side=side,
                prev23=prev,
                tgt23=first,
                hand=open_hand,
                hold_right=hold_right,
                hold_left=hold_left,
                blend_steps=blend_n,
            )
            steps += n_blend

            for i in range(n_wp):
                obj_pos, obj_quat = sim.object_pose(object_name)
                tgt = arm23_from_object_frame(
                    template.approach_mocap_pos_obj[i],
                    template.approach_mocap_quat_obj[i],
                    open_hand,
                    live_obj_pos=obj_pos,
                    live_obj_quat=obj_quat,
                )
                if float(tgt[2]) < TABLE_Z_MIN:
                    tgt[2] = max(float(prev[2]), TABLE_Z_MIN)
                substeps = _substeps_for_segment(prev, tgt, n_wp=n_wp)
                for sub in range(substeps):
                    alpha = (sub + 1) / substeps
                    cmd = interpolate_arm_only(prev, tgt, alpha, hand=open_hand)
                    step_side(sim, side=side, active23=cmd, hold_right=hold_right, hold_left=hold_left)
                    steps += 1
                prev = read_arm23(sim, side)

        obj_pos, obj_quat = sim.object_pose(object_name)
        grasp23 = arm23_from_object_frame(
            template.grasp_mocap_pos_obj,
            template.grasp_mocap_quat_obj,
            template.grasp_hand,
            live_obj_pos=obj_pos,
            live_obj_quat=obj_quat,
        )
        for gi in range(GRASP_SNAP_STEPS):
            alpha = (gi + 1) / GRASP_SNAP_STEPS
            cmd = interpolate_arm23(prev, grasp23, alpha)
            step_side(sim, side=side, active23=cmd, hold_right=hold_right, hold_left=hold_left)
            steps += 1
        prev = read_arm23(sim, side)

        squeeze23 = arm23_from_object_frame(
            template.squeeze_mocap_pos_obj,
            template.squeeze_mocap_quat_obj,
            template.squeeze_hand,
            live_obj_pos=obj_pos,
            live_obj_quat=obj_quat,
        )
        for si in range(SQUEEZE_STEPS):
            alpha = (si + 1) / SQUEEZE_STEPS
            cmd = interpolate_arm23(prev, squeeze23, alpha)
            step_side(sim, side=side, active23=cmd, hold_right=hold_right, hold_left=hold_left)
            steps += 1

        return ApproachResult(success=True, steps=steps)
