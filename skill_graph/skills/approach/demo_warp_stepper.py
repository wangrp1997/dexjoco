"""Step-wise demo_warp planner (one MuJoCo settle step per eval frame)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from skill_graph.adapters.assembly import AssemblySim
from skill_graph.adapters.control import interpolate_arm23, interpolate_arm_only, read_arm23
from skill_graph.constants import APPROACH_LOOKBACK, ObjectName, Side
from skill_graph.math.se3 import arm23_from_object_frame
from skill_graph.skills.approach.demo_warp import (
    BLEND_STEPS,
    GRASP_SNAP_STEPS,
    SQUEEZE_STEPS,
    TABLE_Z_MIN,
    _substeps_for_segment,
)
from skill_graph.skills.templates.schema import GraspTemplate


@dataclass(frozen=True)
class WarpStep:
    side: Side
    active23: np.ndarray
    hold_right: np.ndarray
    hold_left: np.ndarray


def plan_demo_warp_steps(
    sim: AssemblySim,
    template: GraspTemplate,
    *,
    side: Side,
    object_name: ObjectName,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    compact_approach: bool = False,
) -> list[WarpStep]:
    hold_right = np.asarray(hold_right, dtype=np.float64).reshape(23)
    hold_left = np.asarray(hold_left, dtype=np.float64).reshape(23)
    steps: list[WarpStep] = []
    n_wp = int(template.approach_mocap_pos_obj.shape[0])
    open_hand = template.approach_hand[0].copy()
    prev = read_arm23(sim, side)

    def add(active23: np.ndarray) -> None:
        steps.append(
            WarpStep(
                side=side,
                active23=np.asarray(active23, dtype=np.float64).reshape(23),
                hold_right=hold_right.copy(),
                hold_left=hold_left.copy(),
            )
        )

    obj_pos, obj_quat = sim.object_pose(object_name)
    grasp23 = arm23_from_object_frame(
        template.grasp_mocap_pos_obj,
        template.grasp_mocap_quat_obj,
        template.grasp_hand,
        live_obj_pos=obj_pos,
        live_obj_quat=obj_quat,
    )

    if compact_approach:
        blend_n = max(BLEND_STEPS * 2, _substeps_for_segment(prev, grasp23, n_wp=1))
        for bi in range(blend_n):
            alpha = (bi + 1) / blend_n
            add(interpolate_arm_only(prev, grasp23, alpha, hand=open_hand))
        prev = steps[-1].active23 if steps else grasp23
    elif n_wp >= 1:
        first = arm23_from_object_frame(
            template.approach_mocap_pos_obj[0],
            template.approach_mocap_quat_obj[0],
            open_hand,
            live_obj_pos=obj_pos,
            live_obj_quat=obj_quat,
        )
        blend_n = max(BLEND_STEPS, _substeps_for_segment(prev, first, n_wp=n_wp))
        for bi in range(blend_n):
            alpha = (bi + 1) / blend_n
            add(interpolate_arm_only(prev, first, alpha, hand=open_hand))
        prev = steps[-1].active23 if steps else prev

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
                add(interpolate_arm_only(prev, tgt, alpha, hand=open_hand))
            prev = steps[-1].active23

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
        add(interpolate_arm23(prev, grasp23, alpha))
    prev = steps[-1].active23 if steps else grasp23

    squeeze23 = arm23_from_object_frame(
        template.squeeze_mocap_pos_obj,
        template.squeeze_mocap_quat_obj,
        template.squeeze_hand,
        live_obj_pos=obj_pos,
        live_obj_quat=obj_quat,
    )
    for si in range(SQUEEZE_STEPS):
        alpha = (si + 1) / SQUEEZE_STEPS
        add(interpolate_arm23(prev, squeeze23, alpha))

    return steps


@dataclass
class DemoWarpStepper:
    steps: list[WarpStep]
    index: int = 0

    @classmethod
    def from_template(
        cls,
        sim: AssemblySim,
        template: GraspTemplate,
        *,
        side: Side,
        object_name: ObjectName,
        hold_right: np.ndarray,
        hold_left: np.ndarray,
        compact_approach: bool = False,
    ) -> DemoWarpStepper:
        return cls(
            steps=plan_demo_warp_steps(
                sim,
                template,
                side=side,
                object_name=object_name,
                hold_right=hold_right,
                hold_left=hold_left,
                compact_approach=compact_approach,
            )
        )

    def __iter__(self) -> Iterator[WarpStep]:
        while self.index < len(self.steps):
            step = self.steps[self.index]
            self.index += 1
            yield step

    @property
    def done(self) -> bool:
        return self.index >= len(self.steps)

    @property
    def remaining(self) -> int:
        return max(len(self.steps) - self.index, 0)

    def pop_next(self) -> WarpStep | None:
        if self.index >= len(self.steps):
            return None
        step = self.steps[self.index]
        self.index += 1
        return step
