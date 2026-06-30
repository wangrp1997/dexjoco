"""Leaf skill implementations for graph edges."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from skill_graph.adapters.control import read_arm23
from skill_graph.constants import ObjectName, Side
from skill_graph.io.actions import zarr_to_raw_dict
from skill_graph.runtime.context import GraphRunContext
from skill_graph.runtime.monitors import first_triggered_monitor
from skill_graph.skills.regrasp.execute import execute_regrasp


@dataclass
class SkillResult:
    ok: bool
    monitor_index: int | None = None
    reason: str = ""
    insert_ok: bool = False


MonitorPoll = Callable[[], int | None]


def _replay_frames(ctx: GraphRunContext, start: int, end: int, poll: MonitorPoll | None = None) -> SkillResult:
    actions = ctx.zarr_actions
    if actions is None:
        return SkillResult(False, reason="no_zarr_actions")
    for fi in range(int(start), int(end) + 1):
        if poll is not None:
            midx = poll()
            if midx is not None:
                return SkillResult(False, monitor_index=midx, reason="monitor_triggered")
        ctx.sim.raw.step(zarr_to_raw_dict(actions[fi]))
        ctx.sim.on_physics_step()
    return SkillResult(True)


def _timing(ctx: GraphRunContext) -> dict:
    return ctx.manifest_entry["timing"]


def skill_regrasp(ctx: GraphRunContext, kwargs: dict[str, Any], poll: MonitorPoll | None) -> SkillResult:
    side = kwargs.get("side", "right")
    object_name = kwargs.get("object_name", "peg")
    hold_r = read_arm23(ctx.sim, "right")
    hold_l = read_arm23(ctx.sim, "left")
    report = execute_regrasp(
        ctx.sim,
        ctx.templates,
        side=side,
        object_name=object_name,
        hold_right=hold_r,
        hold_left=hold_l,
        prefer_episode=ctx.prefer_episode,
    )
    key = f"regrasp_{object_name}"
    ctx.recovery_counts[key] = ctx.recovery_counts.get(key, 0) + 1
    if poll is not None:
        midx = poll()
        if midx is not None:
            return SkillResult(False, monitor_index=midx, reason="monitor_after_regrasp")
    return SkillResult(report.success, reason="" if report.success else report.reason)


def skill_grasp(ctx: GraphRunContext, kwargs: dict[str, Any], poll: MonitorPoll | None) -> SkillResult:
    obj = str(kwargs.get("obj_name", "peg"))
    side: Side = "left" if obj == "tray" else "right"
    return skill_regrasp(ctx, {"side": side, "object_name": obj}, poll)


def skill_demo_replay(ctx: GraphRunContext, kwargs: dict[str, Any], poll: MonitorPoll | None) -> SkillResult:
    timing = _timing(ctx)
    segment = str(kwargs.get("segment", "peg_lift"))
    if segment == "peg_lift":
        start = int(timing["right_grasp_frame"])
        end = int(timing.get("peg_lift_start", start + 20))
    elif segment == "align":
        start = int(timing.get("peg_lift_start", timing["right_grasp_frame"]))
        end = int(ctx.peg_lift_end_frame or start + 30)
    else:
        return SkillResult(False, reason=f"unknown_segment:{segment}")
    return _replay_frames(ctx, start, end, poll)


def skill_pose_insert(ctx: GraphRunContext, kwargs: dict[str, Any], poll: MonitorPoll | None) -> SkillResult:
    if ctx.insert_ckpt is None:
        return SkillResult(False, reason="no_insert_ckpt")

    from interaction_retarget.skill_replay.insert import run_pose_insert_phase
    from skill_graph.runtime.monitors import first_triggered_monitor

    report = run_pose_insert_phase(
        ctx.sim.env,
        ctx.sim.raw,
        reach_mode=str(kwargs.get("reach_mode", ctx.insert_reach_mode)),
        max_steps=900,
        poseinsert_ckpt=ctx.insert_ckpt,
        poseinsert_data_root=ctx.insert_data_root,
        manifest_entry=ctx.manifest_entry,
        sidecar_dir=ctx.sidecar_dir,
        peg_lift_end_frame=int(ctx.peg_lift_end_frame or 0),
        insert_mode=str(kwargs.get("mode", "action44")),
        peg_rest_z=ctx.peg_rest_z,
        tray_rest_z=ctx.tray_rest_z,
    )
    if not report.insert_ok:
        midx = poll() if poll is not None else first_triggered_monitor(ctx.sim, [])
        if midx is None and ("policy" in report.fail_reason or "peg" in report.fail_reason.lower()):
            midx = 0
        if midx is not None:
            return SkillResult(False, monitor_index=midx, reason=report.fail_reason, insert_ok=False)
    return SkillResult(
        bool(report.success),
        reason=str(report.fail_reason),
        insert_ok=bool(report.insert_ok),
    )


def skill_retry(ctx: GraphRunContext, kwargs: dict[str, Any], poll: MonitorPoll | None) -> SkillResult:
    return SkillResult(True, reason="retry_marker")


_SKILL_FNS = {
    "grasp": skill_grasp,
    "regrasp": skill_regrasp,
    "demo_replay": skill_demo_replay,
    "pose_insert": skill_pose_insert,
    "retry": skill_retry,
}


def execute_skill(
    ctx: GraphRunContext,
    action: dict[str, Any] | None,
    *,
    monitors: list[dict] | None = None,
) -> SkillResult:
    if action is None:
        return SkillResult(True)

    def poll() -> int | None:
        if not monitors:
            return None
        return first_triggered_monitor(ctx.sim, monitors)

    fn = str(action.get("fn", ""))
    kwargs = dict(action.get("kwargs") or {})
    skill = _SKILL_FNS.get(fn)
    if skill is None:
        return SkillResult(False, reason=f"unknown_skill:{fn}")
    return skill(ctx, kwargs, poll if monitors else None)
