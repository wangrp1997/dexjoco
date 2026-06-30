"""Load task graph JSON and expand recovery edges."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from skill_graph.runtime.models import CompiledGraph, GraphEdge, RecoveryBranch


def _edge_action(edge: dict[str, Any]) -> dict[str, Any] | None:
    if edge.get("left_arm_action"):
        return dict(edge["left_arm_action"])
    if edge.get("right_arm_action"):
        return dict(edge["right_arm_action"])
    return None


def _robot_object(step: dict[str, Any]) -> tuple[str, str]:
    robot = str(step.get("robot_name", "right_arm"))
    obj = str(step.get("obj_name", "peg"))
    side = "left" if "left" in robot else "right"
    return side, obj


def compile_graph_bundle(path: Path | str) -> CompiledGraph:
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    task_graph = bundle["task_graph"]
    recovery_spec = bundle.get("recovery_spec", {})
    task = str(task_graph["task"])
    start = str(task_graph["start"])
    goal = str(task_graph["goal"])

    nodes = {str(n["id"]): str(n.get("semantic", "")) for n in task_graph["nodes"]}
    edges: dict[str, GraphEdge] = {}
    nominal_out: dict[str, list[str]] = defaultdict(list)

    for e in task_graph["edges"]:
        eid = str(e["id"])
        edge = GraphEdge(
            id=eid,
            source=str(e["source"]),
            target=str(e["target"]),
            action=_edge_action(e),
        )
        edges[eid] = edge
        nominal_out[edge.source].append(eid)

    recovery_branches: dict[tuple[str, int], RecoveryBranch] = {}

    for bind_idx, binding in enumerate(recovery_spec.get("recovery_bindings", [])):
        edge_id = str(binding["edge_id"])
        failure_name = str(binding.get("failure_name", f"failure_{bind_idx}"))
        monitored = edges[edge_id]
        monitors = list(binding.get("monitors", []))
        monitored.monitors = monitors

        recovery_steps = binding.get("recovery", [])
        recovery_edge_ids: list[str] = []
        merge = str(binding.get("merge", "source"))
        merge_node = monitored.source if merge == "source" else monitored.target

        for step_i, step in enumerate(recovery_steps):
            step = dict(step)
            step_type = str(step.get("type", "regrasp"))
            rid = f"re_{edge_id}_{bind_idx}_{step_i}_{failure_name}"
            if step_type == "regrasp":
                side, obj = _robot_object(step)
                action = {"fn": "regrasp", "kwargs": {"side": side, "object_name": obj}}
            elif step_type == "retry_failed_edge":
                action = {"fn": "retry", "kwargs": {"edge_id": edge_id}}
            else:
                action = {"fn": step_type, "kwargs": dict(step)}
            rec_edge = GraphEdge(
                id=rid,
                source=merge_node,
                target=merge_node,
                action=action,
                is_recovery=True,
            )
            edges[rid] = rec_edge
            recovery_edge_ids.append(rid)

        recovery_branches[(edge_id, bind_idx)] = RecoveryBranch(
            edge_id=edge_id,
            monitor_index=bind_idx,
            failure_name=failure_name,
            recovery_edge_ids=recovery_edge_ids,
        )

    return CompiledGraph(
        task=task,
        start=start,
        goal=goal,
        nodes=nodes,
        edges=edges,
        nominal_outgoing=dict(nominal_out),
        recovery_branches=recovery_branches,
    )
