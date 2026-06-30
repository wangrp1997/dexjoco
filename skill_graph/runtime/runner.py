"""AgentChord-style graph runner with monitor-triggered recovery."""

from __future__ import annotations

from dataclasses import dataclass, field

from skill_graph.runtime.context import GraphRunContext
from skill_graph.runtime.models import CompiledGraph
from skill_graph.runtime.skills import SkillResult, execute_skill


@dataclass
class GraphRunReport:
    success: bool
    reached_goal: bool
    fail_reason: str = ""
    insert_ok: bool = False
    edges_executed: list[str] = field(default_factory=list)
    recovery_counts: dict[str, int] = field(default_factory=dict)


class SkillGraphRunner:
    def __init__(self, graph: CompiledGraph, *, max_transitions: int = 200) -> None:
        self.graph = graph
        self.max_transitions = max_transitions

    def run(self, ctx: GraphRunContext, *, start_node: str | None = None) -> GraphRunReport:
        g = self.graph
        current = start_node or g.start
        pending: list[str] = []
        continuation: list[list[str]] = []
        executed: list[str] = []
        insert_ok = False
        transitions = 0

        while current != g.goal or pending or continuation:
            transitions += 1
            if transitions > self.max_transitions:
                return GraphRunReport(
                    False,
                    False,
                    fail_reason="max_transitions",
                    edges_executed=executed,
                    recovery_counts=dict(ctx.recovery_counts),
                )

            if not pending and continuation:
                pending = continuation.pop()

            if pending:
                edge_id = pending.pop(0)
            else:
                edge_id = self._next_nominal_edge(current)
            edge = g.edges[edge_id]
            executed.append(edge_id)

            result = execute_skill(ctx, edge.action, monitors=edge.monitors if edge.monitors else None)

            if result.monitor_index is not None:
                branch = g.recovery_branches.get((edge.id, result.monitor_index))
                if branch is None:
                    return GraphRunReport(
                        False,
                        False,
                        fail_reason=f"no_recovery:{edge.id}:{result.monitor_index}",
                        edges_executed=executed,
                        recovery_counts=dict(ctx.recovery_counts),
                    )
                cont: list[str] = []
                if branch.recovery_edge_ids:
                    merge_node = g.edges[branch.recovery_edge_ids[-1]].target
                    if merge_node == edge.source and edge.source != edge.target:
                        cont = [edge.id, *pending]
                    elif merge_node != edge.target:
                        return GraphRunReport(
                            False,
                            False,
                            fail_reason=f"bad_recovery_merge:{edge.id}",
                            edges_executed=executed,
                            recovery_counts=dict(ctx.recovery_counts),
                        )
                    else:
                        cont = list(pending)
                if cont:
                    continuation.append(cont)
                pending = list(branch.recovery_edge_ids)
                current = edge.source
                continue

            if not result.ok and edge.action and edge.action.get("fn") != "retry":
                return GraphRunReport(
                    False,
                    current == g.goal,
                    fail_reason=result.reason or f"edge_failed:{edge_id}",
                    insert_ok=insert_ok,
                    edges_executed=executed,
                    recovery_counts=dict(ctx.recovery_counts),
                )

            insert_ok = insert_ok or result.insert_ok
            current = edge.target

        return GraphRunReport(
            True,
            True,
            insert_ok=insert_ok,
            edges_executed=executed,
            recovery_counts=dict(ctx.recovery_counts),
        )

    def _next_nominal_edge(self, node_id: str) -> str:
        for eid in self.graph.nominal_outgoing.get(node_id, []):
            if not self.graph.edges[eid].is_recovery:
                return eid
        raise RuntimeError(f"No nominal edge from {node_id}")
