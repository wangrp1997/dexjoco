"""Compiled skill-graph structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphEdge:
    id: str
    source: str
    target: str
    action: dict[str, Any] | None = None
    monitors: list[dict[str, Any]] = field(default_factory=list)
    is_recovery: bool = False


@dataclass
class RecoveryBranch:
    edge_id: str
    monitor_index: int
    failure_name: str
    recovery_edge_ids: list[str]


@dataclass
class CompiledGraph:
    task: str
    start: str
    goal: str
    nodes: dict[str, str]
    edges: dict[str, GraphEdge]
    nominal_outgoing: dict[str, list[str]]
    recovery_branches: dict[tuple[str, int], RecoveryBranch]
