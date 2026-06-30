"""Approach planner backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from skill_graph.adapters.assembly import AssemblySim
from skill_graph.constants import ObjectName, Side
from skill_graph.skills.templates.schema import GraspTemplate

ApproachBackendName = Literal["demo_warp", "curobo"]


@dataclass
class ApproachResult:
    success: bool
    steps: int
    collision: bool = False
    reason: str = ""


class ApproachBackend(ABC):
    name: ApproachBackendName

    @abstractmethod
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
        raise NotImplementedError


def get_approach_backend(name: ApproachBackendName = "demo_warp") -> ApproachBackend:
    from skill_graph.skills.approach.demo_warp import DemoWarpApproach

    if name == "demo_warp":
        return DemoWarpApproach()
    if name == "curobo":
        raise NotImplementedError("CuRobo approach backend not wired yet")
    raise ValueError(f"Unknown approach backend: {name}")
