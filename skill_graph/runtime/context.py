"""Runtime context passed to every skill invocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skill_graph.adapters.assembly import AssemblySim
from skill_graph.skills.templates.schema import GraspTemplate


@dataclass
class GraphRunContext:
    sim: AssemblySim
    manifest_entry: dict[str, Any]
    sidecar_dir: Path
    templates: list[GraspTemplate]
    zarr_actions: Any = None
    initial_state: Any = None
    peg_lift_end_frame: int | None = None
    peg_rest_z: float | None = None
    tray_rest_z: float | None = None
    insert_ckpt: Path | None = None
    insert_data_root: Path | None = None
    insert_reach_mode: str = "policy"
    prefer_episode: int | None = None
    recovery_counts: dict[str, int] = field(default_factory=dict)

    @property
    def episode_index(self) -> int:
        return int(self.manifest_entry["episode_index"])
