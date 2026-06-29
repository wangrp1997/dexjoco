"""Bench thresholds (DexGraspBench fc_mocap inspired, simplified)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchConfig:
    hold_steps: int = 20
    warmup_steps: int = 8
    contact_window: int = 10
    min_contact_count: int = 3
    max_object_trans_m: float = 0.008
    max_object_rot_rad: float = 0.12
    require_hole_clear: bool = True
