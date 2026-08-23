"""Shared control-state interfaces for the VLA and cerebellum hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from .primitives import AssemblyPrimitiveSet


class CerebellumMode(str, Enum):
    VLA_APPROACH = "vla_approach"
    VLA_GRASP = "vla_grasp"
    VLA_REGRASP = "vla_regrasp"
    GRASP_ASSIST = "grasp_assist"
    GRASP_STABILIZE = "grasp_stabilize"
    TRANSPORT = "transport"
    ALIGN = "align"
    INSERT = "insert"
    RECOVER_REALIGN = "recover_realign"
    RECOVER_REGRASP = "vla_regrasp"
    COMPLETE = "complete"

    VLA = "vla_approach"
    PREGRASP = "vla_grasp"
    GRASP = "vla_grasp"
    STABILIZE = "grasp_stabilize"


@dataclass(frozen=True)
class CerebellumObservation:
    """Minimal state shared by handoff policies and low-level skills."""

    state44: np.ndarray
    primitives: AssemblyPrimitiveSet
    peg_grasped: bool
    tray_grasped: bool
    peg_pregrasp_ready: bool = False
    tray_pregrasp_ready: bool = False
    peg_grasp_stable: bool = False
    tray_grasp_stable: bool = False
    peg_contact_count: int = 0
    tray_contact_count: int = 0
    insert_contact: bool = False
    slip_speed_mps: float | None = None
    rotation_slip_radps: float | None = None

    def __post_init__(self) -> None:
        state = np.asarray(self.state44, dtype=np.float32).reshape(-1)
        if state.shape != (44,):
            raise ValueError(f"state44 must have shape (44,), got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("state44 must contain finite values")
        if self.peg_contact_count < 0 or self.tray_contact_count < 0:
            raise ValueError("contact counts must be non-negative")
        if self.slip_speed_mps is not None and self.slip_speed_mps < 0.0:
            raise ValueError("slip_speed_mps must be non-negative")
        if self.rotation_slip_radps is not None and self.rotation_slip_radps < 0.0:
            raise ValueError("rotation_slip_radps must be non-negative")
        state = state.copy()
        state.setflags(write=False)
        object.__setattr__(self, "state44", state)


@dataclass(frozen=True)
class HandoffDecision:
    mode: CerebellumMode
    reason: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


class HandoffPolicy(Protocol):
    def decide(self, observation: CerebellumObservation) -> HandoffDecision:
        """Select which controller should own the next control step."""


class CerebellumSkill(Protocol):
    mode: CerebellumMode

    def reset(self) -> None:
        """Reset per-episode state."""

    def step(self, observation: CerebellumObservation, vla_action44: np.ndarray) -> np.ndarray:
        """Return the 44D action to execute for this control step."""
