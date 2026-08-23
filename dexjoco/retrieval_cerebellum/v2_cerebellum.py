"""Executable sensor-to-action entry point for the V2 cerebellum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assembly_kinematics import apply_bimanual_wrist_twists
from .sensor_observation import CerebellumSensorObservation
from .v2_contact import V2ContactInterpreter, V2ContactInterpreterConfig
from .v2_control import (
    ModeGatedCompliantController,
    V2AssemblyEstimate,
    V2ContactSignals,
    V2ControllerCommand,
    V2ControllerConfig,
    V2Mode,
)


@dataclass(frozen=True)
class V2StepResult:
    action44: np.ndarray
    command: V2ControllerCommand
    contact: V2ContactSignals

    def __post_init__(self) -> None:
        action = np.asarray(self.action44, dtype=np.float32).reshape(-1)
        if action.shape != (44,):
            raise ValueError(f"action44 must have shape (44,), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("action44 must contain finite values")
        action = action.copy()
        action.flags.writeable = False
        object.__setattr__(self, "action44", action)


class V2Cerebellum:
    """Compose deployable contact interpretation and mode-gated control."""

    def __init__(
        self,
        *,
        controller_config: V2ControllerConfig | None = None,
        contact_config: V2ContactInterpreterConfig | None = None,
    ) -> None:
        self.controller = ModeGatedCompliantController(controller_config)
        self.contact_interpreter = V2ContactInterpreter(contact_config)

    def reset(self) -> None:
        self.controller.reset()
        self.contact_interpreter.reset()

    def step(
        self,
        observation: CerebellumSensorObservation,
        estimate: V2AssemblyEstimate,
        current_action44: np.ndarray,
    ) -> V2StepResult:
        if abs(float(observation.timestamp_s) - estimate.timestamp_s) > 0.2:
            raise ValueError("sensor observation and visual estimate are not time aligned")
        contact = self.contact_interpreter.update(
            observation,
            estimate,
            allow_baseline_update=self.controller.mode == V2Mode.ALIGN,
        )
        command = self.controller.step(estimate, contact)
        action = apply_bimanual_wrist_twists(
            current_action44,
            command.right_twist_world,
            command.left_twist_world,
            finger_reference44=current_action44,
        )
        return V2StepResult(
            action44=action,
            command=command,
            contact=contact,
        )
