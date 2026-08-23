"""Auditable runtime ownership around sensor-only intent chunk execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .intent_chunk_execution import (
    IntentChunkExecutionConfig,
    IntentChunkStep,
    OnlineIntentChunkExecutor,
)
from .sensor_observation import CerebellumSensorObservation


CONTROL_INPUTS = (
    "state46",
    "arm_joint_torque",
    "fingertip_force_world",
    "wrist_wrench_world",
    "previous_action44",
)


def supports_explicit_handoff(metadata: object) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    capabilities = metadata.get("capabilities")
    if not isinstance(capabilities, Mapping):
        return False
    return capabilities.get("explicit_handoff") is True


@dataclass(frozen=True)
class IntentChunkRuntimeAudit:
    policy_handoff_required: bool
    policy_handoff_observed: bool
    synthetic_handoff_observed: bool
    deployable_handoff_observed: bool
    control_inputs: tuple[str, ...]
    privileged_evaluator_enabled: bool
    events: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_handoff_required": self.policy_handoff_required,
            "policy_handoff_observed": self.policy_handoff_observed,
            "synthetic_handoff_observed": self.synthetic_handoff_observed,
            "deployable_handoff_observed": self.deployable_handoff_observed,
            "control_inputs": list(self.control_inputs),
            "privileged_evaluator_enabled": self.privileged_evaluator_enabled,
            "events": [dict(event) for event in self.events],
        }


class OnlineIntentChunkRuntime:
    """Own handoff execution and produce a sensor-only control audit."""

    def __init__(self, config: IntentChunkExecutionConfig | None = None) -> None:
        self.executor = OnlineIntentChunkExecutor(config)
        self.reset()

    def reset(self) -> None:
        self.executor.reset()
        self._previous_command44: np.ndarray | None = None
        self._replan_reason: str | None = None
        self._events: list[dict[str, object]] = []

    @property
    def active(self) -> bool:
        return self.executor.active

    @property
    def previous_command44(self) -> np.ndarray | None:
        if self._previous_command44 is None:
            return None
        return self._previous_command44.copy()

    @property
    def replan_pending(self) -> bool:
        return self._replan_reason is not None

    def record_rejected_handoff(self, *, timestamp: int, chunk_steps: int) -> None:
        self._events.append(
            {
                "timestamp": int(timestamp),
                "event": "handoff_chunk_too_short",
                "chunk_steps": int(chunk_steps),
            }
        )

    def start(
        self,
        action_chunk44: np.ndarray,
        observation: CerebellumSensorObservation,
        current_action44: np.ndarray,
        *,
        timestamp: int,
        handoff_source: str = "policy",
        handoff_details: Mapping[str, object] | None = None,
    ) -> None:
        if handoff_source not in ("policy", "synthetic_test", "deployable_gate"):
            raise ValueError(
                "handoff_source must be 'policy', 'synthetic_test', or 'deployable_gate'"
            )
        self.executor.start(action_chunk44, observation)
        current = np.asarray(current_action44, dtype=np.float64).reshape(-1)
        if current.shape != (44,) or not np.isfinite(current).all():
            raise ValueError("current_action44 must be a finite 44-vector")
        self._previous_command44 = current.copy()
        self._replan_reason = None
        event = {
            "timestamp": int(timestamp),
            "event": {
                "policy": "policy_handoff",
                "synthetic_test": "synthetic_handoff",
                "deployable_gate": "deployable_handoff",
            }[handoff_source],
            "handoff_source": handoff_source,
            "chunk_steps": int(np.asarray(action_chunk44).shape[0]),
        }
        if handoff_details is not None:
            event["handoff_details"] = dict(handoff_details)
        self._events.append(event)

    def step(
        self,
        observation: CerebellumSensorObservation,
        current_action44: np.ndarray,
        *,
        timestamp: int,
    ) -> IntentChunkStep:
        if self._previous_command44 is None:
            raise RuntimeError("intent chunk runtime has not accepted a handoff")
        if observation.previous_action44 is None:
            raise ValueError("observation must include the previous runtime command")
        if not np.allclose(
            observation.previous_action44,
            self._previous_command44,
            atol=1e-7,
            rtol=1e-7,
        ):
            raise ValueError("observation previous_action44 is not the runtime command")
        result = self.executor.step(observation, current_action44)
        self._previous_command44 = np.asarray(result.action44, dtype=np.float64).copy()
        if not result.active:
            self._replan_reason = result.outcome
            self._events.append(
                {
                    "timestamp": int(timestamp),
                    "event": result.outcome,
                    "phase": result.phase,
                    "time_scale": result.time_scale,
                    "force_scale": result.force_scale,
                    "grasp_scale": result.grasp_scale,
                    "tracking_scale": result.tracking_scale,
                    "minimum_grasp_retention": result.minimum_grasp_retention,
                    "right_motion_fraction": result.right_motion_fraction,
                    "grasp_observable": result.grasp_observable,
                    "contact_phase": result.contact_phase,
                    "contact_correction_m": result.contact_correction_m,
                    "contact_rotation_correction_rad": (
                        result.contact_rotation_correction_rad
                    ),
                    "peak_force_n": result.peak_force_n,
                }
            )
        return result

    def mark_replan_requested(self, *, timestamp: int) -> None:
        if self._replan_reason is None:
            return
        self._events.append(
            {
                "timestamp": int(timestamp),
                "event": "policy_replan_requested",
                "reason": self._replan_reason,
            }
        )
        self._replan_reason = None

    def audit(self, *, privileged_evaluator_enabled: bool) -> IntentChunkRuntimeAudit:
        return IntentChunkRuntimeAudit(
            policy_handoff_required=True,
            policy_handoff_observed=any(
                event["event"] == "policy_handoff" for event in self._events
            ),
            synthetic_handoff_observed=any(
                event["event"] == "synthetic_handoff" for event in self._events
            ),
            deployable_handoff_observed=any(
                event["event"] == "deployable_handoff" for event in self._events
            ),
            control_inputs=CONTROL_INPUTS,
            privileged_evaluator_enabled=bool(privileged_evaluator_enabled),
            events=tuple(dict(event) for event in self._events),
        )
