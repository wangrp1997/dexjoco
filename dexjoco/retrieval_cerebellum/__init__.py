"""Retrieval-augmented low-level control for dexterous assembly."""

from typing import TYPE_CHECKING

from .control import CerebellumMode, CerebellumObservation, HandoffDecision
from .grasp_assist import (
    AsymmetricGraspAssist,
    AsymmetricGraspAssistConfig,
    GraspAssistDiagnostics,
)
from .handoff import RuleBasedHandoffPolicy, RuleHandoffConfig
from .handoff_gate import (
    CoarseAlignmentGateConfig,
    CoarseAlignmentGateDecision,
    DeployableGraspHandoffGate,
    DeployableCoarseAlignmentGate,
    HandoffGateConfig,
    HandoffGateDecision,
    SideHandoffEvidence,
)
from .intent_chunk_execution import (
    IntentChunkExecutionConfig,
    IntentChunkStep,
    OnlineIntentChunkExecutor,
)
from .intent_chunk_runtime import (
    CONTROL_INPUTS,
    IntentChunkRuntimeAudit,
    OnlineIntentChunkRuntime,
    supports_explicit_handoff,
)
from .primitives import AssemblyPrimitiveSet, ContactRegion, PriorSource

if TYPE_CHECKING:
    from .monitor import PrivilegedCerebellumEvaluator, ReadOnlyCerebellumMonitor
    from .observer import PrivilegedCerebellumObserver
    from .privileged import PrivilegedAssemblyPrimitiveProvider
    from .real_sensor_model import (
        RealisticCerebellumObservation,
        SensorDegrader,
        SensorModelConfig,
    )
    from .sensor_observation import CerebellumSensorObservation, SensorTraceRecorder
    from .sim_sensor_adapter import SimCerebellumSensorAdapter

__all__ = [
    "AssemblyPrimitiveSet",
    "AsymmetricGraspAssist",
    "AsymmetricGraspAssistConfig",
    "CerebellumMode",
    "CerebellumObservation",
    "ContactRegion",
    "HandoffDecision",
    "GraspAssistDiagnostics",
    "PriorSource",
    "PrivilegedCerebellumEvaluator",
    "PrivilegedCerebellumObserver",
    "PrivilegedAssemblyPrimitiveProvider",
    "ReadOnlyCerebellumMonitor",
    "RuleBasedHandoffPolicy",
    "RuleHandoffConfig",
    "RealisticCerebellumObservation",
    "CerebellumSensorObservation",
    "SensorDegrader",
    "SensorModelConfig",
    "SensorTraceRecorder",
    "SimCerebellumSensorAdapter",
]


def __getattr__(name: str):
    if name == "PrivilegedCerebellumObserver":
        from .observer import PrivilegedCerebellumObserver

        return PrivilegedCerebellumObserver
    if name == "PrivilegedAssemblyPrimitiveProvider":
        from .privileged import PrivilegedAssemblyPrimitiveProvider

        return PrivilegedAssemblyPrimitiveProvider
    if name == "ReadOnlyCerebellumMonitor":
        from .monitor import ReadOnlyCerebellumMonitor

        return ReadOnlyCerebellumMonitor
    if name == "PrivilegedCerebellumEvaluator":
        from .monitor import PrivilegedCerebellumEvaluator

        return PrivilegedCerebellumEvaluator
    if name in {"CerebellumSensorObservation", "SensorTraceRecorder"}:
        from .sensor_observation import CerebellumSensorObservation, SensorTraceRecorder

        return {
            "CerebellumSensorObservation": CerebellumSensorObservation,
            "SensorTraceRecorder": SensorTraceRecorder,
        }[name]
    if name == "SimCerebellumSensorAdapter":
        from .sim_sensor_adapter import SimCerebellumSensorAdapter

        return SimCerebellumSensorAdapter
    if name in {
        "RealisticCerebellumObservation",
        "SensorDegrader",
        "SensorModelConfig",
    }:
        from .real_sensor_model import (
            RealisticCerebellumObservation,
            SensorDegrader,
            SensorModelConfig,
        )

        return {
            "RealisticCerebellumObservation": RealisticCerebellumObservation,
            "SensorDegrader": SensorDegrader,
            "SensorModelConfig": SensorModelConfig,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
