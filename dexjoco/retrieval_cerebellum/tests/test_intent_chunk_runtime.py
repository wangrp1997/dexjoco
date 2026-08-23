import numpy as np
import pytest

from retrieval_cerebellum.intent_chunk_runtime import (
    CONTROL_INPUTS,
    OnlineIntentChunkRuntime,
    supports_explicit_handoff,
)
from retrieval_cerebellum.sensor_observation import CerebellumSensorObservation


def _observation(
    *,
    force_n: float = 0.0,
    previous_action44: np.ndarray | None = None,
) -> CerebellumSensorObservation:
    wrist_wrench = np.zeros((2, 6), dtype=np.float64)
    wrist_wrench[:, 0] = force_n
    return CerebellumSensorObservation(
        timestamp_s=0.0,
        state46=np.zeros(46),
        arm_joint_torque=np.zeros((2, 7)),
        fingertip_force_world=np.zeros((2, 4, 3)),
        wrist_wrench_world=wrist_wrench,
        images={},
        previous_action44=previous_action44,
    )


def _chunk() -> np.ndarray:
    chunk = np.zeros((2, 44), dtype=np.float64)
    chunk[1, [0, 22]] = 0.001
    return chunk


def test_runtime_audits_handoff_completion_and_explicit_replan() -> None:
    runtime = OnlineIntentChunkRuntime()
    current = np.zeros(44)
    runtime.start(_chunk(), _observation(), current, timestamp=10)

    result = runtime.step(
        _observation(previous_action44=runtime.previous_command44),
        current,
        timestamp=11,
    )

    assert not result.active
    assert result.outcome == "chunk_complete"
    assert runtime.replan_pending
    runtime.mark_replan_requested(timestamp=12)
    audit = runtime.audit(privileged_evaluator_enabled=False).to_dict()

    assert audit["policy_handoff_observed"] is True
    assert audit["privileged_evaluator_enabled"] is False
    assert audit["control_inputs"] == list(CONTROL_INPUTS)
    assert [event["event"] for event in audit["events"]] == [
        "policy_handoff",
        "chunk_complete",
        "policy_replan_requested",
    ]
    assert audit["events"][-1]["reason"] == "chunk_complete"
    assert audit["events"][1]["grasp_observable"] is False
    assert audit["events"][1]["contact_phase"] == "inactive"
    assert not runtime.replan_pending


def test_runtime_rejects_observation_with_wrong_previous_command() -> None:
    runtime = OnlineIntentChunkRuntime()
    current = np.zeros(44)
    runtime.start(_chunk(), _observation(), current, timestamp=0)
    wrong_previous = np.zeros(44)
    wrong_previous[0] = 0.01

    with pytest.raises(ValueError, match="not the runtime command"):
        runtime.step(
            _observation(previous_action44=wrong_previous),
            current,
            timestamp=1,
        )


def test_runtime_records_sensor_safety_exit_without_success_claim() -> None:
    runtime = OnlineIntentChunkRuntime()
    current = np.zeros(44)
    runtime.start(_chunk(), _observation(), current, timestamp=0)

    result = runtime.step(
        _observation(
            force_n=20.0,
            previous_action44=runtime.previous_command44,
        ),
        current,
        timestamp=1,
    )
    runtime.mark_replan_requested(timestamp=2)
    events = runtime.audit(privileged_evaluator_enabled=False).events

    assert result.outcome == "hard_force_retreat"
    assert [event["event"] for event in events] == [
        "policy_handoff",
        "hard_force_retreat",
        "policy_replan_requested",
    ]
    assert all(event["event"] != "success" for event in events)


def test_rejected_short_handoff_does_not_count_as_observed() -> None:
    runtime = OnlineIntentChunkRuntime()
    runtime.record_rejected_handoff(timestamp=4, chunk_steps=1)

    audit = runtime.audit(privileged_evaluator_enabled=False)

    assert not audit.policy_handoff_observed
    assert not audit.synthetic_handoff_observed
    assert not audit.deployable_handoff_observed
    assert audit.events == (
        {
            "timestamp": 4,
            "event": "handoff_chunk_too_short",
            "chunk_steps": 1,
        },
    )


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({}, False),
        ({"capabilities": {}}, False),
        ({"capabilities": {"explicit_handoff": False}}, False),
        ({"capabilities": {"explicit_handoff": True}}, True),
        ({"capabilities": {"explicit_handoff": 1}}, False),
        (None, False),
    ],
)
def test_explicit_handoff_capability_requires_strict_true(
    metadata: object,
    expected: bool,
) -> None:
    assert supports_explicit_handoff(metadata) is expected


def test_synthetic_handoff_is_audited_without_policy_claim() -> None:
    runtime = OnlineIntentChunkRuntime()
    current = np.zeros(44)

    runtime.start(
        _chunk(),
        _observation(),
        current,
        timestamp=8,
        handoff_source="synthetic_test",
    )
    audit = runtime.audit(privileged_evaluator_enabled=False)

    assert not audit.policy_handoff_observed
    assert audit.synthetic_handoff_observed
    assert audit.events[0]["event"] == "synthetic_handoff"
    assert audit.events[0]["handoff_source"] == "synthetic_test"


def test_deployable_gate_handoff_is_separate_from_policy_and_synthetic() -> None:
    runtime = OnlineIntentChunkRuntime()
    runtime.start(
        _chunk(),
        _observation(),
        np.zeros(44),
        timestamp=9,
        handoff_source="deployable_gate",
    )

    audit = runtime.audit(privileged_evaluator_enabled=False)

    assert audit.deployable_handoff_observed
    assert not audit.policy_handoff_observed
    assert not audit.synthetic_handoff_observed
    assert audit.events[0]["event"] == "deployable_handoff"
