import numpy as np

from retrieval_cerebellum.handoff_gate import (
    CoarseAlignmentGateConfig,
    DeployableCoarseAlignmentGate,
    DeployableGraspHandoffGate,
    HandoffGateConfig,
)
from retrieval_cerebellum.sensor_observation import CerebellumSensorObservation


def _observation(
    frame: int,
    *,
    right_contact: bool,
    left_contact: bool,
    transport_m: float,
) -> CerebellumSensorObservation:
    state = np.zeros(46, dtype=np.float64)
    progress = frame / 99.0
    state[:3] = [transport_m * progress, 0.0, 0.0]
    state[7:10] = [0.0, transport_m * progress, 0.0]
    fingertip = np.zeros((2, 4, 3), dtype=np.float64)
    if right_contact:
        fingertip[0, :2, 0] = 1.0
    if left_contact:
        fingertip[1, :2, 0] = 1.0
    return CerebellumSensorObservation(
        timestamp_s=frame * 0.02,
        state46=state,
        arm_joint_torque=np.zeros((2, 7)),
        fingertip_force_world=fingertip,
        wrist_wrench_world=np.zeros((2, 6)),
        images={},
    )


def test_gate_rejects_handoff_after_current_grasp_is_lost() -> None:
    gate = DeployableGraspHandoffGate()
    decision = None
    for frame in range(100):
        decision = gate.update(
            _observation(
                frame,
                right_contact=frame < 95,
                left_contact=True,
                transport_m=0.02,
            )
        )

    assert decision is not None and not decision.ready
    assert decision.right.contact_fraction == 0.95
    assert decision.right.trailing_contact_gap_frames == 5
    assert decision.right.current_contact_fingers == 0
    assert decision.right.transport_span_m >= 0.0189


def test_gate_accepts_persistent_current_bilateral_grasp_after_transport() -> None:
    gate = DeployableGraspHandoffGate()
    decision = None
    for frame in range(100):
        decision = gate.update(
            _observation(
                frame,
                right_contact=True,
                left_contact=True,
                transport_m=0.02,
            )
        )

    assert decision is not None and decision.ready
    assert decision.right.current_contact_fingers == 2
    assert decision.left.current_contact_fingers == 2


def test_gate_rejects_fingers_closing_without_contact() -> None:
    gate = DeployableGraspHandoffGate()
    decision = None
    for frame in range(100):
        decision = gate.update(
            _observation(
                frame,
                right_contact=False,
                left_contact=True,
                transport_m=0.04,
            )
        )

    assert decision is not None and not decision.ready
    assert not decision.right.ready
    assert decision.right.contact_fraction == 0.0


def test_gate_rejects_static_contact_without_transport_evidence() -> None:
    gate = DeployableGraspHandoffGate()
    decision = None
    for frame in range(100):
        decision = gate.update(
            _observation(
                frame,
                right_contact=True,
                left_contact=True,
                transport_m=0.0,
            )
        )

    assert decision is not None and not decision.ready
    assert decision.right.contact_fraction == 1.0
    assert decision.right.transport_span_m == 0.0


def test_gate_can_be_calibrated_for_shorter_hardware_windows() -> None:
    gate = DeployableGraspHandoffGate(
        HandoffGateConfig(
            window_frames=10,
            minimum_contact_fraction=0.8,
            maximum_contact_gap_frames=2,
            minimum_transport_span_m=0.005,
        )
    )
    decision = None
    for frame in range(10):
        decision = gate.update(
                _observation(
                    frame * 11,
                    right_contact=frame != 8,
                    left_contact=True,
                    transport_m=0.01,
                )
        )

    assert decision is not None and decision.ready


def test_coarse_alignment_gate_uses_only_robot_side_workspace_signals() -> None:
    gate = DeployableCoarseAlignmentGate(
        CoarseAlignmentGateConfig(persistence_frames=2)
    )
    state = np.zeros(46, dtype=np.float64)
    state[:3] = [-0.41, -0.13, 1.46]
    state[7:10] = [-0.43, 0.14, 1.42]
    state[14:30] = 1.0
    state[30:46] = 1.0
    observation = CerebellumSensorObservation(
        timestamp_s=0.0,
        state46=state,
        arm_joint_torque=np.zeros((2, 7)),
        fingertip_force_world=np.zeros((2, 4, 3)),
        wrist_wrench_world=np.zeros((2, 6)),
        images={},
    )

    assert not gate.update(observation).ready
    decision = gate.update(observation)

    assert decision.ready
    assert decision.ready_streak == 2


def test_coarse_alignment_gate_rejects_open_hands_in_workspace() -> None:
    gate = DeployableCoarseAlignmentGate(
        CoarseAlignmentGateConfig(persistence_frames=1)
    )
    state = np.zeros(46, dtype=np.float64)
    state[:3] = [-0.41, -0.13, 1.46]
    state[7:10] = [-0.43, 0.14, 1.42]
    decision = gate.update(
        CerebellumSensorObservation(
            timestamp_s=0.0,
            state46=state,
            arm_joint_torque=np.zeros((2, 7)),
            fingertip_force_world=np.zeros((2, 4, 3)),
            wrist_wrench_world=np.zeros((2, 6)),
            images={},
        )
    )

    assert not decision.ready
