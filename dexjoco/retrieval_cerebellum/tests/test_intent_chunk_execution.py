import numpy as np

from retrieval_cerebellum.intent_chunk_execution import (
    IntentChunkExecutionConfig,
    OnlineIntentChunkExecutor,
)
from retrieval_cerebellum.sensor_observation import CerebellumSensorObservation


def _observation(
    force_n: float = 0.0,
    *,
    side_force_n: tuple[float, float] | None = None,
    fingertip_force_n: tuple[float, float] = (0.0, 0.0),
    side_torque_nm: tuple[float, float] = (0.0, 0.0),
    previous_action44: np.ndarray | None = None,
) -> CerebellumSensorObservation:
    wrench = np.zeros((2, 6), dtype=np.float64)
    wrench[:, 0] = force_n if side_force_n is None else side_force_n
    wrench[:, 3] = side_torque_nm
    fingertip_force = np.zeros((2, 4, 3), dtype=np.float64)
    fingertip_force[0, :, 0] = fingertip_force_n[0]
    fingertip_force[1, :, 0] = fingertip_force_n[1]
    return CerebellumSensorObservation(
        timestamp_s=0.0,
        state46=np.zeros(46),
        arm_joint_torque=np.zeros((2, 7)),
        fingertip_force_world=fingertip_force,
        wrist_wrench_world=wrench,
        images={},
        previous_action44=previous_action44,
    )


def _chunk() -> np.ndarray:
    chunk = np.zeros((3, 44), dtype=np.float64)
    chunk[:, 0] = [0.0, 0.001, 0.002]
    chunk[:, 22] = [0.0, 0.001, 0.002]
    return chunk


def test_executes_frozen_chunk_to_completion() -> None:
    executor = OnlineIntentChunkExecutor()
    executor.start(_chunk(), _observation())
    current = np.zeros(44)
    first = executor.step(_observation(), current)
    second = executor.step(_observation(), first.action44)
    assert first.active
    assert not second.active
    assert second.outcome == "chunk_complete"
    np.testing.assert_allclose(second.action44[[0, 22]], [0.002, 0.002])


def test_soft_force_retimes_without_skipping_chunk_motion() -> None:
    executor = OnlineIntentChunkExecutor(
        IntentChunkExecutionConfig(
            soft_force_limit_n=2.0,
            hard_force_limit_n=10.0,
            minimum_time_scale=0.1,
        )
    )
    executor.start(_chunk(), _observation())
    result = executor.step(_observation(6.0), np.zeros(44))
    assert result.active
    assert result.time_scale == 0.5
    np.testing.assert_allclose(result.action44[[0, 22]], [0.0005, 0.0005])


def test_hard_force_retreats_opposite_chunk_translation() -> None:
    executor = OnlineIntentChunkExecutor(
        IntentChunkExecutionConfig(soft_force_limit_n=2.0, hard_force_limit_n=5.0)
    )
    executor.start(_chunk(), _observation())
    result = executor.step(_observation(5.0), np.zeros(44))
    assert not result.active
    assert result.outcome == "hard_force_retreat"
    assert result.action44[0] < 0.0
    assert result.action44[22] < 0.0


def test_step_limit_retimes_instead_of_skipping_motion() -> None:
    chunk = _chunk()
    chunk[1:, [0, 22]] *= 10.0
    executor = OnlineIntentChunkExecutor(
        IntentChunkExecutionConfig(maximum_translation_step_m=0.001)
    )
    executor.start(chunk, _observation())
    result = executor.step(_observation(), np.zeros(44))
    assert result.phase == 0.1
    np.testing.assert_allclose(result.action44[[0, 22]], [0.001, 0.001])


def test_grasp_force_loss_stops_without_commanding_more_motion() -> None:
    executor = OnlineIntentChunkExecutor(
        IntentChunkExecutionConfig(enforce_grasp_retention=True)
    )
    baseline = _observation(fingertip_force_n=(2.0, 2.0))
    executor.start(_chunk(), baseline)

    result = executor.step(
        _observation(fingertip_force_n=(0.2, 2.0)),
        np.zeros(44),
    )

    assert not result.active
    assert result.outcome == "grasp_unstable_stop"
    assert np.isclose(result.minimum_grasp_retention, 0.1)
    np.testing.assert_allclose(result.action44, np.zeros(44))


def test_soft_grasp_loss_retimes_phase_progress() -> None:
    executor = OnlineIntentChunkExecutor(
        IntentChunkExecutionConfig(
            hard_grasp_retention=0.2,
            soft_grasp_retention=0.8,
            enforce_grasp_retention=True,
        )
    )
    executor.start(_chunk(), _observation(fingertip_force_n=(2.0, 2.0)))

    result = executor.step(
        _observation(fingertip_force_n=(1.0, 2.0)),
        np.zeros(44),
    )

    assert result.active
    assert np.isclose(result.grasp_scale, 0.5)
    assert np.isclose(result.time_scale, 0.5)
    assert np.isclose(result.phase, 0.5)


def test_tracking_lag_retimes_without_skipping_chunk_motion() -> None:
    previous_command = np.zeros(44)
    previous_command[[0, 22]] = 0.009
    executor = OnlineIntentChunkExecutor(
        IntentChunkExecutionConfig(
            soft_tracking_translation_m=0.001,
            hard_tracking_translation_m=0.011,
        )
    )
    executor.start(_chunk(), _observation())

    result = executor.step(
        _observation(previous_action44=previous_command),
        np.zeros(44),
    )

    assert result.active
    assert np.isclose(result.tracking_scale, 0.2)
    assert np.isclose(result.phase, 0.2)
    np.testing.assert_allclose(
        result.action44[[0, 22]],
        [0.0002, 0.0002],
        atol=1e-9,
    )


def test_hard_tracking_error_stops_instead_of_stalling_active() -> None:
    previous_command = np.zeros(44)
    previous_command[[0, 22]] = 0.02
    executor = OnlineIntentChunkExecutor(
        IntentChunkExecutionConfig(
            soft_tracking_translation_m=0.001,
            hard_tracking_translation_m=0.01,
        )
    )
    executor.start(_chunk(), _observation())

    result = executor.step(
        _observation(previous_action44=previous_command),
        np.zeros(44),
    )

    assert not result.active
    assert result.outcome == "tracking_error_stop"
    assert result.tracking_scale == 0.0
    np.testing.assert_allclose(result.action44, np.zeros(44))


def test_force_allocation_preserves_relative_wrist_intent() -> None:
    chunk = np.zeros((2, 44), dtype=np.float64)
    chunk[1, 0] = 0.001
    chunk[1, 22] = -0.001
    executor = OnlineIntentChunkExecutor(
        IntentChunkExecutionConfig(
            contact_response_enabled=True,
            soft_force_limit_n=8.0,
            hard_force_limit_n=20.0,
            force_allocation_gain=0.2,
        )
    )
    executor.start(chunk, _observation())

    result = executor.step(
        _observation(side_force_n=(4.0, 0.0)),
        np.zeros(44),
    )

    assert not result.active
    assert result.outcome == "chunk_complete"
    assert np.isclose(result.right_motion_fraction, 0.4)
    np.testing.assert_allclose(result.action44[[0, 22]], [0.0008, -0.0012])
    assert np.isclose(result.action44[0] - result.action44[22], 0.002)


def test_zero_fingertip_baseline_is_audited_as_unobservable() -> None:
    executor = OnlineIntentChunkExecutor()
    executor.start(_chunk(), _observation())

    result = executor.step(_observation(), np.zeros(44))

    assert not result.grasp_observable
    assert result.minimum_grasp_retention == 1.0


def test_unvalidated_grasp_retention_does_not_stop_default_runtime() -> None:
    executor = OnlineIntentChunkExecutor()
    executor.start(_chunk(), _observation(fingertip_force_n=(2.0, 2.0)))

    result = executor.step(
        _observation(fingertip_force_n=(0.0, 2.0)),
        np.zeros(44),
    )

    assert result.active
    assert result.grasp_scale == 1.0
    assert result.minimum_grasp_retention == 0.0


def test_contact_response_adds_bounded_tangential_probe_to_relative_intent() -> None:
    chunk = np.zeros((3, 44), dtype=np.float64)
    chunk[:, 2] = [0.0, 0.001, 0.002]
    chunk[:, 24] = [0.0, -0.001, -0.002]
    executor = OnlineIntentChunkExecutor(
        IntentChunkExecutionConfig(
            contact_response_enabled=True,
            soft_force_limit_n=8.0,
            hard_force_limit_n=20.0,
            contact_force_threshold_n=2.0,
            contact_probe_step_m=0.0002,
        )
    )
    baseline = _observation(fingertip_force_n=(2.0, 2.0))
    executor.start(chunk, baseline)

    result = executor.step(
        _observation(force_n=3.0, fingertip_force_n=(2.0, 2.0)),
        np.zeros(44),
    )

    assert result.active
    assert result.grasp_observable
    assert result.contact_phase == "probe_0_1"
    assert np.isclose(result.contact_correction_m, 0.0002)
    relative_translation = result.action44[:3] - result.action44[22:25]
    assert np.linalg.norm(relative_translation[:2]) > 0.0
    assert relative_translation[2] > 0.0


def test_contact_response_uses_wrist_torque_for_bounded_relative_rotation() -> None:
    chunk = np.zeros((3, 44), dtype=np.float64)
    chunk[:, 2] = [0.0, 0.001, 0.002]
    chunk[:, 24] = [0.0, -0.001, -0.002]
    executor = OnlineIntentChunkExecutor(
        IntentChunkExecutionConfig(
            contact_response_enabled=True,
            contact_force_threshold_n=2.0,
            contact_rotation_compliance_rad_per_nm=0.00025,
            maximum_contact_rotation_correction_rad=0.0015,
        )
    )
    executor.start(
        chunk,
        _observation(fingertip_force_n=(2.0, 2.0)),
    )

    result = executor.step(
        _observation(
            force_n=3.0,
            fingertip_force_n=(2.0, 2.0),
            side_torque_nm=(2.0, -2.0),
        ),
        np.zeros(44),
    )

    assert np.isclose(result.contact_rotation_correction_rad, 0.001)
    relative_rotation = result.action44[3:6] - result.action44[25:28]
    assert np.linalg.norm(relative_rotation) > 0.0
    assert np.linalg.norm(relative_rotation) <= 0.001 + 1e-9
