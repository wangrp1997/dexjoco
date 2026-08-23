import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from pathlib import Path

from retrieval_cerebellum.real_sensor_model import (
    SensorDegrader,
    SensorModelConfig,
)
from retrieval_cerebellum.sensor_observation import CerebellumSensorObservation


def _config(**overrides) -> SensorModelConfig:
    values = {
        "name": "test",
        "hardware_verified": False,
        "sample_rate_hz": 30.0,
        "latency_frames": 0,
        "wrist_position_noise_std_m": 0.0,
        "wrist_orientation_noise_std_rad": 0.0,
        "finger_joint_noise_std_rad": 0.0,
        "arm_torque_noise_std_nm": 0.0,
        "arm_torque_bias_walk_std_nm": 0.0,
        "arm_torque_resolution_nm": 0.0,
        "arm_torque_limit_nm": 100.0,
        "fingertip_force_noise_std_n": 0.0,
        "fingertip_force_bias_walk_std_n": 0.0,
        "fingertip_force_resolution_n": 0.0,
        "fingertip_force_limit_n": 100.0,
        "fingertip_contact_threshold_n": 0.5,
        "wrist_force_noise_std_n": 0.0,
        "wrist_torque_noise_std_nm": 0.0,
        "wrist_force_resolution_n": 0.0,
        "wrist_torque_resolution_nm": 0.0,
        "wrist_force_limit_n": 100.0,
        "wrist_torque_limit_nm": 100.0,
        "proprio_dropout_probability": 0.0,
        "arm_torque_dropout_probability": 0.0,
        "fingertip_dropout_probability": 0.0,
        "wrist_wrench_dropout_probability": 0.0,
        "random_seed": 3,
    }
    values.update(overrides)
    return SensorModelConfig(**values)


def _observation(*, timestamp: float = 0.0) -> CerebellumSensorObservation:
    state = np.zeros(46, dtype=np.float32)
    state[3] = 1.0
    state[10] = 1.0
    fingertip = np.zeros((2, 4, 3), dtype=np.float32)
    fingertip[0, 0] = [3.0, 4.0, 0.0]
    fingertip[1, 2] = [0.0, 0.0, 0.4]
    return CerebellumSensorObservation(
        timestamp_s=timestamp,
        state46=state,
        arm_joint_torque=np.arange(14, dtype=np.float32).reshape(2, 7),
        fingertip_force_world=fingertip,
        wrist_wrench_world=np.arange(12, dtype=np.float32).reshape(2, 6),
        images={},
        previous_action44=np.zeros(44, dtype=np.float32),
    )


def test_identity_profile_reduces_tactile_to_magnitude_and_contact():
    degraded = SensorDegrader(_config()).transform(_observation())

    assert degraded is not None
    assert not hasattr(degraded, "fingertip_force_world")
    assert degraded.fingertip_force_magnitude[0, 0] == pytest.approx(5.0)
    assert degraded.fingertip_contact[0, 0]
    assert not degraded.fingertip_contact[1, 2]
    np.testing.assert_allclose(
        degraded.arm_joint_torque.reshape(-1),
        np.arange(14),
    )


def test_wrist_wrench_is_expressed_in_local_sensor_frame():
    observation = _observation()
    state = np.array(observation.state46, copy=True)
    state[3:7] = Rotation.from_euler("z", 90, degrees=True).as_quat(scalar_first=True)
    rotated = CerebellumSensorObservation(
        timestamp_s=0.0,
        state46=state,
        arm_joint_torque=observation.arm_joint_torque,
        fingertip_force_world=observation.fingertip_force_world,
        wrist_wrench_world=np.array(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.zeros(6)],
            dtype=np.float32,
        ),
        images={},
        previous_action44=observation.previous_action44,
    )

    degraded = SensorDegrader(_config()).transform(rotated)

    np.testing.assert_allclose(degraded.wrist_wrench_local[0, :3], [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(degraded.wrist_wrench_local[0, 3:], [1.0, 0.0, 0.0], atol=1e-6)


def test_latency_and_dropout_are_explicit():
    degrader = SensorDegrader(
        _config(
            latency_frames=1,
            arm_torque_dropout_probability=1.0,
            fingertip_dropout_probability=1.0,
            wrist_wrench_dropout_probability=1.0,
        )
    )

    assert degrader.transform(_observation(timestamp=0.0)) is None
    degraded = degrader.transform(_observation(timestamp=1.0))

    assert degraded.timestamp_s == 0.0
    assert not degraded.arm_torque_valid.any()
    assert not degraded.fingertip_valid.any()
    assert not degraded.wrist_wrench_valid.any()
    assert np.isfinite(degraded.arm_joint_torque).all()


def test_unverified_profile_cannot_claim_deployment_readiness():
    with pytest.raises(RuntimeError, match="not hardware verified"):
        _config().require_hardware_verified()


def test_committed_stress_profile_is_explicitly_unverified():
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "sensor_profiles"
        / "sim_stress_v1.json"
    )

    profile = SensorModelConfig.from_json(path)

    assert profile.name == "sim_stress_v1_unverified"
    assert not profile.hardware_verified
    assert profile.latency_s == pytest.approx(1.0 / 30.0)
