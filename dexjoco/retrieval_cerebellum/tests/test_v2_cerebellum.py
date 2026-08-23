import numpy as np
import pytest

from retrieval_cerebellum.sensor_observation import CerebellumSensorObservation
from retrieval_cerebellum.v2_cerebellum import V2Cerebellum
from retrieval_cerebellum.v2_control import V2AssemblyEstimate, V2Mode


def _observation(timestamp=1.0):
    fingertip = np.zeros((2, 4, 3), dtype=np.float32)
    fingertip[:, :, 0] = 1.0
    return CerebellumSensorObservation(
        timestamp_s=timestamp,
        state46=np.zeros(46),
        arm_joint_torque=np.zeros((2, 7)),
        fingertip_force_world=fingertip,
        wrist_wrench_world=np.zeros((2, 6)),
        images={"wrist_right": np.zeros((4, 4, 3), dtype=np.uint8)},
    )


def _estimate(timestamp=1.0):
    return V2AssemblyEstimate(
        timestamp_s=timestamp,
        mean5=np.asarray([0.004, 0.0, 0.02, 0.0, -0.02]),
        covariance5=np.eye(5) * 1e-6,
        hole_rotation_world=np.eye(3),
        visual_reliability=1.0,
    )


def test_v2_cerebellum_generates_action_without_changing_fingers():
    cerebellum = V2Cerebellum()
    current = np.zeros(44, dtype=np.float32)
    current[6:22] = 0.4
    current[28:44] = -0.2

    result = cerebellum.step(_observation(), _estimate(), current)

    assert result.command.mode == V2Mode.ALIGN
    assert result.action44[0] < 0.0
    np.testing.assert_allclose(result.action44[6:22], current[6:22])
    np.testing.assert_allclose(result.action44[28:44], current[28:44])


def test_v2_cerebellum_rejects_stale_visual_estimate():
    cerebellum = V2Cerebellum()

    with pytest.raises(ValueError, match="not time aligned"):
        cerebellum.step(_observation(timestamp=1.0), _estimate(timestamp=0.5), np.zeros(44))
