import numpy as np

from retrieval_cerebellum.sensor_observation import CerebellumSensorObservation
from retrieval_cerebellum.v2_contact import V2ContactInterpreter
from retrieval_cerebellum.v2_control import V2AssemblyEstimate


def _estimate(depth=-0.01):
    return V2AssemblyEstimate(
        timestamp_s=0.0,
        mean5=np.asarray([0.0, 0.0, 0.0, 0.0, depth]),
        covariance5=np.eye(5) * 1e-6,
        hole_rotation_world=np.eye(3),
        visual_reliability=1.0,
    )


def _observation(fingertip_force=1.0, right_wrist_force=None):
    fingertip = np.zeros((2, 4, 3), dtype=np.float32)
    fingertip[:, :, 0] = fingertip_force
    wrist = np.zeros((2, 6), dtype=np.float32)
    if right_wrist_force is not None:
        wrist[0, :3] = right_wrist_force
    return CerebellumSensorObservation(
        timestamp_s=0.0,
        state46=np.zeros(46),
        arm_joint_torque=np.zeros((2, 7)),
        fingertip_force_world=fingertip,
        wrist_wrench_world=wrist,
        images={},
    )


def test_contact_interpreter_uses_wrench_residual_for_rim_contact():
    interpreter = V2ContactInterpreter()
    interpreter.update(
        _observation(right_wrist_force=[5.0, 0.0, 0.0]),
        _estimate(),
        allow_baseline_update=True,
    )

    signals = interpreter.update(
        _observation(right_wrist_force=[9.0, 0.0, 0.0]),
        _estimate(),
        allow_baseline_update=False,
    )

    assert signals.rim_contact
    assert not signals.jammed
    assert signals.right_grasp_stability == 1.0


def test_contact_interpreter_detects_grasp_force_drop_as_slip():
    interpreter = V2ContactInterpreter()
    interpreter.update(
        _observation(fingertip_force=2.0),
        _estimate(),
        allow_baseline_update=True,
    )

    signals = interpreter.update(
        _observation(fingertip_force=0.1),
        _estimate(),
        allow_baseline_update=False,
    )

    assert signals.right_slip
    assert signals.left_slip
    assert signals.right_grasp_stability == 0.0


def test_contact_interpreter_requires_depth_for_bottom_contact():
    interpreter = V2ContactInterpreter()
    interpreter.update(_observation(), _estimate(), allow_baseline_update=True)

    shallow = interpreter.update(
        _observation(right_wrist_force=[0.0, 0.0, 14.0]),
        _estimate(depth=-0.001),
        allow_baseline_update=False,
    )
    deep = interpreter.update(
        _observation(right_wrist_force=[0.0, 0.0, 14.0]),
        _estimate(depth=0.01),
        allow_baseline_update=False,
    )

    assert not shallow.bottom_contact
    assert deep.bottom_contact
