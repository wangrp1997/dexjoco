import numpy as np

from retrieval_cerebellum.v2_control import (
    ModeGatedCompliantController,
    V2AssemblyEstimate,
    V2ContactSignals,
    V2ControllerConfig,
    V2Mode,
)


def _estimate(state5, reliability=1.0):
    return V2AssemblyEstimate(
        timestamp_s=1.0,
        mean5=np.asarray(state5, dtype=np.float64),
        covariance5=np.eye(5) * 1e-6,
        hole_rotation_world=np.eye(3),
        visual_reliability=reliability,
    )


def _contact(**overrides):
    values = {
        "right_wrist_wrench_world": np.zeros(6),
        "left_wrist_wrench_world": np.zeros(6),
        "right_grasp_stability": 1.0,
        "left_grasp_stability": 1.0,
    }
    values.update(overrides)
    return V2ContactSignals(**values)


def test_align_correction_reduces_lateral_and_tilt_error():
    controller = ModeGatedCompliantController()

    command = controller.step(_estimate([0.004, -0.002, 0.04, -0.02, -0.02]), _contact())

    assert command.mode == V2Mode.ALIGN
    assert command.right_twist_world[0] < 0.0
    assert command.right_twist_world[1] > 0.0
    assert command.right_twist_world[3] < 0.0
    assert command.right_twist_world[4] > 0.0


def test_aligned_state_enters_guarded_descent_after_confirmation():
    controller = ModeGatedCompliantController(
        V2ControllerConfig(transition_confirmation_steps=2)
    )
    estimate = _estimate([0.0005, 0.0, 0.01, 0.0, -0.02])

    first = controller.step(estimate, _contact())
    second = controller.step(estimate, _contact())

    assert first.mode == V2Mode.ALIGN
    assert second.mode == V2Mode.GUARDED_DESCENT
    assert second.right_twist_world[2] < 0.0


def test_rim_contact_switches_to_force_relief():
    controller = ModeGatedCompliantController()
    contact = _contact(
        rim_contact=True,
        right_wrist_wrench_world=np.asarray([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )

    command = controller.step(_estimate([0.001, 0.0, 0.01, 0.0, -0.01]), contact)

    assert command.mode == V2Mode.CONTACT_CORRECTION
    assert command.right_twist_world[0] < 0.0
    assert command.right_twist_world[2] == 0.0


def test_bilateral_contact_retreats_instead_of_pushing_down():
    controller = ModeGatedCompliantController()

    command = controller.step(
        _estimate([0.0, 0.0, 0.0, 0.0, -0.005]),
        _contact(bilateral_contact=True),
    )

    assert command.mode == V2Mode.RETREAT
    assert command.right_twist_world[2] > 0.0


def test_low_visual_reliability_retreats():
    controller = ModeGatedCompliantController()

    command = controller.step(
        _estimate([0.0, 0.0, 0.0, 0.0, -0.02], reliability=0.2),
        _contact(),
    )

    assert command.mode == V2Mode.RETREAT


def test_slip_requests_regrasp_without_motion():
    controller = ModeGatedCompliantController()

    command = controller.step(
        _estimate([0.0, 0.0, 0.0, 0.0, -0.02]),
        _contact(right_slip=True),
    )

    assert command.mode == V2Mode.REGRASP_REQUEST
    np.testing.assert_allclose(command.right_twist_world, 0.0)
    np.testing.assert_allclose(command.left_twist_world, 0.0)


def test_target_depth_reports_success():
    controller = ModeGatedCompliantController()

    command = controller.step(
        _estimate([0.0, 0.0, 0.0, 0.0, 0.013]),
        _contact(),
    )

    assert command.mode == V2Mode.SUCCESS
