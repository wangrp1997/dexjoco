import numpy as np

from retrieval_cerebellum.short_horizon_execution import (
    ExecutionSafetyLimits,
    command_increment_safety_reasons,
    execution_safety_reasons,
)
from retrieval_cerebellum.skill_prototype import DataDrivenActionLimits


def _limits() -> ExecutionSafetyLimits:
    return ExecutionSafetyLimits(
        right_force_norm_max_n=10.0,
        left_force_norm_max_n=10.0,
        right_torque_norm_max_nm=1.0,
        left_torque_norm_max_nm=1.0,
        maximum_depth_regression_m=0.001,
        maximum_depth_advance_m=0.001,
        maximum_lateral_increase_m=0.001,
        right_attachment_translation_step_m=0.001,
        left_attachment_translation_step_m=0.001,
        right_attachment_rotation_step_rad=0.01,
        left_attachment_rotation_step_rad=0.01,
    )


def test_existing_false_contact_flag_is_not_reported_as_new_grasp_loss():
    reasons = execution_safety_reasons(
        before_state5=np.zeros(5),
        after_state5=np.zeros(5),
        before_peg_ok=False,
        after_peg_ok=False,
        before_tray_ok=True,
        after_tray_ok=True,
        before_right_attachment6=np.zeros(6),
        after_right_attachment6=np.zeros(6),
        before_left_attachment6=np.zeros(6),
        after_left_attachment6=np.zeros(6),
        wrist_ft_right=np.zeros(6),
        wrist_ft_left=np.zeros(6),
        limits=_limits(),
    )

    assert reasons == ()


def test_attachment_translation_slip_is_stopped():
    after_right = np.zeros(6)
    after_right[0] = 0.002

    reasons = execution_safety_reasons(
        before_state5=np.zeros(5),
        after_state5=np.zeros(5),
        before_peg_ok=False,
        after_peg_ok=False,
        before_tray_ok=True,
        after_tray_ok=True,
        before_right_attachment6=np.zeros(6),
        after_right_attachment6=after_right,
        before_left_attachment6=np.zeros(6),
        after_left_attachment6=np.zeros(6),
        wrist_ft_right=np.zeros(6),
        wrist_ft_left=np.zeros(6),
        limits=_limits(),
    )

    assert "right_attachment_translation_slip" in reasons


def test_command_increment_safety_uses_command_delta():
    current = np.zeros(44)
    target = current.copy()
    target[0] = 2e-4
    limits = DataDrivenActionLimits(
        right_position_step_m=5e-4,
        left_position_step_m=5e-4,
        right_rotation_step_rad=5e-3,
        left_rotation_step_rad=5e-3,
        finger_step_rad=np.full(32, 0.01),
        action_min=np.full(44, -10.0),
        action_max=np.full(44, 10.0),
    )

    assert command_increment_safety_reasons(current, target, limits) == ()
