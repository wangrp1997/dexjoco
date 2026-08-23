"""Kinematic preflight and safety checks for short MuJoCo skill execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .assembly_kinematics import pose_from_action44, world_wrist_twist
from .skill_prototype import DataDrivenActionLimits
from .sqp_skill_adapter import SuccessfulInsertionSkillRecord


@dataclass(frozen=True)
class KinematicPreflightConfig:
    damping: float = 1e-3
    max_condition_number: float = 200.0
    max_joint_velocity_rad_s: float = 2.0
    control_dt_s: float = 0.02
    joint_limit_margin_rad: float = 0.02
    max_position_residual_m: float = 5e-4
    max_rotation_residual_rad: float = 5e-3

    def __post_init__(self) -> None:
        for name in (
            "damping",
            "max_condition_number",
            "max_joint_velocity_rad_s",
            "control_dt_s",
            "joint_limit_margin_rad",
            "max_position_residual_m",
            "max_rotation_residual_rad",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ArmKinematicCheck:
    side: str
    safe: bool
    reasons: tuple[str, ...]
    position_step_m: float
    rotation_step_rad: float
    jacobian_condition: float
    estimated_joint_step_rad: tuple[float, ...]
    estimated_joint_velocity_rad_s: tuple[float, ...]
    position_residual_m: float
    rotation_residual_rad: float
    minimum_joint_margin_rad: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BimanualKinematicCheck:
    safe: bool
    reasons: tuple[str, ...]
    right: ArmKinematicCheck
    left: ArmKinematicCheck
    maximum_finger_step_rad: float

    def to_dict(self) -> dict:
        return {
            "safe": self.safe,
            "reasons": list(self.reasons),
            "right": self.right.to_dict(),
            "left": self.left.to_dict(),
            "maximum_finger_step_rad": self.maximum_finger_step_rad,
        }


@dataclass(frozen=True)
class ExecutionSafetyLimits:
    right_force_norm_max_n: float
    left_force_norm_max_n: float
    right_torque_norm_max_nm: float
    left_torque_norm_max_nm: float
    maximum_depth_regression_m: float
    maximum_depth_advance_m: float
    maximum_lateral_increase_m: float
    right_attachment_translation_step_m: float
    left_attachment_translation_step_m: float
    right_attachment_rotation_step_rad: float
    left_attachment_rotation_step_rad: float

    @classmethod
    def from_successful_records(
        cls,
        records: list[SuccessfulInsertionSkillRecord],
        *,
        quantile: float = 0.95,
        confidence_multiplier: float = 1.0,
    ) -> ExecutionSafetyLimits:
        if not records:
            raise ValueError("successful records must be non-empty")
        if not 0.5 <= quantile < 1.0:
            raise ValueError("quantile must be in [0.5, 1.0)")

        def envelope(values: np.ndarray, floor: float) -> float:
            array = np.asarray(values, dtype=np.float64).reshape(-1)
            median = float(np.median(array))
            robust_std = 1.4826 * float(np.median(np.abs(array - median)))
            return max(
                floor,
                float(np.quantile(array, quantile))
                + confidence_multiplier * max(robust_std, 1e-9),
            )

        right_wrench = np.concatenate([record.right_wrenches for record in records])
        left_wrench = np.concatenate([record.left_wrenches for record in records])
        depth_regressions = []
        depth_advances = []
        lateral_increases = []
        right_attachment_translation = []
        left_attachment_translation = []
        right_attachment_rotation = []
        left_attachment_rotation = []
        for record in records:
            depth_delta = np.diff(record.states[:, 4])
            depth_regressions.extend(np.maximum(-depth_delta, 0.0).tolist())
            depth_advances.extend(np.maximum(depth_delta, 0.0).tolist())
            lateral = np.linalg.norm(record.states[:, :2], axis=1)
            lateral_increases.extend(np.maximum(np.diff(lateral), 0.0).tolist())
            features = record.geometry_features
            right_attachment_translation.extend(
                np.linalg.norm(np.diff(features[:, 6:9], axis=0), axis=1).tolist()
            )
            left_attachment_translation.extend(
                np.linalg.norm(np.diff(features[:, 12:15], axis=0), axis=1).tolist()
            )
            right_rotation = Rotation.from_rotvec(features[:, 9:12])
            left_rotation = Rotation.from_rotvec(features[:, 15:18])
            right_attachment_rotation.extend(
                (right_rotation[1:] * right_rotation[:-1].inv()).magnitude().tolist()
            )
            left_attachment_rotation.extend(
                (left_rotation[1:] * left_rotation[:-1].inv()).magnitude().tolist()
            )
        return cls(
            right_force_norm_max_n=envelope(
                np.linalg.norm(right_wrench[:, :3], axis=1),
                1.0,
            ),
            left_force_norm_max_n=envelope(
                np.linalg.norm(left_wrench[:, :3], axis=1),
                1.0,
            ),
            right_torque_norm_max_nm=envelope(
                np.linalg.norm(right_wrench[:, 3:], axis=1),
                0.1,
            ),
            left_torque_norm_max_nm=envelope(
                np.linalg.norm(left_wrench[:, 3:], axis=1),
                0.1,
            ),
            maximum_depth_regression_m=envelope(
                np.asarray(depth_regressions),
                5e-4,
            ),
            maximum_depth_advance_m=envelope(
                np.asarray(depth_advances),
                5e-4,
            ),
            maximum_lateral_increase_m=envelope(
                np.asarray(lateral_increases),
                5e-4,
            ),
            right_attachment_translation_step_m=envelope(
                np.asarray(right_attachment_translation),
                5e-4,
            ),
            left_attachment_translation_step_m=envelope(
                np.asarray(left_attachment_translation),
                5e-4,
            ),
            right_attachment_rotation_step_rad=envelope(
                np.asarray(right_attachment_rotation),
                5e-3,
            ),
            left_attachment_rotation_step_rad=envelope(
                np.asarray(left_attachment_rotation),
                5e-3,
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _arm_joint_addresses(model, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = np.asarray(
        [int(model.joint(f"joint{index}_{side}").id) for index in range(1, 8)],
        dtype=np.int64,
    )
    qpos_addresses = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int64)
    dof_addresses = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int64)
    return joint_ids, qpos_addresses, dof_addresses


def _arm_kinematic_check(
    raw_env,
    target_action44: np.ndarray,
    *,
    side: str,
    position_step_limit_m: float,
    rotation_step_limit_rad: float,
    config: KinematicPreflightConfig,
) -> ArmKinematicCheck:
    model = raw_env._model
    data = raw_env._data
    site_id = int(model.site(f"attachment_site_{side}").id)
    current_position = np.asarray(data.site_xpos[site_id], dtype=np.float64)
    current_rotation = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    target_pose = pose_from_action44(target_action44, side=side)
    position_delta = target_pose[:3, 3] - current_position
    rotation_delta = target_pose[:3, :3] @ current_rotation.T
    rotation_vector = Rotation.from_matrix(rotation_delta).as_rotvec()
    desired_twist = np.concatenate([position_delta, rotation_vector])

    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(
        model,
        data,
        jacobian_position,
        jacobian_rotation,
        site_id,
    )
    joint_ids, qpos_addresses, dof_addresses = _arm_joint_addresses(model, side)
    jacobian = np.concatenate(
        [
            jacobian_position[:, dof_addresses],
            jacobian_rotation[:, dof_addresses],
        ],
        axis=0,
    )
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(
        np.inf
        if singular_values[-1] <= 1e-12
        else singular_values[0] / singular_values[-1]
    )
    regularized = jacobian @ jacobian.T
    regularized += np.eye(6, dtype=np.float64) * config.damping**2
    joint_step = jacobian.T @ np.linalg.solve(regularized, desired_twist)
    residual = desired_twist - jacobian @ joint_step
    joint_velocity = joint_step / config.control_dt_s

    qpos = np.asarray(data.qpos[qpos_addresses], dtype=np.float64)
    next_qpos = qpos + joint_step
    margins = []
    joint_limit_violation = False
    for joint_id, value in zip(joint_ids, next_qpos, strict=True):
        if not bool(model.jnt_limited[joint_id]):
            continue
        lower, upper = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
        margin = min(value - lower, upper - value)
        margins.append(float(margin))
        joint_limit_violation |= margin < config.joint_limit_margin_rad
    minimum_margin = float(np.inf if not margins else min(margins))

    position_step = float(np.linalg.norm(position_delta))
    rotation_step = float(np.linalg.norm(rotation_vector))
    position_residual = float(np.linalg.norm(residual[:3]))
    rotation_residual = float(np.linalg.norm(residual[3:]))
    reasons = []
    if position_step > position_step_limit_m + 1e-9:
        reasons.append("position_step_exceeds_success_envelope")
    if rotation_step > rotation_step_limit_rad + 1e-9:
        reasons.append("rotation_step_exceeds_success_envelope")
    if not np.isfinite(condition) or condition > config.max_condition_number:
        reasons.append("jacobian_condition_exceeds_limit")
    if float(np.max(np.abs(joint_velocity))) > config.max_joint_velocity_rad_s:
        reasons.append("estimated_joint_velocity_exceeds_limit")
    if joint_limit_violation:
        reasons.append("estimated_joint_position_near_limit")
    if position_residual > config.max_position_residual_m:
        reasons.append("position_residual_exceeds_limit")
    if rotation_residual > config.max_rotation_residual_rad:
        reasons.append("rotation_residual_exceeds_limit")
    return ArmKinematicCheck(
        side=side,
        safe=not reasons,
        reasons=tuple(reasons),
        position_step_m=position_step,
        rotation_step_rad=rotation_step,
        jacobian_condition=condition,
        estimated_joint_step_rad=tuple(float(value) for value in joint_step),
        estimated_joint_velocity_rad_s=tuple(float(value) for value in joint_velocity),
        position_residual_m=position_residual,
        rotation_residual_rad=rotation_residual,
        minimum_joint_margin_rad=minimum_margin,
    )


def check_bimanual_action_kinematics(
    raw_env,
    current_action44: np.ndarray,
    target_action44: np.ndarray,
    action_limits: DataDrivenActionLimits,
    config: KinematicPreflightConfig | None = None,
) -> BimanualKinematicCheck:
    cfg = config or KinematicPreflightConfig()
    current = np.asarray(current_action44, dtype=np.float64).reshape(44)
    target = np.asarray(target_action44, dtype=np.float64).reshape(44)
    right = _arm_kinematic_check(
        raw_env,
        target,
        side="right",
        position_step_limit_m=action_limits.right_position_step_m,
        rotation_step_limit_rad=action_limits.right_rotation_step_rad,
        config=cfg,
    )
    left = _arm_kinematic_check(
        raw_env,
        target,
        side="left",
        position_step_limit_m=action_limits.left_position_step_m,
        rotation_step_limit_rad=action_limits.left_rotation_step_rad,
        config=cfg,
    )
    finger_step = np.abs(
        np.concatenate(
            [target[6:22] - current[6:22], target[28:44] - current[28:44]]
        )
    )
    finger_excess = finger_step - np.asarray(action_limits.finger_step_rad)
    reasons = [f"right:{reason}" for reason in right.reasons]
    reasons.extend(f"left:{reason}" for reason in left.reasons)
    if np.any(finger_excess > 1e-9):
        reasons.append("finger_step_exceeds_success_envelope")
    return BimanualKinematicCheck(
        safe=not reasons,
        reasons=tuple(reasons),
        right=right,
        left=left,
        maximum_finger_step_rad=float(finger_step.max(initial=0.0)),
    )


def command_increment_safety_reasons(
    current_action44: np.ndarray,
    target_action44: np.ndarray,
    action_limits: DataDrivenActionLimits,
) -> tuple[str, ...]:
    current = np.asarray(current_action44, dtype=np.float64).reshape(44)
    target = np.asarray(target_action44, dtype=np.float64).reshape(44)
    reasons = []
    for side, position_limit, rotation_limit in (
        (
            "right",
            action_limits.right_position_step_m,
            action_limits.right_rotation_step_rad,
        ),
        (
            "left",
            action_limits.left_position_step_m,
            action_limits.left_rotation_step_rad,
        ),
    ):
        twist = world_wrist_twist(
            pose_from_action44(current, side=side),
            pose_from_action44(target, side=side),
        )
        if float(np.linalg.norm(twist[:3])) > position_limit + 1e-9:
            reasons.append(f"{side}:command_position_step_exceeds_success_envelope")
        if float(np.linalg.norm(twist[3:])) > rotation_limit + 1e-9:
            reasons.append(f"{side}:command_rotation_step_exceeds_success_envelope")
    finger_step = np.abs(
        np.concatenate(
            [target[6:22] - current[6:22], target[28:44] - current[28:44]]
        )
    )
    if np.any(finger_step - np.asarray(action_limits.finger_step_rad) > 1e-9):
        reasons.append("finger_step_exceeds_success_envelope")
    return tuple(reasons)


def execution_safety_reasons(
    *,
    before_state5: np.ndarray,
    after_state5: np.ndarray,
    before_peg_ok: bool,
    after_peg_ok: bool,
    before_tray_ok: bool,
    after_tray_ok: bool,
    before_right_attachment6: np.ndarray,
    after_right_attachment6: np.ndarray,
    before_left_attachment6: np.ndarray,
    after_left_attachment6: np.ndarray,
    wrist_ft_right: np.ndarray,
    wrist_ft_left: np.ndarray,
    limits: ExecutionSafetyLimits,
) -> tuple[str, ...]:
    before = np.asarray(before_state5, dtype=np.float64).reshape(5)
    after = np.asarray(after_state5, dtype=np.float64).reshape(5)
    right = np.asarray(wrist_ft_right, dtype=np.float64).reshape(6)
    left = np.asarray(wrist_ft_left, dtype=np.float64).reshape(6)
    before_right_attachment = np.asarray(
        before_right_attachment6,
        dtype=np.float64,
    ).reshape(6)
    after_right_attachment = np.asarray(
        after_right_attachment6,
        dtype=np.float64,
    ).reshape(6)
    before_left_attachment = np.asarray(
        before_left_attachment6,
        dtype=np.float64,
    ).reshape(6)
    after_left_attachment = np.asarray(
        after_left_attachment6,
        dtype=np.float64,
    ).reshape(6)
    reasons = []
    if before_peg_ok and not after_peg_ok:
        reasons.append("right_grasp_contact_lost")
    if before_tray_ok and not after_tray_ok:
        reasons.append("left_grasp_contact_lost")
    right_attachment_translation = float(
        np.linalg.norm(after_right_attachment[:3] - before_right_attachment[:3])
    )
    left_attachment_translation = float(
        np.linalg.norm(after_left_attachment[:3] - before_left_attachment[:3])
    )
    right_attachment_rotation = float(
        (
            Rotation.from_rotvec(after_right_attachment[3:])
            * Rotation.from_rotvec(before_right_attachment[3:]).inv()
        ).magnitude()
    )
    left_attachment_rotation = float(
        (
            Rotation.from_rotvec(after_left_attachment[3:])
            * Rotation.from_rotvec(before_left_attachment[3:]).inv()
        ).magnitude()
    )
    if right_attachment_translation > limits.right_attachment_translation_step_m:
        reasons.append("right_attachment_translation_slip")
    if left_attachment_translation > limits.left_attachment_translation_step_m:
        reasons.append("left_attachment_translation_slip")
    if right_attachment_rotation > limits.right_attachment_rotation_step_rad:
        reasons.append("right_attachment_rotation_slip")
    if left_attachment_rotation > limits.left_attachment_rotation_step_rad:
        reasons.append("left_attachment_rotation_slip")
    if np.linalg.norm(right[:3]) > limits.right_force_norm_max_n:
        reasons.append("right_wrist_force_exceeds_success_envelope")
    if np.linalg.norm(left[:3]) > limits.left_force_norm_max_n:
        reasons.append("left_wrist_force_exceeds_success_envelope")
    if np.linalg.norm(right[3:]) > limits.right_torque_norm_max_nm:
        reasons.append("right_wrist_torque_exceeds_success_envelope")
    if np.linalg.norm(left[3:]) > limits.left_torque_norm_max_nm:
        reasons.append("left_wrist_torque_exceeds_success_envelope")
    if before[4] - after[4] > limits.maximum_depth_regression_m:
        reasons.append("insertion_depth_regressed")
    if after[4] - before[4] > limits.maximum_depth_advance_m:
        reasons.append("insertion_depth_advanced_too_far")
    before_lateral = float(np.linalg.norm(before[:2]))
    after_lateral = float(np.linalg.norm(after[:2]))
    if after_lateral - before_lateral > limits.maximum_lateral_increase_m:
        reasons.append("lateral_error_increased")
    return tuple(reasons)
