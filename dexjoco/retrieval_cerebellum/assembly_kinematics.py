"""SE(3) kinematics for the five-dimensional bimanual assembly state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


ASSEMBLY_STATE_DIM = 5
WRIST_TWIST_DIM = 6
ACTION44_DIM = 44

_RIGHT_WRIST = slice(0, 6)
_RIGHT_HAND = slice(6, 22)
_LEFT_WRIST = slice(22, 28)
_LEFT_HAND = slice(28, 44)

_TCP_TO_PALM = np.asarray(
    [
        [0.0, 0.0, 1.0, 0.05],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.03],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _vector(value: np.ndarray, size: int, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def pose_matrix(position: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(
        _vector(rotvec, 3, name="rotvec")
    ).as_matrix()
    transform[:3, 3] = _vector(position, 3, name="position")
    return transform


def invert_pose(transform: np.ndarray) -> np.ndarray:
    pose = np.asarray(transform, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {pose.shape}")
    rotation = pose[:3, :3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ pose[:3, 3]
    return inverse


def pose_from_action44(action44: np.ndarray, *, side: str) -> np.ndarray:
    action = _vector(action44, ACTION44_DIM, name="action44")
    if side == "right":
        wrist = action[_RIGHT_WRIST]
    elif side == "left":
        wrist = action[_LEFT_WRIST]
    else:
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    return pose_matrix(wrist[:3], wrist[3:6])


def palm_pose_from_action44(action44: np.ndarray, *, side: str) -> np.ndarray:
    """Convert the policy TCP frame to the Allegro palm frame."""
    return pose_from_action44(action44, side=side) @ _TCP_TO_PALM


def perturb_world_pose(transform: np.ndarray, twist: np.ndarray) -> np.ndarray:
    """Apply independent world-frame translation and rotation increments."""
    pose = np.asarray(transform, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {pose.shape}")
    delta = _vector(twist, WRIST_TWIST_DIM, name="twist")
    perturbed = pose.copy()
    perturbed[:3, 3] += delta[:3]
    perturbed[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix() @ pose[:3, :3]
    return perturbed


def world_wrist_twist(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    start_pose = np.asarray(start, dtype=np.float64)
    end_pose = np.asarray(end, dtype=np.float64)
    if start_pose.shape != (4, 4) or end_pose.shape != (4, 4):
        raise ValueError("start and end must have shape (4, 4)")
    rotation_delta = end_pose[:3, :3] @ start_pose[:3, :3].T
    return np.concatenate(
        [
            end_pose[:3, 3] - start_pose[:3, 3],
            Rotation.from_matrix(rotation_delta).as_rotvec(),
        ]
    )


def wrist_twists_from_action44(actions44: np.ndarray, *, side: str) -> np.ndarray:
    actions = np.asarray(actions44, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION44_DIM:
        raise ValueError(f"actions44 must have shape (T, 44), got {actions.shape}")
    if actions.shape[0] < 2:
        raise ValueError("actions44 must contain at least two rows")
    poses = [pose_from_action44(action, side=side) for action in actions]
    return np.stack(
        [world_wrist_twist(start, end) for start, end in zip(poses[:-1], poses[1:])]
    )


def apply_bimanual_wrist_twists(
    current_action44: np.ndarray,
    right_twist: np.ndarray,
    left_twist: np.ndarray,
    *,
    finger_reference44: np.ndarray | None = None,
) -> np.ndarray:
    action = _vector(current_action44, ACTION44_DIM, name="current_action44")
    if finger_reference44 is not None:
        reference = _vector(
            finger_reference44,
            ACTION44_DIM,
            name="finger_reference44",
        )
        action[_RIGHT_HAND] = reference[_RIGHT_HAND]
        action[_LEFT_HAND] = reference[_LEFT_HAND]
    for side, wrist_slice, twist in (
        ("right", _RIGHT_WRIST, right_twist),
        ("left", _LEFT_WRIST, left_twist),
    ):
        updated = perturb_world_pose(
            pose_from_action44(action, side=side),
            twist,
        )
        action[wrist_slice.start : wrist_slice.start + 3] = updated[:3, 3]
        action[wrist_slice.start + 3 : wrist_slice.stop] = Rotation.from_matrix(
            updated[:3, :3]
        ).as_rotvec()
    return action.astype(np.float32)


def bimanual_controls_to_action44(
    current_action44: np.ndarray,
    right_controls: np.ndarray,
    left_controls: np.ndarray,
    *,
    finger_references44: np.ndarray | None = None,
) -> np.ndarray:
    right = np.asarray(right_controls, dtype=np.float64)
    left = np.asarray(left_controls, dtype=np.float64)
    if right.ndim != 2 or right.shape[1] != WRIST_TWIST_DIM:
        raise ValueError(f"right_controls must have shape (H, 6), got {right.shape}")
    if left.shape != right.shape:
        raise ValueError(f"left_controls must have shape {right.shape}, got {left.shape}")
    references = None
    if finger_references44 is not None:
        references = np.asarray(finger_references44, dtype=np.float64)
        if references.shape != (right.shape[0], ACTION44_DIM):
            raise ValueError(
                "finger_references44 must have shape "
                f"({right.shape[0]}, 44), got {references.shape}"
            )
    commands = []
    current = _vector(current_action44, ACTION44_DIM, name="current_action44")
    for step in range(right.shape[0]):
        current = apply_bimanual_wrist_twists(
            current,
            right[step],
            left[step],
            finger_reference44=None if references is None else references[step],
        )
        commands.append(current)
    return np.asarray(commands, dtype=np.float32)


def align_and_project_finger_references(
    current_action44: np.ndarray,
    nominal_actions44: np.ndarray,
    finger_step_limits_rad: np.ndarray,
) -> np.ndarray:
    """Align retrieved hand postures to the current grasp and limit each step."""
    current = _vector(current_action44, ACTION44_DIM, name="current_action44")
    nominal = np.asarray(nominal_actions44, dtype=np.float64)
    if nominal.ndim != 2 or nominal.shape[1] != ACTION44_DIM:
        raise ValueError(
            f"nominal_actions44 must have shape (H + 1, 44), got {nominal.shape}"
        )
    if nominal.shape[0] < 2:
        raise ValueError("nominal_actions44 must contain at least two rows")
    limits = _vector(
        finger_step_limits_rad,
        32,
        name="finger_step_limits_rad",
    )
    source_fingers = np.concatenate(
        [nominal[:, _RIGHT_HAND], nominal[:, _LEFT_HAND]],
        axis=1,
    )
    current_fingers = np.concatenate(
        [current[_RIGHT_HAND], current[_LEFT_HAND]],
    )
    aligned = source_fingers + (current_fingers - source_fingers[0])
    projected = np.empty_like(aligned[1:])
    previous = current_fingers.copy()
    for step, target in enumerate(aligned[1:]):
        previous = previous + np.clip(target - previous, -limits, limits)
        projected[step] = previous
    references = np.repeat(current[None, :], projected.shape[0], axis=0)
    references[:, _RIGHT_HAND] = projected[:, :16]
    references[:, _LEFT_HAND] = projected[:, 16:]
    return references.astype(np.float32)


@dataclass(frozen=True)
class BimanualAssemblyKinematics:
    """Rigid bilateral attachment model around one insertion belief."""

    peg_in_right_palm: np.ndarray
    hole_in_left_palm: np.ndarray
    peg_tip_in_peg: np.ndarray

    def __post_init__(self) -> None:
        for name in ("peg_in_right_palm", "hole_in_left_palm"):
            transform = np.asarray(getattr(self, name), dtype=np.float64)
            if transform.shape != (4, 4):
                raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}")
            if not np.isfinite(transform).all():
                raise ValueError(f"{name} must contain finite values")
            object.__setattr__(self, name, transform.copy())
        object.__setattr__(
            self,
            "peg_tip_in_peg",
            _vector(self.peg_tip_in_peg, 3, name="peg_tip_in_peg"),
        )

    @classmethod
    def from_observation(
        cls,
        *,
        right_palm_world: np.ndarray,
        left_palm_world: np.ndarray,
        peg_in_right_palm: np.ndarray,
        peg_tip_in_hole_position: np.ndarray,
        peg_in_hole_rotvec: np.ndarray,
        peg_tip_in_peg: np.ndarray,
    ) -> BimanualAssemblyKinematics:
        right = np.asarray(right_palm_world, dtype=np.float64)
        left = np.asarray(left_palm_world, dtype=np.float64)
        peg_attachment = np.asarray(peg_in_right_palm, dtype=np.float64)
        if right.shape != (4, 4) or left.shape != (4, 4):
            raise ValueError("palm transforms must have shape (4, 4)")
        if peg_attachment.shape != (4, 4):
            raise ValueError("peg_in_right_palm must have shape (4, 4)")
        tip_in_peg = _vector(peg_tip_in_peg, 3, name="peg_tip_in_peg")
        tip_in_hole = _vector(
            peg_tip_in_hole_position,
            3,
            name="peg_tip_in_hole_position",
        )
        peg_world = right @ peg_attachment
        peg_in_hole_rotation = Rotation.from_rotvec(
            _vector(peg_in_hole_rotvec, 3, name="peg_in_hole_rotvec")
        ).as_matrix()
        hole_rotation_world = peg_world[:3, :3] @ peg_in_hole_rotation.T
        tip_world = peg_world[:3, :3] @ tip_in_peg + peg_world[:3, 3]
        hole_position_world = tip_world - hole_rotation_world @ tip_in_hole
        hole_world = np.eye(4, dtype=np.float64)
        hole_world[:3, :3] = hole_rotation_world
        hole_world[:3, 3] = hole_position_world
        return cls(
            peg_in_right_palm=peg_attachment,
            hole_in_left_palm=invert_pose(left) @ hole_world,
            peg_tip_in_peg=tip_in_peg,
        )

    def assembly_state(
        self,
        right_palm_world: np.ndarray,
        left_palm_world: np.ndarray,
    ) -> np.ndarray:
        peg_world = np.asarray(right_palm_world, dtype=np.float64) @ self.peg_in_right_palm
        hole_world = np.asarray(left_palm_world, dtype=np.float64) @ self.hole_in_left_palm
        tip_world = peg_world[:3, :3] @ self.peg_tip_in_peg + peg_world[:3, 3]
        tip_in_hole = hole_world[:3, :3].T @ (tip_world - hole_world[:3, 3])
        peg_in_hole_rotation = hole_world[:3, :3].T @ peg_world[:3, :3]
        tilt = Rotation.from_matrix(peg_in_hole_rotation).as_rotvec()
        return np.asarray(
            [tip_in_hole[0], tip_in_hole[1], tilt[0], tilt[1], -tip_in_hole[2]],
            dtype=np.float64,
        )

    def state_jacobians(
        self,
        right_palm_world: np.ndarray,
        left_palm_world: np.ndarray,
        *,
        translation_step_m: float = 1e-6,
        rotation_step_rad: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        if translation_step_m <= 0.0 or rotation_step_rad <= 0.0:
            raise ValueError("finite-difference steps must be positive")
        right = np.asarray(right_palm_world, dtype=np.float64)
        left = np.asarray(left_palm_world, dtype=np.float64)
        steps = np.asarray(
            [translation_step_m] * 3 + [rotation_step_rad] * 3,
            dtype=np.float64,
        )

        def jacobian(side: str) -> np.ndarray:
            result = np.empty((ASSEMBLY_STATE_DIM, WRIST_TWIST_DIM), dtype=np.float64)
            for column, step in enumerate(steps):
                delta = np.zeros(WRIST_TWIST_DIM, dtype=np.float64)
                delta[column] = step
                if side == "right":
                    plus = self.assembly_state(perturb_world_pose(right, delta), left)
                    minus = self.assembly_state(perturb_world_pose(right, -delta), left)
                else:
                    plus = self.assembly_state(right, perturb_world_pose(left, delta))
                    minus = self.assembly_state(right, perturb_world_pose(left, -delta))
                result[:, column] = (plus - minus) / (2.0 * step)
            return result

        return jacobian("right"), jacobian("left")
