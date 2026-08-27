"""Fail-closed public sensor and action adapters for ECHO-Insert."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def _checked(value: np.ndarray, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=np.float64).copy()
    result.flags.writeable = False
    return result


def _checked_ego_depth(value: np.ndarray) -> np.ndarray:
    depth = np.asarray(value, dtype=np.float32)
    if depth.shape != (640, 640):
        raise ValueError(f"ego_depth_m must have shape (640, 640), got {depth.shape}")
    finite = np.isfinite(depth)
    if np.isnan(depth).any() or np.isneginf(depth).any() or not finite.any():
        raise ValueError("ego_depth_m must contain positive metric depth or +inf")
    if np.any(depth[finite] <= 0.0):
        raise ValueError("finite ego_depth_m values must be positive")
    result = depth.copy()
    result.flags.writeable = False
    return result


def _rotation_from_wxyz(quaternion: np.ndarray, *, name: str) -> Rotation:
    quat = _checked(quaternion, (4,), name=name)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError(f"{name} must have non-zero norm")
    return Rotation.from_quat(quat / norm, scalar_first=True)


def _basis(value: np.ndarray, *, name: str) -> np.ndarray:
    basis = _checked(value, (3, 3), name=name)
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(basis), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} must be right-handed")
    return basis


@dataclass(frozen=True, slots=True)
class PublicObservation:
    """The complete allowlisted observation visible to the online controller."""

    state46: np.ndarray
    previous_action44: np.ndarray
    wrist_wrench_local: np.ndarray
    fingertip_load: np.ndarray | None = None
    ego_depth_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state46",
            _readonly(_checked(self.state46, (46,), name="state46")),
        )
        object.__setattr__(
            self,
            "previous_action44",
            _readonly(
                _checked(self.previous_action44, (44,), name="previous_action44")
            ),
        )
        object.__setattr__(
            self,
            "wrist_wrench_local",
            _readonly(
                _checked(self.wrist_wrench_local, (2, 6), name="wrist_wrench_local")
            ),
        )
        if self.fingertip_load is not None:
            object.__setattr__(
                self,
                "fingertip_load",
                _readonly(
                    _checked(self.fingertip_load, (2, 4), name="fingertip_load")
                ),
            )
        if self.ego_depth_m is not None:
            object.__setattr__(self, "ego_depth_m", _checked_ego_depth(self.ego_depth_m))


def state46_to_action44(state46: np.ndarray) -> np.ndarray:
    """Convert public dual-arm proprioception to the absolute rotvec action layout."""
    state = _checked(state46, (46,), name="state46")
    right_rotation = _rotation_from_wxyz(state[3:7], name="right TCP quaternion")
    left_rotation = _rotation_from_wxyz(state[10:14], name="left TCP quaternion")
    action = np.concatenate(
        [
            state[0:3],
            right_rotation.as_rotvec(),
            state[14:30],
            state[7:10],
            left_rotation.as_rotvec(),
            state[30:46],
        ]
    )
    return _readonly(action)


def checked_task_basis(value: np.ndarray) -> np.ndarray:
    """Validate and freeze a world-frame task basis."""
    return _readonly(_basis(value, name="basis_world"))


def wrist_wrench_task(
    observation: PublicObservation,
    basis_world: np.ndarray,
) -> np.ndarray:
    """Rotate local right/left wrist `[force, torque]` vectors into task axes."""
    if not isinstance(observation, PublicObservation):
        raise TypeError("observation must be a PublicObservation")
    task_world = _basis(basis_world, name="basis_world")
    rotations = (
        _rotation_from_wxyz(
            observation.state46[3:7],
            name="right TCP quaternion",
        ).as_matrix(),
        _rotation_from_wxyz(
            observation.state46[10:14],
            name="left TCP quaternion",
        ).as_matrix(),
    )
    result = np.empty((2, 6), dtype=np.float64)
    for side, wrist_world in enumerate(rotations):
        local = observation.wrist_wrench_local[side]
        result[side, 0:3] = task_world.T @ (wrist_world @ local[0:3])
        result[side, 3:6] = task_world.T @ (wrist_world @ local[3:6])
    return _readonly(result)


def apply_right_micro_action(
    previous_action44: np.ndarray,
    frozen_action44: np.ndarray,
    u5: np.ndarray,
    basis_world: np.ndarray,
) -> np.ndarray:
    """Integrate `[tx, ty, advance, roll, pitch]` and freeze every other target.

    The task basis' third axis is defined as the positive insertion direction.
    Rotation increments are task-frame axes expressed in world coordinates and
    are left-multiplied onto the previous right-TCP orientation.
    """
    previous = _checked(previous_action44, (44,), name="previous_action44")
    frozen = _checked(frozen_action44, (44,), name="frozen_action44")
    micro = _checked(u5, (5,), name="u5")
    task_world = _basis(basis_world, name="basis_world")

    translation_world = task_world @ micro[0:3]
    rotation_world = task_world @ np.asarray([micro[3], micro[4], 0.0])
    previous_rotation = Rotation.from_rotvec(previous[3:6])
    updated_rotation = Rotation.from_rotvec(rotation_world) * previous_rotation

    action = frozen.copy()
    action[0:3] = previous[0:3] + translation_world
    action[3:6] = updated_rotation.as_rotvec()
    return _readonly(action)


def apply_right_tip_pivot_action(
    state46: np.ndarray,
    frozen_action44: np.ndarray,
    rotation_step_world: np.ndarray,
    pivot_world: np.ndarray,
    target_pivot_world: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate the wrist and move its attached pivot to an optional world target."""
    state = _checked(state46, (46,), name="state46")
    frozen = _checked(frozen_action44, (44,), name="frozen_action44")
    step = Rotation.from_rotvec(
        _checked(rotation_step_world, (3,), name="rotation_step_world")
    )
    pivot = _checked(pivot_world, (3,), name="pivot_world")
    target_pivot = (
        pivot
        if target_pivot_world is None
        else _checked(target_pivot_world, (3,), name="target_pivot_world")
    )
    wrist_position = state[0:3]
    wrist_rotation = _rotation_from_wxyz(
        state[3:7],
        name="right TCP quaternion",
    )

    action = frozen.copy()
    action[0:3] = target_pivot - step.apply(pivot - wrist_position)
    action[3:6] = (step * wrist_rotation).as_rotvec()
    return _readonly(action)
