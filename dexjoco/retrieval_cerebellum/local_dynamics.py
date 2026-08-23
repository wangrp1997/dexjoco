"""Paired-rollout identification of local bimanual insertion dynamics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import mujoco
import numpy as np


ASSEMBLY_STATE_DIM = 5
WRIST_TWIST_DIM = 6


@dataclass(frozen=True)
class MujocoIntegrationSnapshot:
    state: np.ndarray
    env_step: int | None

    @classmethod
    def capture(cls, raw_env) -> MujocoIntegrationSnapshot:
        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(
            mujoco.mj_stateSize(raw_env._model, specification),
            dtype=np.float64,
        )
        mujoco.mj_getState(raw_env._model, raw_env._data, state, specification)
        return cls(
            state=state,
            env_step=getattr(raw_env, "env_step", None),
        )

    def restore(self, raw_env) -> None:
        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(raw_env._model, raw_env._data, self.state, specification)
        mujoco.mj_forward(raw_env._model, raw_env._data)
        if self.env_step is not None:
            raw_env.env_step = self.env_step


@dataclass(frozen=True)
class OneStepDynamicsLinearization:
    drift: np.ndarray
    right_state_jacobian: np.ndarray
    left_state_jacobian: np.ndarray
    right_even_residual: np.ndarray
    left_even_residual: np.ndarray
    translation_step_m: float
    rotation_step_rad: float
    rollout_steps: int = 1

    def __post_init__(self) -> None:
        shapes = {
            "drift": (ASSEMBLY_STATE_DIM,),
            "right_state_jacobian": (ASSEMBLY_STATE_DIM, WRIST_TWIST_DIM),
            "left_state_jacobian": (ASSEMBLY_STATE_DIM, WRIST_TWIST_DIM),
            "right_even_residual": (ASSEMBLY_STATE_DIM, WRIST_TWIST_DIM),
            "left_even_residual": (ASSEMBLY_STATE_DIM, WRIST_TWIST_DIM),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must contain finite values")
            object.__setattr__(self, name, value.copy())
        if self.translation_step_m <= 0.0 or self.rotation_step_rad <= 0.0:
            raise ValueError("finite-difference steps must be positive")
        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")

    @property
    def combined_state_jacobian(self) -> np.ndarray:
        return np.concatenate(
            [self.right_state_jacobian, self.left_state_jacobian],
            axis=1,
        )

    @property
    def maximum_even_residual(self) -> float:
        return float(
            max(
                np.max(np.abs(self.right_even_residual)),
                np.max(np.abs(self.left_even_residual)),
            )
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        for name in (
            "drift",
            "right_state_jacobian",
            "left_state_jacobian",
            "right_even_residual",
            "left_even_residual",
        ):
            payload[name] = getattr(self, name).tolist()
        payload["condition_number"] = float(
            np.linalg.cond(self.combined_state_jacobian)
        )
        payload["maximum_even_residual"] = self.maximum_even_residual
        return payload


def identify_one_step_dynamics(
    rollout_delta: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    translation_step_m: float = 2.5e-4,
    rotation_step_rad: float = 2.5e-3,
    rollout_steps: int = 1,
) -> OneStepDynamicsLinearization:
    """Identify drift and action response from reset-matched central differences.

    ``rollout_delta`` must restore the exact same pre-action state before each call
    and return the resulting five-dimensional one-step assembly-state change.
    """

    if translation_step_m <= 0.0 or rotation_step_rad <= 0.0:
        raise ValueError("finite-difference steps must be positive")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    zero = np.zeros(WRIST_TWIST_DIM, dtype=np.float64)
    drift = np.asarray(rollout_delta(zero, zero), dtype=np.float64).reshape(-1)
    if drift.shape != (ASSEMBLY_STATE_DIM,):
        raise ValueError("rollout_delta must return shape (5,)")
    steps = np.asarray(
        [translation_step_m] * 3 + [rotation_step_rad] * 3,
        dtype=np.float64,
    )

    jacobians = []
    residuals = []
    for side in ("right", "left"):
        jacobian = np.empty((ASSEMBLY_STATE_DIM, WRIST_TWIST_DIM), dtype=np.float64)
        even_residual = np.empty_like(jacobian)
        for column, step in enumerate(steps):
            perturbation = np.zeros(WRIST_TWIST_DIM, dtype=np.float64)
            perturbation[column] = step
            if side == "right":
                plus = rollout_delta(perturbation, zero)
                minus = rollout_delta(-perturbation, zero)
            else:
                plus = rollout_delta(zero, perturbation)
                minus = rollout_delta(zero, -perturbation)
            plus = np.asarray(plus, dtype=np.float64).reshape(ASSEMBLY_STATE_DIM)
            minus = np.asarray(minus, dtype=np.float64).reshape(ASSEMBLY_STATE_DIM)
            jacobian[:, column] = (plus - minus) / (2.0 * step)
            even_residual[:, column] = 0.5 * (plus + minus) - drift
        jacobians.append(jacobian)
        residuals.append(even_residual)

    return OneStepDynamicsLinearization(
        drift=drift,
        right_state_jacobian=jacobians[0],
        left_state_jacobian=jacobians[1],
        right_even_residual=residuals[0],
        left_even_residual=residuals[1],
        translation_step_m=translation_step_m,
        rotation_step_rad=rotation_step_rad,
        rollout_steps=rollout_steps,
    )
