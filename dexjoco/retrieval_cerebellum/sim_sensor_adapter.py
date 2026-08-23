"""Trusted simulation boundary exposing only real-deployable sensor channels."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .sensor_observation import CerebellumSensorObservation


_FINGER_TIP_BODIES = (
    (
        "ff_tip_right",
        "mf_tip_right",
        "rf_tip_right",
        "th_tip_right",
    ),
    (
        "ff_tip_left",
        "mf_tip_left",
        "rf_tip_left",
        "th_tip_left",
    ),
)
_WRIST_SITES = ("attachment_site_right", "attachment_site_left")
_WRIST_FORCE_SENSORS = (
    "panda/wrist_force_right",
    "panda/wrist_force_left",
)
_WRIST_TORQUE_SENSORS = (
    "panda/wrist_torque_right",
    "panda/wrist_torque_left",
)
_ARM_JOINT_NAMES = tuple(
    tuple(f"joint{joint}_{side}" for joint in range(1, 8))
    for side in ("right", "left")
)


class SimCerebellumSensorAdapter:
    """Read robot/tactile sensors without reading object pose or task truth."""

    def __init__(self, raw_env) -> None:
        model = raw_env._model
        self._data = raw_env._data
        self._finger_body_ids = tuple(
            tuple(int(model.body(name).id) for name in names)
            for names in _FINGER_TIP_BODIES
        )
        self._wrist_site_ids = tuple(int(model.site(name).id) for name in _WRIST_SITES)
        self._arm_dof_addresses = tuple(
            tuple(int(model.joint(name).dofadr[0]) for name in names)
            for names in _ARM_JOINT_NAMES
        )

    def capture(
        self,
        policy_observation: Mapping[str, object],
        *,
        previous_action44: np.ndarray | None = None,
    ) -> CerebellumSensorObservation:
        state = np.asarray(policy_observation["state"], dtype=np.float32).reshape(-1)
        if state.shape != (46,):
            raise ValueError(f"dual-arm sensor observation requires state46, got {state.shape}")
        images = {
            str(name): np.asarray(value)
            for name, value in policy_observation.items()
            if name not in {"state", "prompt", "force"}
            and isinstance(value, np.ndarray)
            and value.ndim == 3
        }
        return CerebellumSensorObservation(
            timestamp_s=float(self._data.time),
            state46=state,
            arm_joint_torque=self._arm_joint_torque(),
            fingertip_force_world=self._fingertip_force_world(),
            wrist_wrench_world=self._wrist_wrench_world(),
            images=images,
            previous_action44=previous_action44,
        )

    def _arm_joint_torque(self) -> np.ndarray:
        return np.asarray(
            [
                [self._data.qfrc_actuator[address] for address in addresses]
                for addresses in self._arm_dof_addresses
            ],
            dtype=np.float32,
        )

    def _fingertip_force_world(self) -> np.ndarray:
        return np.asarray(
            [
                [self._data.cfrc_ext[body_id, :3] for body_id in body_ids]
                for body_ids in self._finger_body_ids
            ],
            dtype=np.float32,
        )

    def _wrist_wrench_world(self) -> np.ndarray:
        wrists: list[np.ndarray] = []
        for site_id, force_name, torque_name in zip(
            self._wrist_site_ids,
            _WRIST_FORCE_SENSORS,
            _WRIST_TORQUE_SENSORS,
            strict=True,
        ):
            rotation = np.asarray(self._data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
            force = rotation @ self._sensor_vector(force_name, size=3)
            torque = rotation @ self._sensor_vector(torque_name, size=3)
            wrists.append(np.concatenate([force, torque]))
        return np.asarray(wrists, dtype=np.float32)

    def _sensor_vector(self, name: str, *, size: int) -> np.ndarray:
        value = np.asarray(self._data.sensor(name).data, dtype=np.float64).reshape(-1)
        if value.shape != (size,):
            raise ValueError(f"sensor {name!r} expected {size} values, got {value.shape}")
        return value
