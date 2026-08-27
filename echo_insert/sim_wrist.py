"""Fail-closed access to the deployable wrist F/T sensor channels."""

from __future__ import annotations

import numpy as np


_WRIST_SENSORS = (
    ("panda/wrist_force_right", "panda/wrist_torque_right"),
    ("panda/wrist_force_left", "panda/wrist_torque_left"),
)


def read_wrist_wrench_local(raw) -> np.ndarray:
    """Return right/left local wrist ``[force, torque]`` readings."""
    result = np.empty((2, 6), dtype=np.float64)
    for side, (force_name, torque_name) in enumerate(_WRIST_SENSORS):
        force = np.asarray(raw._data.sensor(force_name).data, dtype=np.float64).reshape(-1)
        torque = np.asarray(raw._data.sensor(torque_name).data, dtype=np.float64).reshape(-1)
        if force.shape != (3,) or torque.shape != (3,):
            raise ValueError(
                f"wrist sensors must be 3D, got {force_name}={force.shape}, "
                f"{torque_name}={torque.shape}"
            )
        result[side] = np.concatenate([force, torque])
    if not np.isfinite(result).all():
        raise ValueError("wrist sensors must contain only finite values")
    return result.copy()
