"""MuJoCo sim snapshot / restore (decoupled multi-agent planning)."""

from __future__ import annotations

import mujoco
import numpy as np


SimSnapshot = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def snapshot_sim(raw_env) -> SimSnapshot:
    data = raw_env._data
    return (
        data.qpos.copy(),
        data.qvel.copy(),
        data.mocap_pos.copy(),
        data.mocap_quat.copy(),
    )


def restore_sim(raw_env, state: SimSnapshot) -> None:
    qpos, qvel, mocap_pos, mocap_quat = state
    data = raw_env._data
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.mocap_pos[:] = mocap_pos
    data.mocap_quat[:] = mocap_quat
    mujoco.mj_forward(raw_env._model, data)
