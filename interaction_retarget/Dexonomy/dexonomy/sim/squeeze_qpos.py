"""Dexonomy ``get_squeeze_qpos`` (mujoco_env.py L681–721), Allegro adapter.

Source: refs/Dexonomy/dexonomy/sim/mujoco_env.py
"""

from __future__ import annotations

import mujoco
import numpy as np


def get_squeeze_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    grasp_qpos: np.ndarray,
    hand_cbody: list[str],
    hand_cp_w: np.ndarray,
    contact_wrenches: np.ndarray,
    *,
    hand_prefix: str = "",
    hand_joint_start: int = 6,
    hand_nv: int | None = None,
) -> np.ndarray:
    """Apply contact wrenches via mj_applyFT → delta finger qpos (Dexonomy original)."""
    grasp_qpos = np.asarray(grasp_qpos, dtype=np.float64).reshape(-1)
    hand_cp_w = np.asarray(hand_cp_w, dtype=np.float64).reshape(-1, 3)
    contact_wrenches = np.asarray(contact_wrenches, dtype=np.float64).reshape(-1, 6)
    nv = int(hand_nv if hand_nv is not None else model.nv)

    data.qfrc_applied[:] = 0
    for h_cb, h_cp_w, h_cw in zip(hand_cbody, hand_cp_w, contact_wrenches):
        body_name = f"{hand_prefix}{h_cb}" if hand_prefix else h_cb
        bid = model.body(body_name).id
        mujoco.mj_applyFT(model, data, h_cw[:3], np.zeros(3, dtype=np.float64), h_cp_w[:3], bid, data.qfrc_applied)

    delta_qpos = np.copy(data.qfrc_applied)
    data.qfrc_applied[:] = 0
    actuator_gainprm = model.actuator_gainprm[:, 0]
    qpos2ctrl = np.zeros((model.nu, model.nv), dtype=np.float64)
    mujoco.mju_sparse2dense(
        qpos2ctrl,
        data.actuator_moment,
        data.moment_rownnz,
        data.moment_rowadr,
        data.moment_colind,
    )
    for i in range(hand_joint_start, nv):
        actuator_id = np.where(qpos2ctrl[:, i] != 0)[0]
        if len(actuator_id) > 0:
            aid = int(actuator_id[0])
            delta_qpos[i] /= actuator_gainprm[aid] * qpos2ctrl[aid].sum()

    finger_start = 7 if grasp_qpos.size >= 23 else hand_joint_start
    n_finger = grasp_qpos.size - finger_start
    delta_finger = delta_qpos[hand_joint_start : hand_joint_start + n_finger]
    squeeze = grasp_qpos.copy()
    squeeze[finger_start:] = grasp_qpos[finger_start:] + delta_finger[:n_finger]
    return squeeze


def squeeze23_from_contacts(
    raw_env,
    *,
    side: str,
    grasp23: np.ndarray,
    ho_contacts: dict,
    ext_wrench: np.ndarray,
    ext_center: np.ndarray,
    wrench_scale: float = 10.0,
) -> np.ndarray | None:
    """Build squeeze_qpos using Dexonomy + DexGraspBench GraspQP wrenches."""
    from interaction_retarget.DexGraspBench.src.task.eval_func.fc_metric.qp import GraspQP

    pos = np.asarray(ho_contacts.get("pos"), dtype=np.float64)
    normal = np.asarray(ho_contacts.get("normal"), dtype=np.float64)
    bodies = ho_contacts.get("bn1") or ho_contacts.get("hand_body") or []
    if pos.size == 0 or normal.size == 0 or len(bodies) == 0:
        return None

    qp = GraspQP(miu_coef=[0.6, 0.02])
    wrenches, err = qp.solve(pos, normal, ext_wrench, ext_center)
    if wrenches is None:
        return None
    # Dexonomy gen_grasp: ho_c["wrench"] = 10 * contact_wrenches (after filter pass).
    wrenches = float(wrench_scale) * np.asarray(wrenches, dtype=np.float64)

    prefix = ""  # dexjoco body names already end with _left/_right
    return get_squeeze_qpos(
        raw_env._model,
        raw_env._data,
        np.asarray(grasp23, dtype=np.float64),
        list(bodies),
        pos,
        wrenches,
        hand_prefix=prefix,
    )
