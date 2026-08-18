"""Privileged wrap: slide peg toward middle, curl only that finger, recapture o2h."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from dexjoco.sim.envs.assembly_geometry import names_from_raw
from embodied_grasp_insertion.physics.grasp_metrics import (
    ObjectInHandPose,
    peg_hand_contact_counts,
)
from embodied_grasp_insertion.simulation.calibrated_interventions import (
    RIGHT_FINGER_ACTUATOR_NAMES,
)


def clip_right_fingers(raw, fr: np.ndarray) -> np.ndarray:
    out = np.asarray(fr, dtype=np.float64).reshape(16).copy()
    for i, name in enumerate(RIGHT_FINGER_ACTUATOR_NAMES):
        aid = int(raw._model.actuator(name).id)
        lo, hi = raw._model.actuator_ctrlrange[aid]
        out[i] = float(np.clip(out[i], lo, hi))
    return out


def contact_summary(raw) -> dict[str, int]:
    c = peg_hand_contact_counts(raw)
    return {"total": int(c.total), **{k: int(v) for k, v in c.by_class.items()}}


def wrap_target_from_pinch(fr: np.ndarray) -> np.ndarray:
    """Curl middle (+ a little ring). Do not retighten index/thumb."""
    t = np.asarray(fr, dtype=np.float64).reshape(16).copy()
    t[4] = float(np.clip(t[4] + 0.06, -0.47, 0.47))
    t[5] = max(float(t[5]), 1.50)
    t[6] = max(float(t[6]), 1.45)
    t[7] = max(float(t[7]), 1.00)
    t[8] = float(np.clip(t[8] + 0.10, -0.47, 0.47))
    t[9] = max(float(t[9]), 1.40)
    t[10] = max(float(t[10]), 1.35)
    t[11] = max(float(t[11]), 0.70)
    return t


def shift_o2h_toward_body(
    raw,
    o2h: ObjectInHandPose,
    body: str,
    *,
    meters: float,
) -> ObjectInHandPose:
    """Move peg toward a hand body, perpendicular to the peg axis."""
    model, data = raw._model, raw._data
    peg_id = int(model.body(names_from_raw(raw).peg_body).id)
    bid = int(model.body(body).id)
    palm_id = int(model.body(o2h.reference_body).id)
    peg = np.asarray(data.xpos[peg_id], dtype=np.float64)
    tgt = np.asarray(data.xpos[bid], dtype=np.float64)
    peg_rot = R.from_quat(np.asarray(data.xquat[peg_id], dtype=np.float64), scalar_first=True)
    peg_z = peg_rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    v = tgt - peg
    v = v - peg_z * float(np.dot(v, peg_z))
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return o2h
    new_peg = peg + (v / n) * float(meters)
    palm_pos = np.asarray(data.xpos[palm_id], dtype=np.float64)
    palm_rot = R.from_quat(np.asarray(data.xquat[palm_id], dtype=np.float64), scalar_first=True)
    rel_t = palm_rot.inv().apply(new_peg - palm_pos)
    return ObjectInHandPose(
        reference_body=o2h.reference_body,
        translation=np.asarray(rel_t, dtype=np.float64),
        rotvec=np.asarray(o2h.rotvec, dtype=np.float64).copy(),
    )
