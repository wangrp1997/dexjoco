"""Adaptive privileged PBVS for peg insertion.

Unlike the hard-snap baseline, this controller uses a soft object-in-hand
constraint that fades near the socket, where contact dynamics must be allowed
to settle instead of being overwritten every frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from embodied_grasp_insertion.physics.grasp_metrics import REFERENCE_BODY
from hybrid_insert.geometry import (
    height_along_axis,
    lateral_error,
    toward_socket_delta,
    wrist_rotvec_align_peg_axis,
)
from reach_insert_rl.env.full_obs import privileged_full_features


@dataclass
class Gains:
    target_along_m: float = -0.040
    standoff_m: float = 0.055
    tip_step_m: float = 0.008
    lat_gate_m: float = 0.007
    axis_gate_rad: float = 0.10
    rot_gain: float = 0.14
    max_rot_step_rad: float = 0.012


@dataclass
class Command:
    xyz: np.ndarray
    rotvec: np.ndarray
    phase: str
    tip_dist: float
    lat_err: float
    along: float
    axis_err: float


def _palm_pos(raw) -> np.ndarray:
    bid = int(raw._model.body(REFERENCE_BODY).id)
    return np.asarray(raw._data.xpos[bid], dtype=np.float64).copy()


def command(raw, hold44: np.ndarray, *, gains: Gains | None = None) -> Command:
    g = gains or Gains()
    hold = np.asarray(hold44, dtype=np.float64).reshape(44)
    feat = privileged_full_features(raw, target_along_m=g.target_along_m)
    tip = np.asarray(feat["tip"], dtype=np.float64)
    socket = np.asarray(feat["socket"], dtype=np.float64)
    hole = np.asarray(feat["hole"], dtype=np.float64)
    hole_u = hole / (np.linalg.norm(hole) + 1e-8)
    peg_axis = np.asarray(feat["peg_axis"], dtype=np.float64)
    lat, _ = lateral_error(tip, socket, hole)
    along = height_along_axis(tip, socket, hole)
    dist = float(np.linalg.norm(tip - socket))
    axis_err = float(feat["axis_err"])

    if dist > 0.32:
        phase, target, rotate = "recover", socket + hole_u * 0.12, False
    elif lat > g.lat_gate_m:
        phase, target, rotate = "align_xy", socket + hole_u * max(g.standoff_m, along), False
    elif axis_err > g.axis_gate_rad:
        phase, target, rotate = "align_axis", socket + hole_u * g.standoff_m, True
    elif along > g.standoff_m + 0.006:
        phase, target, rotate = "approach", socket + hole_u * g.standoff_m, False
    else:
        phase, target, rotate = "insert", socket + hole_u * g.target_along_m, False

    delta = toward_socket_delta(tip, target, gain=1.0, max_step_m=g.tip_step_m)
    rot = hold[3:6].copy()
    if rotate:
        aligned = wrist_rotvec_align_peg_axis(
            peg_axis,
            hole,
            rot,
            angle_tol_rad=0.035,
            gain=g.rot_gain,
            max_step_rad=g.max_rot_step_rad,
        )
        if aligned is not None:
            old = R.from_rotvec(rot)
            new = R.from_rotvec(aligned)
            palm = _palm_pos(raw)
            tip_after = palm + (new * old.inv()).apply(tip - palm)
            delta += target - tip_after
            norm = float(np.linalg.norm(delta))
            if norm > g.tip_step_m * 1.4:
                delta *= g.tip_step_m * 1.4 / norm
            rot = np.asarray(aligned, dtype=np.float64)
    return Command(
        xyz=hold[:3] + delta,
        rotvec=rot,
        phase=phase,
        tip_dist=dist,
        lat_err=float(lat),
        along=float(along),
        axis_err=axis_err,
    )
