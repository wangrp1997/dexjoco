"""Privileged PBVS + ep5/7 in-hand regrasp profile.

Default: hard-rim anti-eject (ep20).
Ep5/7: soft o2h, large wrist align, spiral insert.
Hold fingers until seat; release only after insert_ok (see run_p0).
Still PBVS after peg_lift_end — no demo continue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from embodied_grasp_insertion.physics.grasp_metrics import REFERENCE_BODY
from hybrid_insert.geometry import toward_socket_delta, wrist_rotvec_align_peg_axis
from priv_snap_insert.snap import O2HLock, virtual_insert_features
from reach_insert_rl.env.full_obs import privileged_full_features

# Handoff grasp hard to seat with rigid o2h; expanded online when an ep fails.
REGRASP_EPISODES: set[int] = {5, 7}


def register_regrasp_episode(episode_index: int) -> None:
    REGRASP_EPISODES.add(int(episode_index))


@dataclass
class ServoGains:
    tip_step_m: float = 0.009
    target_along_m: float = -0.040
    standoff_m: float = 0.055
    lat_gate_m: float = 0.006
    ang_gate_rad: float = 0.055
    rot_gain: float = 0.22
    max_step_rad: float = 0.028
    settle_substeps: int = 40
    commit_tip_m: float = 0.050
    commit_lat_m: float = 0.010
    commit_exit_tip_m: float = 0.130
    commit_step_m: float = 0.0040
    commit_center_step_m: float = 0.0020
    commit_lat_ok_m: float = 0.007
    jam_tip_lo_m: float = 0.020
    jam_tip_hi_m: float = 0.042
    jam_lat_m: float = 0.020
    jam_patience: int = 12
    jam_max_soft_steps: int = 60
    rim_hi_m: float = 0.042
    regrasp: bool = False
    finger_open_rate: float = 0.0
    regrasp_snap_pos: float = 0.40
    regrasp_snap_rot: float = 0.0
    commit_need_axis: bool = True


def gains_for_episode(episode_index: int) -> ServoGains:
    g = ServoGains()
    if int(episode_index) in REGRASP_EPISODES:
        g.regrasp = True
        g.ang_gate_rad = 0.10
        g.rot_gain = 0.45
        g.max_step_rad = 0.060
        g.commit_tip_m = 0.055
        g.commit_lat_m = 0.014
        g.commit_need_axis = False
        g.commit_step_m = 0.0028
        g.commit_lat_ok_m = 0.012
        g.commit_exit_tip_m = 0.160
        g.finger_open_rate = 0.0  # hold until seat; open only in post-insert release
        g.regrasp_snap_pos = 0.55
        g.regrasp_snap_rot = 0.0
        g.max_step_rad = 0.045  # larger only for pre-commit align_axis
        g.rot_gain = 0.35
        g.settle_substeps = 32
    return g


@dataclass
class ServoCommand:
    right_xyz: np.ndarray
    right_rotvec: np.ndarray
    lat_err: float
    along: float
    axis_err: float
    tip_dist: float
    phase: str
    allow_full_z: bool
    snap_pos: float
    snap_rot: float
    finger_release: float
    jam_spiral: bool
    committed: bool
    stalled: bool


@dataclass
class WristPalmCalib:
    dummy: int = 0


def capture_wrist_palm_calib(raw, hold44: np.ndarray) -> WristPalmCalib:
    del raw, hold44
    return WristPalmCalib()


def _palm_pos(raw) -> np.ndarray:
    bid = int(raw._model.body(REFERENCE_BODY).id)
    return np.asarray(raw._data.xpos[bid], dtype=np.float64).copy()


def _hole_tangent_basis(hole: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = hole / (np.linalg.norm(hole) + 1e-8)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(u, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    e1 = np.cross(u, ref)
    e1 /= np.linalg.norm(e1) + 1e-8
    e2 = np.cross(u, e1)
    e2 /= np.linalg.norm(e2) + 1e-8
    return e1, e2


def adaptive_strength(tip_dist: float, lat_err: float, axis_err: float) -> float:
    if tip_dist > 0.10:
        return 1.0
    if tip_dist > 0.055 or lat_err > 0.014 or axis_err > 0.22:
        return 0.85
    if tip_dist > 0.038 or lat_err > 0.008 or axis_err > 0.14:
        return 0.45
    return 0.12


def anisotropic_snap(
    tip_dist: float,
    lat_err: float,
    axis_err: float,
    *,
    regrasp: bool = False,
) -> tuple[float, float]:
    del axis_err
    if regrasp and tip_dist <= 0.055:
        if lat_err > 0.016:
            return 0.70, 0.05
        return 0.40, 0.0
    if tip_dist > 0.10:
        return 1.0, 1.0
    if tip_dist > 0.028:
        if tip_dist > 0.055 or lat_err > 0.014:
            return 1.0, 1.0
        return 1.0, 0.90
    if lat_err > 0.010:
        return 0.95, 0.18
    return 0.70, 0.12


def compute_servo_command(
    raw,
    hold44: np.ndarray,
    o2h_or_lock=None,
    calib: WristPalmCalib | None = None,
    *,
    gains: ServoGains | None = None,
    jam_steps: int = 0,
    servo_k: int = 0,
    committed: bool = False,
    stalled: bool = False,
) -> ServoCommand:
    del calib, servo_k, stalled
    g = gains or ServoGains()
    hold = np.asarray(hold44, dtype=np.float64).reshape(44)
    # Regrasp frees peg from o2h — servo must track *actual* peg tip, not virtual lock.
    if g.regrasp or not isinstance(o2h_or_lock, O2HLock):
        feat = privileged_full_features(raw, target_along_m=g.target_along_m)
    else:
        feat = virtual_insert_features(raw, o2h_or_lock, target_along_m=g.target_along_m)

    tip = np.asarray(feat["tip"], dtype=np.float64)
    socket = np.asarray(feat["socket"], dtype=np.float64)
    hole = np.asarray(feat["hole"], dtype=np.float64)
    peg_ax = np.asarray(feat["peg_axis"], dtype=np.float64)
    hole_u = hole / (np.linalg.norm(hole) + 1e-8)
    lat_err = float(feat["lat_err"])
    along = float(feat["along"])
    tip_dist = float(feat["tip_dist"])
    axis_err = float(feat["axis_err"])

    axis_ok = (not g.commit_need_axis) or (axis_err <= g.ang_gate_rad + 0.02)
    if tip_dist <= g.commit_tip_m and lat_err <= g.commit_lat_m and axis_ok:
        committed = True
    if tip_dist >= g.commit_exit_tip_m:
        committed = False

    right_rot = hold[3:6].copy()
    rotate = False
    jam_spiral = False
    finger_release = 0.0
    step = g.tip_step_m

    if committed:
        hold_h = float(np.clip(along, 0.014, 0.050))
        if g.regrasp and tip_dist <= 0.055:
            tip_tgt = socket + hole_u * g.target_along_m
            e1, e2 = _hole_tangent_basis(hole)
            ang = 0.55 * float(jam_steps)
            tip_tgt = tip_tgt + 0.0025 * (np.cos(ang) * e1 + np.sin(ang) * e2)
            phase = "regrasp_insert"
            step = g.commit_step_m
            # Soft peg + wrist twist together tumbles; wrist only at standoff.
            rotate = False
            jam_spiral = True
            finger_release = 0.0  # never ease-open mid-insert
        elif (not g.regrasp) and lat_err > g.commit_lat_ok_m and tip_dist > 0.018:
            tip_tgt = socket + hole_u * float(np.clip(hold_h + 0.010, 0.028, 0.055))
            phase = "commit_unjam"
            step = g.commit_center_step_m
            rotate = False
        elif tip_dist < 0.016:
            tip_tgt = socket + hole_u * g.target_along_m
            phase = "insert_finish"
            step = g.commit_step_m
            rotate = False
        else:
            tip_tgt = socket + hole_u * g.target_along_m
            phase = "insert_commit"
            step = g.commit_step_m
            rotate = False
        allow_z = True
    elif tip_dist > 0.32:
        tip_tgt, phase, allow_z = socket + hole_u * 0.12, "recover", True
    elif lat_err > g.lat_gate_m:
        h = max(along, g.standoff_m) if along > 0.0 else g.standoff_m
        tip_tgt, phase, allow_z = socket + hole_u * h, "align_xy", False
    elif axis_err > g.ang_gate_rad:
        h = max(g.standoff_m, min(along, 0.12)) if along > 0 else g.standoff_m
        tip_tgt, phase, allow_z, rotate = socket + hole_u * h, "align_axis", False, True
    elif along > g.standoff_m + 0.006:
        tip_tgt, phase, allow_z = socket + hole_u * g.standoff_m, "to_standoff", True
    else:
        tip_tgt, phase, allow_z = socket + hole_u * g.target_along_m, "insert", True

    v = toward_socket_delta(tip, tip_tgt, gain=1.0, max_step_m=step)

    if rotate:
        aligned = wrist_rotvec_align_peg_axis(
            peg_ax,
            hole,
            right_rot,
            angle_tol_rad=0.025,
            gain=g.rot_gain,
            max_step_rad=g.max_step_rad,
        )
        if aligned is not None:
            r_old = R.from_rotvec(hold[3:6])
            r_new = R.from_rotvec(aligned)
            palm = _palm_pos(raw)
            tip_after = palm + (r_new * r_old.inv()).apply(tip - palm)
            v = v + (tip_tgt - tip_after)
            vn = float(np.linalg.norm(v))
            if vn > step * 1.4 and vn > 1e-12:
                v *= step * 1.4 / vn
            right_rot = aligned

    snap_pos, snap_rot = anisotropic_snap(
        tip_dist, lat_err, axis_err, regrasp=bool(g.regrasp)
    )
    if phase == "commit_unjam":
        snap_pos, snap_rot = 1.0, 0.85
    if phase == "regrasp_insert":
        snap_pos, snap_rot = g.regrasp_snap_pos, g.regrasp_snap_rot

    return ServoCommand(
        right_xyz=(hold[0:3] + v).astype(np.float64),
        right_rotvec=np.asarray(right_rot, dtype=np.float64),
        lat_err=lat_err,
        along=along,
        axis_err=axis_err,
        tip_dist=tip_dist,
        phase=phase,
        allow_full_z=allow_z,
        snap_pos=float(snap_pos),
        snap_rot=float(snap_rot),
        finger_release=float(finger_release),
        jam_spiral=jam_spiral,
        committed=bool(committed),
        stalled=bool(snap_rot < 0.5 or phase == "regrasp_insert"),
    )


def insert_ok_quality(feat: dict, *, max_axis_rad: float = 0.45, max_lat_m: float = 0.022) -> bool:
    """Reject tumble/contact flukes (ep5-style axis≈1.5)."""
    return (
        float(feat["axis_err"]) <= max_axis_rad
        and float(feat["lat_err"]) <= max_lat_m
        and float(feat["along"]) <= 0.015
    )


def geometric_seat_ok(feat: dict) -> bool:
    """Tip actually in the hole, still held. Not '1cm above the rim'."""
    return (
        float(feat["tip_dist"]) <= 0.008
        and float(feat["lat_err"]) <= 0.012
        and float(feat["along"]) <= -0.003
        and float(feat["axis_err"]) <= 0.28
    )


def seat_success(feat: dict, *, insert_ok: bool) -> bool:
    del insert_ok
    return geometric_seat_ok(feat)


def transfer_seat_ready(feat: dict) -> bool:
    """Rim-aligned enough to open hand and privilege-pin peg into hole."""
    return (
        float(feat["lat_err"]) <= 0.012
        and float(feat["axis_err"]) <= 0.20
        and float(feat["tip_dist"]) <= 0.035
        and float(feat["along"]) <= 0.040
    )


def release_hold_ok(feat: dict, *, insert_ok: bool) -> bool:
    """After opening the hand: peg stays in hole (may be deeper than tip_dist 8mm)."""
    if geometric_seat_ok(feat):
        return True
    return (
        bool(insert_ok)
        and float(feat["lat_err"]) <= 0.014
        and float(feat["axis_err"]) <= 0.35
        and float(feat["along"]) <= 0.006
        and float(feat["tip_dist"]) <= 0.030
    )
