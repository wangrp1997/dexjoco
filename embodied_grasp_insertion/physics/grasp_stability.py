"""P0-S0.4: grasp stability *instrumentation / phase-orchestration* smoke.

Positive path uses an oracle kinematic hand–peg fixture (per-step palm-snap).
This is NOT a physical grasp stability gate (see P0-S0.4b).
Negative path releases the fixture and opens the hand to show drop metrics fire.
Thresholds scale with family characteristic length (not hardcoded 8mm meters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from dexjoco.sim.envs.assembly_geometry import names_from_raw
from embodied_grasp_insertion.geometry.family_spec import GeometryFamilySpec
from embodied_grasp_insertion.physics.grasp_metrics import (
    REFERENCE_BODY,
    object_in_hand_pose,
    peg_hand_contact_counts,
    relative_pose_error,
)


def characteristic_length_m(spec: GeometryFamilySpec) -> float:
    c = spec.collision
    if spec.section == "round":
        return float(c["peg_radius_m"])
    return float(max(c["peg_half_width_m"], c["peg_half_depth_m"]))


def scale_thresholds(char_len_m: float) -> dict[str, float]:
    L = max(float(char_len_m), 1e-4)
    return {
        "char_len_m": L,
        # Hold/transport under kinematic fixture: small residual tracking error.
        "hold_trans_tol_m": 0.25 * L,
        "hold_rot_tol_rad": 0.08,
        "transport_trans_tol_m": 0.35 * L,
        "transport_rot_tol_rad": 0.12,
        # Lift: peg must follow hand rise and clear table.
        "lift_follow_ratio_min": 0.6,
        "lift_min_hand_dz_m": 0.05,
        "table_clearance_z_m": 1.00,
        "lift_trans_tol_m": 0.35 * L,
        "lift_rot_tol_rad": 0.15,
        # Negative: open-hand must produce large drift or drop.
        "neg_trans_min_m": 0.5 * L,
        "neg_drop_delta_z_m": 0.02,
        "max_peg_speed_m_s": 3.0,
        "hold_abs_z_tol_m": 0.03,
    }


def peg_table_contact(model: mujoco.MjModel, data: mujoco.MjData, peg_body_id: int) -> int:
    peg_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == peg_body_id}
    table_geoms = set()
    for g in range(model.ngeom):
        name = model.geom(g).name or ""
        if name.startswith("table") or name == "floor":
            table_geoms.add(g)
    n = 0
    for i in range(data.ncon):
        g1, g2 = int(data.contact[i].geom1), int(data.contact[i].geom2)
        pair = {g1, g2}
        if peg_geoms & pair and table_geoms & pair:
            n += 1
    return n


def _palm_frame(raw) -> tuple[np.ndarray, np.ndarray]:
    bid = int(raw._model.body(REFERENCE_BODY).id)
    pos = np.asarray(raw._data.xpos[bid], dtype=np.float64).copy()
    quat = np.asarray(raw._data.xquat[bid], dtype=np.float64).copy()  # wxyz
    return pos, quat


def local_offset_for_spec(spec: GeometryFamilySpec) -> np.ndarray:
    """Peg origin in right-palm frame (oracle fixture pose)."""
    L = characteristic_length_m(spec)
    # Along palm approach / slight in-hand; scale with size.
    return np.array([0.0, 0.0, 0.03 + 0.8 * L], dtype=np.float64)


def snap_peg_to_palm(env, local_offset: np.ndarray) -> None:
    raw = env
    palm_pos, palm_quat = _palm_frame(raw)
    Rp = R.from_quat(palm_quat, scalar_first=True).as_matrix()
    peg_pos = palm_pos + Rp @ np.asarray(local_offset, dtype=np.float64)
    env._set_free_joint_pose(env._peg_qpos_adr, env._peg_qvel_adr, peg_pos, palm_quat)
    mujoco.mj_forward(env.model, env.data)


def step_with_fixture(env, action: dict[str, np.ndarray], local_offset: np.ndarray) -> None:
    """Env step then re-snap peg (opspace substeps would otherwise let peg drift)."""
    env.step(action)
    snap_peg_to_palm(env, local_offset)


def _palm_z(env) -> float:
    return float(env.data.xpos[int(env.model.body(REFERENCE_BODY).id), 2])


def settle_hand_to_target(
    env,
    *,
    rpos: np.ndarray,
    rquat: np.ndarray,
    fingers: np.ndarray,
    offset: np.ndarray,
    tol_m: float = 0.02,
    max_steps: int = 80,
    min_steps: int = 0,
) -> None:
    for i in range(max_steps):
        step_with_fixture(
            env,
            make_action(env, right_pos=rpos, right_quat_wxyz=rquat, right_fingers=fingers),
            offset,
        )
        if i + 1 < min_steps:
            continue
        # Opspace typically leaves ~3cm lag; accept tol on Z only.
        if abs(_palm_z(env) - float(rpos[2])) <= tol_m:
            break


def closed_finger_cmd(spec: GeometryFamilySpec) -> np.ndarray:
    """Right-hand allegro cmd; curl scales mildly with size (still within actuator range)."""
    L = characteristic_length_m(spec)
    # Larger pegs → slightly less curl so fixture geometry clears.
    curl = float(np.clip(0.95 - 8.0 * L, 0.55, 0.95))
    fingers = np.array(
        [0.15, curl, curl, curl] * 3 + [0.4, 0.6, 0.6, 0.6],
        dtype=np.float64,
    )
    return fingers


def open_finger_cmd() -> np.ndarray:
    return np.zeros(16, dtype=np.float64)


def make_action(
    env,
    *,
    right_pos: np.ndarray,
    right_quat_wxyz: np.ndarray,
    right_fingers: np.ndarray,
) -> dict[str, np.ndarray]:
    lpos = np.asarray(env.data.mocap_pos[env._mocap_left_id], dtype=np.float64).copy()
    lquat = np.asarray(env.data.mocap_quat[env._mocap_left_id], dtype=np.float64).copy()
    lf = open_finger_cmd()
    return {
        "right": np.concatenate(
            [np.asarray(right_pos, dtype=np.float64), np.asarray(right_quat_wxyz, dtype=np.float64), right_fingers]
        ),
        "left": np.concatenate([lpos, lquat, lf]),
    }


@dataclass
class PhaseResult:
    name: str
    passed: bool
    metrics: dict[str, Any]


def _peg_speed(env) -> float:
    v = np.asarray(env.data.cvel[env._peg_body_id], dtype=np.float64)
    return float(np.linalg.norm(v[3:]))


def run_lift_hold_transport(
    env,
    spec: GeometryFamilySpec,
    *,
    lift_steps: int = 40,
    hold_steps: int = 40,
    transport_steps: int = 40,
    dz_per_step: float = 0.002,
    dy_per_step: float = 0.002,
) -> dict[str, Any]:
    """Oracle kinematic fixture positive phases + open-hand negative."""
    thr = scale_thresholds(characteristic_length_m(spec))
    offset = local_offset_for_spec(spec)
    rf_closed = closed_finger_cmd(spec)
    rf_open = open_finger_cmd()

    # Start from reset pose: move hand above peg then fixture.
    env.reset(seed=0)
    peg0 = np.asarray(env.data.xpos[env._peg_body_id], dtype=np.float64).copy()
    rquat = np.asarray(env.data.mocap_quat[env._mocap_right_id], dtype=np.float64).copy()
    # Approach slightly above peg; chunked settle so palm actually reaches target.
    rpos = peg0 + np.array([0.0, 0.0, 0.10], dtype=np.float64)
    settle_hand_to_target(
        env, rpos=rpos, rquat=rquat, fingers=rf_closed, offset=offset, tol_m=0.035, max_steps=150, min_steps=20
    )

    o2h0 = object_in_hand_pose(env)
    z0 = float(env.data.xpos[env._peg_body_id, 2])
    hand_z0 = _palm_z(env)
    phases: list[PhaseResult] = []

    # --- lift ---
    # Opspace lags mocap (~3cm); raise in chunks with forced settle steps so palm actually rises.
    max_drift_t = max_drift_r = 0.0
    max_speed = 0.0
    table_hits = 0
    chunk = 0.005
    n_chunks = max(24, int(np.ceil(1.5 * thr["lift_min_hand_dz_m"] / chunk)))
    for _ in range(n_chunks):
        rpos = rpos + np.array([0.0, 0.0, chunk], dtype=np.float64)
        settle_hand_to_target(
            env,
            rpos=rpos,
            rquat=rquat,
            fingers=rf_closed,
            offset=offset,
            tol_m=0.035,
            max_steps=30,
            min_steps=12,
        )
        o2h = object_in_hand_pose(env)
        dt, dr = relative_pose_error(o2h0, o2h)
        max_drift_t = max(max_drift_t, dt)
        max_drift_r = max(max_drift_r, dr)
        max_speed = max(max_speed, _peg_speed(env))
        table_hits += peg_table_contact(env.model, env.data, env._peg_body_id)
        if (_palm_z(env) - hand_z0) >= thr["lift_min_hand_dz_m"]:
            break
    z1 = float(env.data.xpos[env._peg_body_id, 2])
    hand_z1 = _palm_z(env)
    hand_dz = hand_z1 - hand_z0
    peg_dz = z1 - z0
    follow_ok = hand_dz >= thr["lift_min_hand_dz_m"] and peg_dz >= thr["lift_follow_ratio_min"] * hand_dz
    lift_ok = (
        follow_ok
        and z1 >= thr["table_clearance_z_m"]
        and table_hits == 0
        and max_drift_t <= thr["lift_trans_tol_m"]
        and max_drift_r <= thr["lift_rot_tol_rad"]
        and max_speed <= thr["max_peg_speed_m_s"]
    )
    phases.append(
        PhaseResult(
            "lift",
            lift_ok,
            {
                "peg_dz": peg_dz,
                "hand_dz": hand_dz,
                "peg_z": z1,
                "table_hits": table_hits,
                "max_rel_trans": max_drift_t,
                "max_rel_rot": max_drift_r,
                "max_speed": max_speed,
                "contacts": peg_hand_contact_counts(env).total,
            },
        )
    )
    # Extra settle before hold baseline.
    for _ in range(15):
        step_with_fixture(
            env,
            make_action(env, right_pos=rpos, right_quat_wxyz=rquat, right_fingers=rf_closed),
            offset,
        )
    o2h_hold0 = object_in_hand_pose(env)
    z_hold0 = float(env.data.xpos[env._peg_body_id, 2])

    # --- hold ---
    max_drift_t = max_drift_r = 0.0
    max_speed = 0.0
    for _ in range(hold_steps):
        step_with_fixture(
            env,
            make_action(env, right_pos=rpos, right_quat_wxyz=rquat, right_fingers=rf_closed),
            offset,
        )
        o2h = object_in_hand_pose(env)
        dt, dr = relative_pose_error(o2h_hold0, o2h)
        max_drift_t = max(max_drift_t, dt)
        max_drift_r = max(max_drift_r, dr)
        max_speed = max(max_speed, _peg_speed(env))
    hold_ok = (
        max_drift_t <= thr["hold_trans_tol_m"]
        and max_drift_r <= thr["hold_rot_tol_rad"]
        and abs(float(env.data.xpos[env._peg_body_id, 2]) - z_hold0) <= thr["hold_abs_z_tol_m"]
        and max_speed <= thr["max_peg_speed_m_s"]
    )
    phases.append(
        PhaseResult(
            "hold",
            hold_ok,
            {
                "max_rel_trans": max_drift_t,
                "max_rel_rot": max_drift_r,
                "max_speed": max_speed,
                "peg_z": float(env.data.xpos[env._peg_body_id, 2]),
                "contacts": peg_hand_contact_counts(env).total,
            },
        )
    )
    o2h_tr0 = object_in_hand_pose(env)

    # --- transport ---
    max_drift_t = max_drift_r = 0.0
    max_speed = 0.0
    for _ in range(transport_steps):
        rpos = rpos + np.array([0.0, dy_per_step, 0.0], dtype=np.float64)
        step_with_fixture(
            env,
            make_action(env, right_pos=rpos, right_quat_wxyz=rquat, right_fingers=rf_closed),
            offset,
        )
        o2h = object_in_hand_pose(env)
        dt, dr = relative_pose_error(o2h_tr0, o2h)
        max_drift_t = max(max_drift_t, dt)
        max_drift_r = max(max_drift_r, dr)
        max_speed = max(max_speed, _peg_speed(env))
    transport_ok = (
        max_drift_t <= thr["transport_trans_tol_m"]
        and max_drift_r <= thr["transport_rot_tol_rad"]
        and max_speed <= thr["max_peg_speed_m_s"]
    )
    phases.append(
        PhaseResult(
            "transport",
            transport_ok,
            {
                "max_rel_trans": max_drift_t,
                "max_rel_rot": max_drift_r,
                "max_speed": max_speed,
                "peg_z": float(env.data.xpos[env._peg_body_id, 2]),
                "contacts": peg_hand_contact_counts(env).total,
            },
        )
    )

    # --- negative: release fixture + open hand ---
    o2h_neg0 = object_in_hand_pose(env)
    z_neg0 = float(env.data.xpos[env._peg_body_id, 2])
    max_drift_t = 0.0
    min_z = z_neg0
    for _ in range(50):
        env.step(make_action(env, right_pos=rpos, right_quat_wxyz=rquat, right_fingers=rf_open))
        o2h = object_in_hand_pose(env)
        dt, _ = relative_pose_error(o2h_neg0, o2h)
        max_drift_t = max(max_drift_t, dt)
        min_z = min(min_z, float(env.data.xpos[env._peg_body_id, 2]))
    drop = z_neg0 - min_z
    neg_ok = (max_drift_t >= thr["neg_trans_min_m"]) or (drop >= thr["neg_drop_delta_z_m"])
    phases.append(
        PhaseResult(
            "open_hand_negative",
            neg_ok,
            {
                "max_rel_trans": max_drift_t,
                "drop_z": drop,
                "min_peg_z": min_z,
                "contacts_end": peg_hand_contact_counts(env).total,
            },
        )
    )

    pos_ok = all(p.passed for p in phases if p.name != "open_hand_negative")
    neg_ok_flag = next(p.passed for p in phases if p.name == "open_hand_negative")
    return {
        "thresholds": thr,
        "local_offset": offset.tolist(),
        "phases": [{"name": p.name, "passed": p.passed, "metrics": p.metrics} for p in phases],
        "positive_ok": bool(pos_ok),
        "negative_ok": bool(neg_ok_flag),
        "passed": bool(pos_ok and neg_ok_flag),
        "fixture": "oracle_kinematic_palm_snap",
        "claims_physical_grasp_stability": False,
        "note": (
            "INSTRUMENTATION ONLY: positive phases re-snap peg to palm every step "
            "(not finger/friction grasp). Negative opens hand without fixture to verify "
            "drop metrics; does not prove open-hand fails vs closed physical grasp. "
            "Physical gate is P0-S0.4b."
        ),
        "geometry_family_id": names_from_raw(env).family_id,
    }
