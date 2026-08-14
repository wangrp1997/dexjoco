"""P0-S0.4b: physical grasp stability gate (no per-step snap).

Oracle allowed only once: restore to a demo transport root (real hand–peg contacts).
Lift / hold / transport run under pure FullEpisodeEnv dynamics (delta actions).
Open-hand negative restores the same root and must be worse than closed hold.
Does NOT claim a learned grasp policy; still bans collection/training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from embodied_grasp_insertion.physics.grasp_metrics import (
    REFERENCE_BODY,
    object_in_hand_pose,
    peg_hand_contact_counts,
    relative_pose_error,
)
from embodied_grasp_insertion.physics.grasp_stability import peg_table_contact
from embodied_grasp_insertion.simulation.full_episode_snapshot import FullEpisodeSnapshot
from embodied_grasp_insertion.simulation.full_episode_utils import RIGHT_FINGER_IDX


def scale_physical_thresholds(char_len_m: float = 0.018) -> dict[str, float]:
    """Thresholds for physical grasp; scale pose tol by characteristic length."""
    L = max(float(char_len_m), 1e-4)
    return {
        "char_len_m": L,
        "min_contacts": 2,
        "hold_trans_tol_m": max(0.02, 1.2 * L),
        "hold_rot_tol_rad": 0.25,
        "lift_min_hand_dz_m": 0.04,
        "lift_follow_ratio_min": 0.6,
        "lift_trans_tol_m": max(0.025, 1.5 * L),
        "lift_rot_tol_rad": 0.30,
        "transport_trans_tol_m": max(0.025, 1.5 * L),
        "transport_rot_tol_rad": 0.30,
        "transport_min_hand_lateral_m": 0.03,
        "transport_lateral_follow_ratio_min": 0.55,
        "max_peg_speed_m_s": 4.0,
        "neg_drop_delta_z_m": 0.02,
        "neg_trans_min_m": max(0.03, 2.0 * L),
        # Closed must beat open on drop or relative drift.
        "closed_better_drop_margin_m": 0.01,
        "closed_better_drift_margin_m": 0.01,
    }


@dataclass
class PhaseResult:
    name: str
    passed: bool
    metrics: dict[str, Any]


def _peg_speed(raw) -> float:
    v = np.asarray(raw.data.cvel[raw._peg_body_id], dtype=np.float64)
    return float(np.linalg.norm(v[3:]))


def _palm_xyz(raw) -> np.ndarray:
    return np.asarray(raw.data.xpos[int(raw.model.body(REFERENCE_BODY).id)], dtype=np.float64).copy()


def _palm_z(raw) -> float:
    return float(_palm_xyz(raw)[2])


def _peg_xyz(raw) -> np.ndarray:
    return np.asarray(raw.data.xpos[raw._peg_body_id], dtype=np.float64).copy()


def _zeros(n: int) -> list[np.ndarray]:
    return [np.zeros(44, dtype=np.float64) for _ in range(n)]


def _wrist_delta(axis: int, n: int, mag: float) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for _ in range(n):
        a = np.zeros(44, dtype=np.float64)
        a[int(axis)] = float(mag)
        out.append(a)
    return out


def _open_finger_deltas(n: int, mag: float = 1.0) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for _ in range(n):
        a = np.zeros(44, dtype=np.float64)
        a[RIGHT_FINGER_IDX] = -float(mag)
        out.append(a)
    return out


def _run_actions(
    env,
    actions: list[np.ndarray],
    *,
    track_lateral: bool = False,
    commanded_lateral_m: float | None = None,
) -> dict[str, Any]:
    raw = env._raw
    o0 = object_in_hand_pose(raw)
    peg0 = _peg_xyz(raw)
    palm0 = _palm_xyz(raw)
    z0 = float(peg0[2])
    hz0 = float(palm0[2])
    max_dt = max_dr = 0.0
    min_contacts = 10**9
    table_hits = 0
    max_speed = 0.0
    for a in actions:
        if bool(env._done):
            # Allow short gates near episode end by clearing done after restore only.
            raise RuntimeError("episode done mid-phase; shorten horizon or restore earlier")
        env.step(a)
        dt, dr = relative_pose_error(o0, object_in_hand_pose(raw))
        max_dt = max(max_dt, dt)
        max_dr = max(max_dr, dr)
        min_contacts = min(min_contacts, int(peg_hand_contact_counts(raw).total))
        table_hits += int(peg_table_contact(raw.model, raw.data, raw._peg_body_id))
        max_speed = max(max_speed, _peg_speed(raw))
    peg1 = _peg_xyz(raw)
    palm1 = _palm_xyz(raw)
    out: dict[str, Any] = {
        "hand_dz": float(palm1[2] - hz0),
        "peg_dz": float(peg1[2] - z0),
        "peg_z": float(peg1[2]),
        "max_rel_trans": max_dt,
        "max_rel_rot": max_dr,
        "min_contacts": int(min_contacts if min_contacts < 10**9 else 0),
        "contacts_end": int(peg_hand_contact_counts(raw).total),
        "table_hits": int(table_hits),
        "max_speed": max_speed,
        "drop_z": max(0.0, z0 - float(peg1[2])),
    }
    if track_lateral:
        hand_dxy = palm1[:2] - palm0[:2]
        peg_dxy = peg1[:2] - peg0[:2]
        out["hand_dy"] = float(hand_dxy[1])
        out["peg_dy"] = float(peg_dxy[1])
        out["hand_lateral_m"] = float(np.linalg.norm(hand_dxy))
        out["peg_lateral_m"] = float(np.linalg.norm(peg_dxy))
        out["commanded_lateral_m"] = (
            float(commanded_lateral_m) if commanded_lateral_m is not None else 0.0
        )
    return out


def run_physical_from_snapshot(
    env,
    snap: FullEpisodeSnapshot,
    *,
    thr: dict[str, float] | None = None,
    hold_steps: int = 30,
    lift_steps: int = 35,
    transport_steps: int = 40,
    neg_steps: int = 40,
    lift_mag: float = 0.5,
    transport_mag: float = 0.5,
) -> dict[str, Any]:
    """Physical lift/hold/transport + open-hand negative from one restored root."""
    thr = dict(thr or scale_physical_thresholds())
    thr.setdefault("transport_min_hand_lateral_m", 0.03)
    thr.setdefault("transport_lateral_follow_ratio_min", 0.55)
    phases: list[PhaseResult] = []

    # --- positive: closed fingers (zero finger delta) ---
    snap.restore(env)
    root_contacts = int(peg_hand_contact_counts(env._raw).total)
    if root_contacts < thr["min_contacts"]:
        return {
            "passed": False,
            "positive_ok": False,
            "negative_ok": False,
            "closed_beats_open": False,
            "thresholds": thr,
            "root_contacts": root_contacts,
            "phases": [],
            "note": "root lacks hand–peg contacts; refuse physical gate",
            "fixture": "demo_transport_root_restore_once",
            "claims_physical_grasp_stability": False,
            "claims_stable_grasp_policy": False,
        }

    hold_m = _run_actions(env, _zeros(hold_steps))
    hold_ok = (
        hold_m["min_contacts"] >= thr["min_contacts"]
        and hold_m["max_rel_trans"] <= thr["hold_trans_tol_m"]
        and hold_m["max_rel_rot"] <= thr["hold_rot_tol_rad"]
        and hold_m["table_hits"] == 0
        and hold_m["max_speed"] <= thr["max_peg_speed_m_s"]
    )
    phases.append(PhaseResult("hold", hold_ok, hold_m))

    lift_m = _run_actions(env, _wrist_delta(2, lift_steps, lift_mag))
    follow_ok = (
        lift_m["hand_dz"] >= thr["lift_min_hand_dz_m"]
        and lift_m["peg_dz"] >= thr["lift_follow_ratio_min"] * lift_m["hand_dz"]
    )
    lift_ok = (
        follow_ok
        and lift_m["min_contacts"] >= thr["min_contacts"]
        and lift_m["max_rel_trans"] <= thr["lift_trans_tol_m"]
        and lift_m["max_rel_rot"] <= thr["lift_rot_tol_rad"]
        and lift_m["table_hits"] == 0
        and lift_m["max_speed"] <= thr["max_peg_speed_m_s"]
    )
    phases.append(PhaseResult("lift", lift_ok, lift_m))

    pos_scale = float(getattr(env, "pos_scale", 0.008))
    cmd_lat = abs(float(transport_mag)) * int(transport_steps) * pos_scale
    tr_m = _run_actions(
        env,
        _wrist_delta(1, transport_steps, transport_mag),
        track_lateral=True,
        commanded_lateral_m=cmd_lat,
    )
    lateral_ok = (
        tr_m["hand_lateral_m"] >= thr["transport_min_hand_lateral_m"]
        and tr_m["peg_lateral_m"]
        >= thr["transport_lateral_follow_ratio_min"] * tr_m["hand_lateral_m"]
    )
    transport_ok = (
        lateral_ok
        and tr_m["min_contacts"] >= thr["min_contacts"]
        and tr_m["max_rel_trans"] <= thr["transport_trans_tol_m"]
        and tr_m["max_rel_rot"] <= thr["transport_rot_tol_rad"]
        and tr_m["table_hits"] == 0
        and tr_m["max_speed"] <= thr["max_peg_speed_m_s"]
    )
    phases.append(PhaseResult("transport", transport_ok, tr_m))
    pos_ok = all(p.passed for p in phases)

    # --- negative: same root, open fingers ---
    snap.restore(env)
    open_m = _run_actions(env, _open_finger_deltas(neg_steps, mag=1.0))
    neg_ok = (open_m["drop_z"] >= thr["neg_drop_delta_z_m"]) or (
        open_m["max_rel_trans"] >= thr["neg_trans_min_m"]
    )
    phases.append(PhaseResult("open_hand_negative", neg_ok, open_m))

    # --- closed control from same root (matched horizon) ---
    snap.restore(env)
    closed_m = _run_actions(env, _zeros(neg_steps))
    closed_beats = (
        closed_m["drop_z"] + thr["closed_better_drop_margin_m"] <= open_m["drop_z"]
    ) or (
        closed_m["max_rel_trans"] + thr["closed_better_drift_margin_m"]
        <= open_m["max_rel_trans"]
    )
    # Prefer: closed retains contacts while open loses them / drops.
    if open_m["min_contacts"] < thr["min_contacts"] and closed_m["min_contacts"] >= thr[
        "min_contacts"
    ]:
        closed_beats = True

    passed = bool(pos_ok and neg_ok and closed_beats)
    return {
        "passed": passed,
        "positive_ok": bool(pos_ok),
        "negative_ok": bool(neg_ok),
        "closed_beats_open": bool(closed_beats),
        "thresholds": thr,
        "root_contacts": root_contacts,
        "closed_matched": closed_m,
        "phases": [{"name": p.name, "passed": p.passed, "metrics": p.metrics} for p in phases],
        "fixture": "demo_transport_root_restore_once",
        "claims_physical_grasp_stability": bool(passed),
        "claims_stable_grasp_policy": False,
        "note": (
            "Physical dynamics after a single demo-root restore; no per-step snap/weld. "
            "Transport records measured hand/peg lateral travel. "
            "Not a learned grasp policy. Formal multi-family scripted physical grasp not claimed."
        ),
    }
