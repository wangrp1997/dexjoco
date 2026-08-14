"""P0-S0.4c hardened: multi-family physical grasp with pre-collection regressions.

Regressions vs first S0.4c pass:
1. After establish+settle, capture MjData snapshot; open/closed restore the same state.
2. Explicit snap_call_count_after_establish == 0 during dynamics phases.
3. Transport records actual hand/peg lateral displacement and requires a min distance.

Naming: multi-family oracle-established physical grasp recipe smoke.
Does NOT claim a learned grasp policy. Still bans collection/training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from embodied_grasp_insertion.geometry.family_spec import GeometryFamilySpec
from embodied_grasp_insertion.physics.grasp_metrics import (
    REFERENCE_BODY,
    object_in_hand_pose,
    peg_hand_contact_counts,
    relative_pose_error,
)
from embodied_grasp_insertion.physics.grasp_stability import (
    characteristic_length_m,
    make_action,
    peg_table_contact,
)
from embodied_grasp_insertion.physics.grasp_stability_physical import (
    PhaseResult,
    scale_physical_thresholds,
)

_DEMO_T = np.array([0.1723988118047515, 0.024430899378065415, -0.18087802516297768], dtype=np.float64)
_DEMO_RV = np.array([-0.30202356373350836, -0.5937349338665724, 0.10290819104341495], dtype=np.float64)
_DEMO_F = np.array(
    [
        0.46070992946624756,
        1.4446288347244263,
        1.3756532669067383,
        0.027738912031054497,
        0.2946900427341461,
        1.3766905069351196,
        1.254265308380127,
        -0.1402033120393753,
        0.24894791841506958,
        1.4296174049377441,
        1.2697280645370483,
        -0.08462505787611008,
        1.164820671081543,
        0.6026830077171326,
        0.7775329351425171,
        1.0722956657409668,
    ],
    dtype=np.float64,
)
_OPEN = np.zeros(16, dtype=np.float64)

FAMILY_ESTABLISH: dict[str, dict[str, float]] = {
    "round_16mm": {"t_scale": 1.0, "finger_squeeze": 0.10, "t_y_extra": 0.0},
    "rectangular_16mm": {"t_scale": 0.85, "finger_squeeze": 0.12, "t_y_extra": 0.0},
    "rectangular_8mm": {"t_scale": 0.75, "finger_squeeze": 0.08, "t_y_extra": 0.0},
    "round_8mm": {"t_scale": 1.0, "finger_squeeze": 0.05, "t_y_extra": 0.0},
}


@dataclass
class SnapCounter:
    """Count peg teleports; after establish must stay at 0."""

    count: int = 0
    enabled: bool = True

    def reset(self) -> None:
        self.count = 0

    def mark(self) -> None:
        if self.enabled:
            self.count += 1


# When set, every _snap_peg_in_hand call increments it (detect post-establish snaps).
_ACTIVE_SNAP_COUNTER: SnapCounter | None = None


@dataclass
class GymMjSnapshot:
    """Minimal MuJoCo state restore for formal gym env (open/closed matched branch)."""

    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    time: float
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    act: np.ndarray | None = None
    rpos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rquat: np.ndarray = field(default_factory=lambda: np.zeros(4))
    fingers: np.ndarray = field(default_factory=lambda: np.zeros(16))

    @classmethod
    def capture(cls, env, *, rpos: np.ndarray, rquat: np.ndarray, fingers: np.ndarray) -> "GymMjSnapshot":
        data = env.data
        act = None
        if getattr(data, "act", None) is not None and data.act.size:
            act = np.asarray(data.act, dtype=np.float64).copy()
        return cls(
            qpos=np.asarray(data.qpos, dtype=np.float64).copy(),
            qvel=np.asarray(data.qvel, dtype=np.float64).copy(),
            ctrl=np.asarray(data.ctrl, dtype=np.float64).copy(),
            time=float(data.time),
            mocap_pos=np.asarray(data.mocap_pos, dtype=np.float64).copy(),
            mocap_quat=np.asarray(data.mocap_quat, dtype=np.float64).copy(),
            act=act,
            rpos=np.asarray(rpos, dtype=np.float64).copy(),
            rquat=np.asarray(rquat, dtype=np.float64).copy(),
            fingers=np.asarray(fingers, dtype=np.float64).copy(),
        )

    def restore(self, env) -> None:
        data = env.data
        np.copyto(data.qpos, self.qpos)
        np.copyto(data.qvel, self.qvel)
        np.copyto(data.ctrl, self.ctrl)
        data.time = float(self.time)
        np.copyto(data.mocap_pos, self.mocap_pos)
        np.copyto(data.mocap_quat, self.mocap_quat)
        if self.act is not None and getattr(data, "act", None) is not None and data.act.size:
            np.copyto(data.act, self.act)
        mujoco.mj_forward(env.model, data)


def _palm_xyz(env) -> np.ndarray:
    return np.asarray(env.data.xpos[int(env.model.body(REFERENCE_BODY).id)], dtype=np.float64).copy()


def _palm_z(env) -> float:
    return float(_palm_xyz(env)[2])


def _peg_xyz(env) -> np.ndarray:
    return np.asarray(env.data.xpos[env._peg_body_id], dtype=np.float64).copy()


def _peg_speed(env) -> float:
    v = np.asarray(env.data.cvel[env._peg_body_id], dtype=np.float64)
    return float(np.linalg.norm(v[3:]))


def _snap_peg_in_hand(
    env,
    local_t: np.ndarray,
    local_rv: np.ndarray,
    *,
    counter: SnapCounter | None = None,
) -> None:
    if counter is not None:
        counter.mark()
    if _ACTIVE_SNAP_COUNTER is not None:
        _ACTIVE_SNAP_COUNTER.mark()
    bid = int(env.model.body(REFERENCE_BODY).id)
    palm_pos = np.asarray(env.data.xpos[bid], dtype=np.float64)
    palm_quat = np.asarray(env.data.xquat[bid], dtype=np.float64)
    Rp = R.from_quat(palm_quat, scalar_first=True)
    peg_pos = palm_pos + Rp.apply(np.asarray(local_t, dtype=np.float64))
    peg_quat = (Rp * R.from_rotvec(np.asarray(local_rv, dtype=np.float64))).as_quat(scalar_first=True)
    env._set_free_joint_pose(env._peg_qpos_adr, env._peg_qvel_adr, peg_pos, peg_quat)
    mujoco.mj_forward(env.model, env.data)


def _closed_fingers(squeeze: float) -> np.ndarray:
    f = _DEMO_F.copy()
    idx = [1, 2, 5, 6, 9, 10, 13, 14, 15]
    f[idx] = np.clip(f[idx] + float(squeeze), -0.2, 1.7)
    return f


@dataclass
class EstablishState:
    rpos: np.ndarray
    rquat: np.ndarray
    fingers: np.ndarray
    local_t: np.ndarray
    local_rv: np.ndarray
    root_contacts: int
    snap_calls_during_establish: int


def establish_oracle_once(
    env,
    spec: GeometryFamilySpec,
    *,
    approach_dz: float = 0.25,
    approach_steps: int = 110,
    close_steps: int = 55,
    settle_steps: int = 25,
    snap_counter: SnapCounter | None = None,
) -> EstablishState:
    """Oracle establish flow (may snap many times). Dynamics phases must not snap."""
    kn = FAMILY_ESTABLISH.get(spec.family_id, {"t_scale": 1.0, "finger_squeeze": 0.10, "t_y_extra": 0.0})
    local_t = _DEMO_T * float(kn["t_scale"])
    local_t = local_t.copy()
    local_t[1] += float(kn.get("t_y_extra", 0.0))
    local_rv = _DEMO_RV.copy()
    fingers = _closed_fingers(float(kn["finger_squeeze"]))
    ctr = snap_counter or SnapCounter()
    ctr.reset()

    peg0 = np.asarray(env.data.xpos[env._peg_body_id], dtype=np.float64).copy()
    rquat = np.asarray(env.data.mocap_quat[env._mocap_right_id], dtype=np.float64).copy()
    rpos = peg0 + np.array([0.0, 0.0, approach_dz], dtype=np.float64)

    for _ in range(approach_steps):
        env.step(make_action(env, right_pos=rpos, right_quat_wxyz=rquat, right_fingers=_OPEN))
        _snap_peg_in_hand(env, local_t, local_rv, counter=ctr)
    for k in range(close_steps):
        a = (k + 1) / float(close_steps)
        ff = (1.0 - a) * _OPEN + a * fingers
        env.step(make_action(env, right_pos=rpos, right_quat_wxyz=rquat, right_fingers=ff))
        _snap_peg_in_hand(env, local_t, local_rv, counter=ctr)
    for _ in range(settle_steps):
        env.step(make_action(env, right_pos=rpos, right_quat_wxyz=rquat, right_fingers=fingers))
        _snap_peg_in_hand(env, local_t, local_rv, counter=ctr)
    _snap_peg_in_hand(env, local_t, local_rv, counter=ctr)
    root_c = int(peg_hand_contact_counts(env).total)
    return EstablishState(
        rpos=rpos,
        rquat=rquat,
        fingers=fingers,
        local_t=local_t,
        local_rv=local_rv,
        root_contacts=root_c,
        snap_calls_during_establish=int(ctr.count),
    )


def _run_gym_actions(
    env,
    *,
    rpos: np.ndarray,
    rquat: np.ndarray,
    fingers: np.ndarray,
    n_steps: int,
    move_per_step: np.ndarray | None = None,
    track_lateral: bool = False,
) -> dict[str, Any]:
    o0 = object_in_hand_pose(env)
    peg0 = _peg_xyz(env)
    palm0 = _palm_xyz(env)
    z0 = float(peg0[2])
    hz0 = float(palm0[2])
    max_dt = max_dr = 0.0
    min_contacts = 10**9
    table_hits = 0
    max_speed = 0.0
    pos = np.asarray(rpos, dtype=np.float64).copy()
    for _ in range(n_steps):
        if move_per_step is not None:
            pos = pos + np.asarray(move_per_step, dtype=np.float64)
        env.step(make_action(env, right_pos=pos, right_quat_wxyz=rquat, right_fingers=fingers))
        dt, dr = relative_pose_error(o0, object_in_hand_pose(env))
        max_dt = max(max_dt, dt)
        max_dr = max(max_dr, dr)
        min_contacts = min(min_contacts, int(peg_hand_contact_counts(env).total))
        table_hits += int(peg_table_contact(env.model, env.data, env._peg_body_id))
        max_speed = max(max_speed, _peg_speed(env))
    peg1 = _peg_xyz(env)
    palm1 = _palm_xyz(env)
    out: dict[str, Any] = {
        "hand_dz": float(palm1[2] - hz0),
        "peg_dz": float(peg1[2] - z0),
        "peg_z": float(peg1[2]),
        "max_rel_trans": max_dt,
        "max_rel_rot": max_dr,
        "min_contacts": int(min_contacts if min_contacts < 10**9 else 0),
        "contacts_end": int(peg_hand_contact_counts(env).total),
        "table_hits": int(table_hits),
        "max_speed": max_speed,
        "drop_z": max(0.0, z0 - float(peg1[2])),
        "rpos": pos,
    }
    if track_lateral:
        hand_dxy = palm1[:2] - palm0[:2]
        peg_dxy = peg1[:2] - peg0[:2]
        out["hand_dy"] = float(hand_dxy[1])
        out["peg_dy"] = float(peg_dxy[1])
        out["hand_lateral_m"] = float(np.linalg.norm(hand_dxy))
        out["peg_lateral_m"] = float(np.linalg.norm(peg_dxy))
        out["commanded_lateral_m"] = (
            float(abs(float(move_per_step[1]) * n_steps)) if move_per_step is not None else 0.0
        )
    return out


def run_physical_formal_family(
    env,
    spec: GeometryFamilySpec,
    *,
    hold_steps: int = 30,
    lift_chunks: int = 20,
    lift_dz: float = 0.005,
    lift_substeps: int = 12,
    transport_steps: int = 50,
    transport_dy: float = 0.0025,
    neg_steps: int = 45,
) -> dict[str, Any]:
    """Physical gate on formal arena via oracle-establish + matched snapshot branches."""
    thr = scale_physical_thresholds(characteristic_length_m(spec))
    thr = dict(thr)
    thr["hold_trans_tol_m"] = max(thr["hold_trans_tol_m"], 0.10)
    thr["hold_rot_tol_rad"] = max(thr["hold_rot_tol_rad"], 0.45)
    thr["lift_trans_tol_m"] = max(thr["lift_trans_tol_m"], 0.08)
    thr["lift_rot_tol_rad"] = max(thr["lift_rot_tol_rad"], 0.35)
    thr["lift_follow_ratio_min"] = 0.55
    thr["transport_trans_tol_m"] = max(thr["transport_trans_tol_m"], 0.08)
    thr["transport_min_hand_lateral_m"] = 0.03
    thr["transport_lateral_follow_ratio_min"] = 0.55

    establish_ctr = SnapCounter()
    global _ACTIVE_SNAP_COUNTER
    _ACTIVE_SNAP_COUNTER = None
    env.reset(seed=0)
    est = establish_oracle_once(env, spec, snap_counter=establish_ctr)
    if est.root_contacts < thr["min_contacts"]:
        return {
            "passed": False,
            "family_id": spec.family_id,
            "thresholds": thr,
            "root_contacts": est.root_contacts,
            "snap_calls_during_establish": est.snap_calls_during_establish,
            "snap_call_count_after_establish": None,
            "phases": [],
            "fixture": "oracle_establish_flow_then_dynamics",
            "claims_physical_grasp_stability": False,
            "claims_stable_grasp_policy": False,
            "note": "establish produced insufficient hand–peg contacts",
        }

    rpos, rquat, fingers = est.rpos, est.rquat, est.fingers

    post_ctr = SnapCounter()
    post_ctr.reset()
    _ACTIVE_SNAP_COUNTER = post_ctr
    try:
        for _ in range(40):
            env.step(make_action(env, right_pos=rpos, right_quat_wxyz=rquat, right_fingers=fingers))
        if int(peg_hand_contact_counts(env).total) < thr["min_contacts"]:
            return {
                "passed": False,
                "family_id": spec.family_id,
                "thresholds": thr,
                "root_contacts": est.root_contacts,
                "contacts_after_settle": int(peg_hand_contact_counts(env).total),
                "snap_calls_during_establish": est.snap_calls_during_establish,
                "snap_call_count_after_establish": int(post_ctr.count),
                "phases": [],
                "fixture": "oracle_establish_flow_then_dynamics",
                "claims_physical_grasp_stability": False,
                "claims_stable_grasp_policy": False,
                "note": "lost contacts during post-oracle settle",
            }

        root_snap = GymMjSnapshot.capture(env, rpos=rpos, rquat=rquat, fingers=fingers)
        phases: list[PhaseResult] = []

        hold_m = _run_gym_actions(env, rpos=rpos, rquat=rquat, fingers=fingers, n_steps=hold_steps)
        rpos = hold_m.pop("rpos")
        hold_ok = (
            hold_m["min_contacts"] >= thr["min_contacts"]
            and hold_m["max_rel_trans"] <= thr["hold_trans_tol_m"]
            and hold_m["max_rel_rot"] <= thr["hold_rot_tol_rad"]
            and hold_m["table_hits"] == 0
            and hold_m["max_speed"] <= thr["max_peg_speed_m_s"]
        )
        phases.append(PhaseResult("hold", hold_ok, hold_m))

        o_lift0 = object_in_hand_pose(env)
        z_lift0 = float(env.data.xpos[env._peg_body_id, 2])
        hz0 = _palm_z(env)
        max_dt = max_dr = 0.0
        min_c = 10**9
        table = 0
        max_speed = 0.0
        for _ in range(lift_chunks):
            rpos = rpos + np.array([0.0, 0.0, lift_dz], dtype=np.float64)
            for _s in range(lift_substeps):
                env.step(make_action(env, right_pos=rpos, right_quat_wxyz=rquat, right_fingers=fingers))
                dt, dr = relative_pose_error(o_lift0, object_in_hand_pose(env))
                max_dt = max(max_dt, dt)
                max_dr = max(max_dr, dr)
                min_c = min(min_c, int(peg_hand_contact_counts(env).total))
                table += int(peg_table_contact(env.model, env.data, env._peg_body_id))
                max_speed = max(max_speed, _peg_speed(env))
            if (_palm_z(env) - hz0) >= thr["lift_min_hand_dz_m"] + 0.01:
                break
        for _ in range(25):
            env.step(make_action(env, right_pos=rpos, right_quat_wxyz=rquat, right_fingers=fingers))
            dt, dr = relative_pose_error(o_lift0, object_in_hand_pose(env))
            max_dt = max(max_dt, dt)
            max_dr = max(max_dr, dr)
            min_c = min(min_c, int(peg_hand_contact_counts(env).total))
            table += int(peg_table_contact(env.model, env.data, env._peg_body_id))
            max_speed = max(max_speed, _peg_speed(env))
        hand_dz = _palm_z(env) - hz0
        peg_dz = float(env.data.xpos[env._peg_body_id, 2]) - z_lift0
        lift_m = {
            "hand_dz": hand_dz,
            "peg_dz": peg_dz,
            "peg_z": float(env.data.xpos[env._peg_body_id, 2]),
            "max_rel_trans": max_dt,
            "max_rel_rot": max_dr,
            "min_contacts": int(min_c if min_c < 10**9 else 0),
            "contacts_end": int(peg_hand_contact_counts(env).total),
            "table_hits": int(table),
            "max_speed": max_speed,
        }
        follow_ok = hand_dz >= thr["lift_min_hand_dz_m"] and peg_dz >= thr["lift_follow_ratio_min"] * hand_dz
        lift_ok = (
            follow_ok
            and lift_m["min_contacts"] >= thr["min_contacts"]
            and lift_m["max_rel_trans"] <= thr["lift_trans_tol_m"]
            and lift_m["max_rel_rot"] <= thr["lift_rot_tol_rad"]
            and lift_m["table_hits"] == 0
            and lift_m["max_speed"] <= thr["max_peg_speed_m_s"]
        )
        phases.append(PhaseResult("lift", lift_ok, lift_m))

        tr_m = _run_gym_actions(
            env,
            rpos=rpos,
            rquat=rquat,
            fingers=fingers,
            n_steps=transport_steps,
            move_per_step=np.array([0.0, transport_dy, 0.0], dtype=np.float64),
            track_lateral=True,
        )
        rpos = tr_m.pop("rpos")
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

        root_snap.restore(env)
        open_m = _run_gym_actions(
            env,
            rpos=root_snap.rpos,
            rquat=root_snap.rquat,
            fingers=_OPEN,
            n_steps=neg_steps,
        )
        open_m.pop("rpos", None)
        neg_ok = (
            (open_m["drop_z"] >= thr["neg_drop_delta_z_m"])
            or (open_m["max_rel_trans"] >= thr["neg_trans_min_m"])
            or (
                open_m["min_contacts"] < thr["min_contacts"]
                and (open_m["table_hits"] > 0 or open_m["max_rel_rot"] >= 0.5)
            )
        )
        phases.append(PhaseResult("open_hand_negative", neg_ok, open_m))

        root_snap.restore(env)
        closed_m = _run_gym_actions(
            env,
            rpos=root_snap.rpos,
            rquat=root_snap.rquat,
            fingers=root_snap.fingers,
            n_steps=neg_steps,
        )
        closed_m.pop("rpos", None)
        closed_beats = (
            closed_m["drop_z"] + thr["closed_better_drop_margin_m"] <= open_m["drop_z"]
        ) or (
            closed_m["max_rel_trans"] + thr["closed_better_drift_margin_m"]
            <= open_m["max_rel_trans"]
        )
        if open_m["min_contacts"] < thr["min_contacts"] and closed_m["min_contacts"] >= thr["min_contacts"]:
            closed_beats = True

        snap_after = int(post_ctr.count)
        no_snap_ok = snap_after == 0
        if not no_snap_ok:
            raise AssertionError(f"snap_call_count_after_establish={snap_after} != 0")

        passed = bool(pos_ok and neg_ok and closed_beats and no_snap_ok)
        return {
            "passed": passed,
            "positive_ok": bool(pos_ok),
            "negative_ok": bool(neg_ok),
            "closed_beats_open": bool(closed_beats),
            "no_snap_after_establish": bool(no_snap_ok),
            "family_id": spec.family_id,
            "thresholds": thr,
            "root_contacts": est.root_contacts,
            "snap_calls_during_establish": est.snap_calls_during_establish,
            "snap_call_count_after_establish": snap_after,
            "matched_snapshot_branch": True,
            "closed_matched": closed_m,
            "phases": [{"name": p.name, "passed": p.passed, "metrics": p.metrics} for p in phases],
            "fixture": "oracle_establish_flow_then_dynamics",
            "establish": FAMILY_ESTABLISH.get(spec.family_id, {}),
            "claims_physical_grasp_stability": bool(passed),
            "claims_stable_grasp_policy": False,
            "note": (
                "Establish may snap repeatedly; after settle a MjData snapshot is captured. "
                "hold/lift/transport and open/closed restore that snapshot with "
                "snap_call_count_after_establish==0. Transport requires measured hand/peg "
                "lateral travel. Not a learned grasp."
            ),
        }
    finally:
        _ACTIVE_SNAP_COUNTER = None
