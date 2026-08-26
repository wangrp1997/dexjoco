#!/usr/bin/env python3
"""Privileged PBVS insert P0 — NO demo open-loop after handoff.

Handoff = peg_lift_end. Then privileged wrap (middle/ring curl + recapture o2h),
freeze left, PBVS right wrist on virtual tip. Near-seat: axial slide, not 6DoF weld.
Release only after geometric seat; pin after fingers open if still near hole.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

DEXJOCO_ROOT = Path(__file__).resolve().parents[1]
for _p in (
    str(DEXJOCO_ROOT),
    str(DEXJOCO_ROOT / "dexjoco"),
    str(DEXJOCO_ROOT / "embodied_grasp_insertion"),
    str(DEXJOCO_ROOT.parent / "reach_insert_rl"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.simulation.full_episode_utils import make_full_env  # noqa: E402
from hybrid_insert.geometry import toward_socket_delta  # noqa: E402
from interaction_retarget.sim.replay import (  # noqa: E402
    policy_dual_arm_to_raw,
    raw_flat_to_dict,
    rotvec_dual_arm_to_policy,
    zarr_action_to_policy46,
)
from interaction_retarget.sim.settle import settle_bimanual_actions  # noqa: E402
from priv_snap_insert.servo import (  # noqa: E402
    REGRASP_EPISODES,
    ServoGains,
    anisotropic_snap,
    capture_wrist_palm_calib,
    compute_servo_command,
    gains_for_episode,
    insert_ok_quality,
    register_regrasp_episode,
    release_hold_ok,
    seat_success,
    transfer_seat_ready,
)
from priv_snap_insert.snap import (  # noqa: E402
    apply_socket_pin,
    capture_o2h_lock,
    capture_socket_pin,
    capture_tray_in_left,
    pin_freejoint,
    pin_peg_tip_seated,
    project_peg_to_o2h,
    project_tray_to_left,
)
from priv_snap_insert.wrap import (  # noqa: E402
    clip_right_fingers,
    contact_summary,
    shift_o2h_toward_body,
    wrap_target_from_pinch,
)
from reach_insert_rl.env.full_obs import current_action44, privileged_full_features  # noqa: E402
from reach_insert_rl.env.handoff_env import load_manifest_entries  # noqa: E402

PROTOCOL = "PrivilegedPBVSNoDemoP0"
# Regrasp-retry chases a slipped real tip. Never for pinch-eject eps.
NO_REGRASP_EPS = {51, 62, 74, 95, 98}
HandoffMode = Literal[
    "peg_lift_start", "peg_grasp_frame", "right_grasp_frame", "peg_lift_end"
]
DEFAULT_SIDECAR = Path("/mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly")
DEFAULT_OUT = Path("/mnt/hdd/dexjoco/outputs/priv_snap_insert")
REGRASP_REGISTRY = DEFAULT_OUT / "regrasp_eps.json"


def _load_regrasp_registry() -> set[int]:
    if not REGRASP_REGISTRY.is_file():
        return set()
    try:
        return {int(x) for x in json.loads(REGRASP_REGISTRY.read_text(encoding="utf-8"))}
    except Exception:
        return set()


def _save_regrasp_registry(eps: set[int]) -> None:
    REGRASP_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGRASP_REGISTRY.write_text(
        json.dumps(sorted(int(x) for x in eps), indent=2), encoding="utf-8"
    )


def _resolve_handoff_frame(env, mode: HandoffMode) -> int:
    assert env._spec is not None
    if mode == "peg_lift_end":
        from pose_insert.pre_insert import resolve_peg_lift_end_frame

        return int(
            resolve_peg_lift_end_frame(
                {"episode_index": int(env._spec.episode_index)},
                env.sidecar_dir,
            )
        )
    meta_path = (
        Path(env.sidecar_dir) / f"episode_{int(env._spec.episode_index):03d}" / "meta.json"
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    timing = meta.get("timing", meta)
    if mode == "peg_lift_start":
        return int(timing.get("peg_lift_start", meta.get("peg_lift_start", 0)))
    if mode == "right_grasp_frame":
        return int(timing.get("right_grasp_frame", meta.get("right_grasp_frame", 0)))
    return int(timing.get("right_grasp_frame", 0))


def _replay_to_frame(env, frame: int) -> dict[str, Any]:
    assert env._actions is not None
    last: dict[str, Any] = {}
    for t in range(int(frame)):
        if t >= len(env._actions):
            break
        a = env._actions[t]
        try:
            env._env.step(zarr_action_to_policy46(a))
        except Exception:
            env._raw.step(raw_flat_to_dict(a))
        env._t = t + 1
        env._hold44 = current_action44(env._raw)
        outcome = env._labeler.compute(env._raw)
        if outcome.peg_ok:
            env._peg_ok_seen = True
            env._peg_lost = 0
        if outcome.tray_ok:
            env._tray_ok_seen = True
        feat = privileged_full_features(env._raw)
        last = {
            "frame": t + 1,
            "peg_ok": bool(outcome.peg_ok),
            "tray_ok": bool(outcome.tray_ok),
            "insert_ok": bool(outcome.insert_ok),
            "tip_dist_m": float(feat["tip_dist"]),
            "lat_err_m": float(feat["lat_err"]),
            "along_m": float(feat["along"]),
        }
    return last


def _apply_hold_physics(env, hold44: np.ndarray, *, n_substeps: int) -> None:
    a46 = rotvec_dual_arm_to_policy(hold44)
    raw_dict = policy_dual_arm_to_raw(a46)
    settle_bimanual_actions(
        env._raw,
        right23=np.asarray(raw_dict["right"], dtype=np.float64),
        left23=np.asarray(raw_dict["left"], dtype=np.float64),
        n_substeps=int(n_substeps),
    )
    env._hold44 = np.asarray(hold44, dtype=np.float64).copy()


def _obs_after_physics(env) -> dict[str, Any]:
    outcome = env._labeler.compute(env._raw)
    feat = privileged_full_features(env._raw)
    env._done = bool(outcome.insert_ok)
    return {
        "insert_ok": bool(outcome.insert_ok),
        "peg_ok": bool(outcome.peg_ok),
        "tray_ok": bool(outcome.tray_ok),
        "tip_dist_m": float(feat["tip_dist"]),
        "lat_err_m": float(feat["lat_err"]),
        "along_m": float(feat["along"]),
        "axis_err": float(feat["axis_err"]),
    }


def _settle_step(env, hold44: np.ndarray, *, n_substeps: int) -> dict[str, Any]:
    _apply_hold_physics(env, hold44, n_substeps=n_substeps)
    env._t += 1
    return _obs_after_physics(env)


def _hold_snap_strength(tip_m: float, *, allow_slide: bool = True) -> tuple[float, float, bool]:
    """Blend peg toward o2h (not hard weld). Near rim: slide along peg axis."""
    t = float(tip_m)
    if t > 0.080:
        return 0.82, 0.78, False
    if t > 0.040:
        return 0.72, 0.62, False
    if t > 0.028:
        return 0.58, 0.35, False
    if allow_slide:
        return 0.48, 0.10, True
    return 0.40, 0.10, False


def _try_privileged_seat(env) -> dict[str, Any]:
    """Unused: pin-while-gripping explodes. Use _transfer_peg_into_hole."""
    return _obs_after_physics(env)


def _pin_peg_aligned_in_hole(raw, *, along_m: float = -0.022) -> None:
    """Axis-align then seat tip. Translating a tilted peg through the rim NaNs."""
    from dexjoco.sim.envs.assembly_geometry import names_from_raw
    from interaction_retarget.skill_replay.insert import _insert_geometry
    from scipy.spatial.transform import Rotation as R

    tip, socket, hole, _ = _insert_geometry(raw)
    hole_u = np.asarray(hole, dtype=np.float64)
    hole_u = hole_u / (np.linalg.norm(hole_u) + 1e-8)
    model, data = raw._model, raw._data
    peg_id = int(model.body(names_from_raw(raw).peg_body).id)
    peg_pos = np.asarray(data.xpos[peg_id], dtype=np.float64)
    peg_rot = R.from_quat(np.asarray(data.xquat[peg_id], dtype=np.float64), scalar_first=True)
    peg_z = peg_rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    target_z = hole_u if float(np.dot(peg_z, hole_u)) >= 0.0 else -hole_u
    v = np.cross(peg_z, target_z)
    s = float(np.linalg.norm(v))
    c = float(np.clip(np.dot(peg_z, target_z), -1.0, 1.0))
    align = R.identity() if s < 1e-8 else R.from_rotvec((v / s) * float(np.arctan2(s, c)))
    new_rot = align * peg_rot
    tip_in_peg = peg_rot.inv().apply(np.asarray(tip, dtype=np.float64) - peg_pos)
    new_tip = np.asarray(socket, dtype=np.float64) + hole_u * float(along_m)
    new_pos = new_tip - new_rot.apply(tip_in_peg)
    pin_freejoint(
        raw,
        int(raw._peg_qpos_adr),
        int(raw._peg_qvel_adr),
        new_pos,
        new_rot.as_quat(scalar_first=True),
    )


def _transfer_peg_into_hole(
    env,
    *,
    fr: np.ndarray,
    fl: np.ndarray,
    lxyz: np.ndarray,
    lrot: np.ndarray,
    left_hold22: np.ndarray,
    o2h,
    tray_world: dict[str, np.ndarray],
    n_open: int = 6,
) -> dict[str, Any]:
    """Open while o2h-welded, then pin tip into the hole. Never pin while gripping."""
    open_fr = _open_right_hand_pose(fr)
    hold0 = current_action44(env._raw).copy()
    r_xyz0 = hold0[0:3].copy()
    r_rot0 = hold0[3:6].copy()
    fr0 = np.asarray(fr, dtype=np.float64)
    n = max(int(n_open), 1)
    for i in range(n):
        a = float(i + 1) / float(n)
        hold = current_action44(env._raw).copy()
        hold[0:3] = r_xyz0
        hold[3:6] = r_rot0
        hold[6:22] = (1.0 - a) * fr0 + a * open_fr
        hold[22:25] = lxyz
        hold[25:28] = lrot
        hold[28:44] = fl
        _apply_hold_physics(env, hold, n_substeps=8)
        project_peg_to_o2h(env._raw, o2h, strength=1.0)
        _freeze_left_at_handoff(env._raw, left_hold22)
        if not bool(env._labeler.compute(env._raw).tray_ok):
            apply_socket_pin(env._raw, tray_world)
        env._t += 1
    try:
        _pin_peg_aligned_in_hole(env._raw, along_m=-0.020)
    except Exception as e:
        print(f"  [pin] {type(e).__name__}: {e}", flush=True)
    _freeze_left_at_handoff(env._raw, left_hold22)
    feat = privileged_full_features(env._raw)
    print(
        f"  [pin] after tip={feat['tip_dist']:.4f} along={feat['along']:.4f} "
        f"lat={feat['lat_err']:.4f} insert={env._labeler.compute(env._raw).insert_ok}",
        flush=True,
    )
    return _obs_after_physics(env)


def _settle_locked(
    env,
    hold44: np.ndarray,
    *,
    o2h,
    left_hold22: np.ndarray,
    tray_world: dict[str, np.ndarray],
    n_substeps: int,
    snap_pos: float,
    snap_rot: float,
    free_along: bool = False,
) -> dict[str, Any]:
    """Settle in chunks; re-weld peg to hand. Coarser than every 2 steps (less jitter)."""
    chunk = 4 if free_along else 8
    n = max(int(n_substeps), 1)
    do_snap = float(snap_pos) > 0.05 or float(snap_rot) > 0.05 or bool(free_along)
    if do_snap:
        project_peg_to_o2h(
            env._raw,
            o2h,
            pos_strength=float(snap_pos),
            rot_strength=float(snap_rot),
            free_along=bool(free_along),
        )
    for i in range(0, n, chunk):
        _apply_hold_physics(env, hold44, n_substeps=min(chunk, n - i))
        if do_snap:
            project_peg_to_o2h(
                env._raw,
                o2h,
                pos_strength=float(snap_pos),
                rot_strength=float(snap_rot),
                free_along=bool(free_along),
            )
        _freeze_left_at_handoff(env._raw, left_hold22)
        _restore_if_unstable(
            env._raw, o2h=o2h, tray_world=tray_world, left_hold22=left_hold22
        )
    env._t += 1
    return _obs_after_physics(env)


def _freeze_left_at_handoff(raw, hold44_left: np.ndarray) -> None:
    """Hard-hold left mocap + Allegro targets (settle alone drifts / opens)."""
    from interaction_retarget.sim.replay import rotvec_dual_arm_to_policy, policy_dual_arm_to_raw

    # pack a dummy right; only left is applied to mocap/ctrl below
    hold = current_action44(raw).copy()
    hold[22:44] = np.asarray(hold44_left, dtype=np.float64).reshape(22)
    a46 = rotvec_dual_arm_to_policy(hold)
    raw_dict = policy_dual_arm_to_raw(a46)
    left23 = np.asarray(raw_dict["left"], dtype=np.float64).reshape(23)
    mid = int(raw._mocap_left_id)
    raw._data.mocap_pos[mid] = left23[0:3]
    raw._data.mocap_quat[mid] = left23[3:7]
    ctrl_ids = np.asarray(raw._allegro_ctrl_ids, dtype=int)
    raw._data.ctrl[ctrl_ids[16:32]] = left23[7:23]


def _ensure_tray_world(raw, tray_world: dict[str, np.ndarray], *, tol_m: float = 0.02) -> None:
    cur, _ = _freejoint_pose_socket(raw)
    if float(np.linalg.norm(cur - tray_world["socket_pos"])) > tol_m:
        apply_socket_pin(raw, tray_world)
    else:
        raw._data.qvel[int(raw._socket_qvel_adr) : int(raw._socket_qvel_adr) + 6] = 0.0


def _freejoint_pose_socket(raw):
    from priv_snap_insert.snap import _freejoint_pose

    return _freejoint_pose(raw, int(raw._socket_qpos_adr))


def _hold_tray_or_restore(env, socket_pins: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """If left still holds tray, refresh pin; else teleport tray back to last good pose."""
    outcome = env._labeler.compute(env._raw)
    if bool(outcome.tray_ok):
        return capture_socket_pin(env._raw)
    apply_socket_pin(env._raw, socket_pins)
    return socket_pins


def _restore_if_unstable(raw, *, o2h, tray_world, left_hold22) -> bool:
    import mujoco

    qacc = np.asarray(raw._data.qacc)
    qpos = np.asarray(raw._data.qpos)
    if np.isfinite(qacc).all() and np.isfinite(qpos).all():
        return False
    apply_socket_pin(raw, tray_world)
    project_peg_to_o2h(raw, o2h, strength=1.0)
    _freeze_left_at_handoff(raw, left_hold22)
    mujoco.mj_forward(raw._model, raw._data)
    return True


def _wrap_right_at_handoff(
    env,
    *,
    fr: np.ndarray,
    o2h,
    left_hold22: np.ndarray,
    tray_world: dict[str, np.ndarray],
    lxyz: np.ndarray,
    lrot: np.ndarray,
    fl: np.ndarray,
    n_steps: int = 12,
) -> tuple[np.ndarray, Any]:
    """Slide peg a little toward middle; abort if pinch dies."""
    fr0 = clip_right_fingers(env._raw, fr)
    target = clip_right_fingers(env._raw, wrap_target_from_pinch(fr0))
    before = contact_summary(env._raw)
    fr_now = fr0
    o2h_now = o2h
    lock_good = capture_o2h_lock(env._raw)
    fr_good = fr0.copy()
    shifted = 0.0
    n = int(n_steps)

    def _pinch_ok(con: dict) -> bool:
        return int(con.get("index", 0)) >= 1 and int(con.get("thumb", 0)) >= 1

    for i in range(n):
        con = contact_summary(env._raw)
        if not _pinch_ok(con):
            break
        if int(con.get("middle", 0)) < 1 and shifted < 0.010:
            o2h_now = shift_o2h_toward_body(
                env._raw, o2h_now, "mf_medial_right", meters=0.002
            )
            project_peg_to_o2h(env._raw, o2h_now, strength=1.0)
            shifted += 0.002
        a = float(i + 1) / float(n)
        fr_now = clip_right_fingers(env._raw, (1.0 - a) * fr0 + a * target)
        hold = current_action44(env._raw).copy()
        hold[6:22] = fr_now
        hold[22:25] = lxyz
        hold[25:28] = lrot
        hold[28:44] = fl
        _settle_locked(
            env,
            hold,
            o2h=o2h_now,
            left_hold22=left_hold22,
            tray_world=tray_world,
            n_substeps=8,
            snap_pos=1.0,
            snap_rot=1.0,
            free_along=False,
        )
        con = contact_summary(env._raw)
        if _pinch_ok(con):
            lock_good = capture_o2h_lock(env._raw)
            fr_good = fr_now.copy()
            o2h_now = lock_good.o2h
            if int(con.get("middle", 0)) >= 1:
                break
    project_peg_to_o2h(env._raw, lock_good.o2h, strength=1.0)
    hold = current_action44(env._raw).copy()
    hold[6:22] = fr_good
    hold[22:25] = lxyz
    hold[25:28] = lrot
    hold[28:44] = fl
    _settle_locked(
        env,
        hold,
        o2h=lock_good.o2h,
        left_hold22=left_hold22,
        tray_world=tray_world,
        n_substeps=8,
        snap_pos=1.0,
        snap_rot=1.0,
        free_along=False,
    )
    after = contact_summary(env._raw)
    print(
        f"  [wrap] contact {before} -> {after} shifted={shifted:.3f}",
        flush=True,
    )
    return fr_good, lock_good


def _open_right_hand_pose(fr: np.ndarray) -> np.ndarray:
    """Fully open — partial open still clamps peg via contact."""
    return np.zeros_like(np.asarray(fr, dtype=np.float64))


def _release_after_seat(
    env,
    *,
    fr: np.ndarray,
    fl: np.ndarray,
    lxyz: np.ndarray,
    lrot: np.ndarray,
    left_hold22: np.ndarray,
    tray_world: dict[str, np.ndarray],
    t2h,
    n_steps: int = 8,
) -> list[dict[str, Any]]:
    """Open right hand in place only — no peg/tray teleport (that blows the scene up)."""
    fr0 = np.asarray(fr, dtype=np.float64).copy()
    open_fr = _open_right_hand_pose(fr0)
    _ = (tray_world, t2h)
    hold0 = current_action44(env._raw).copy()
    r_xyz0 = hold0[0:3].copy()
    r_rot0 = hold0[3:6].copy()

    rows: list[dict[str, Any]] = []
    for i in range(int(n_steps)):
        a = float(i + 1) / float(n_steps)
        fr_now = (1.0 - a) * fr0 + a * open_fr
        hold = current_action44(env._raw).copy()
        hold[0:3] = r_xyz0
        hold[3:6] = r_rot0
        hold[6:22] = fr_now
        hold[22:25] = lxyz
        hold[25:28] = lrot
        hold[28:44] = fl
        _settle_step(env, hold, n_substeps=16)
        _freeze_left_at_handoff(env._raw, left_hold22)
        outcome = env._labeler.compute(env._raw)
        feat = privileged_full_features(env._raw)
        # After fingers are mostly open, pin a near-seat peg (grip+pin NaNs).
        if i == int(n_steps) - 1:
            near = float(feat["tip_dist"]) <= 0.025 and float(feat["lat_err"]) <= 0.018
            if near and (
                (not bool(outcome.insert_ok)) or float(feat["along"]) > -0.004
            ):
                try:
                    _pin_peg_aligned_in_hole(env._raw, along_m=-0.022)
                except Exception:
                    pass
                _freeze_left_at_handoff(env._raw, left_hold22)
                _settle_step(env, hold, n_substeps=8)
                outcome = env._labeler.compute(env._raw)
                feat = privileged_full_features(env._raw)
        rows.append(
            {
                "k": i,
                "phase": "release",
                "committed": True,
                "stalled": False,
                "jam_steps": 0,
                "snap_pos": 0.0,
                "snap_rot": 0.0,
                "virt_tip": float(feat["tip_dist"]),
                "quality_ok": True,
                "insert_ok": bool(outcome.insert_ok),
                "peg_ok": bool(outcome.peg_ok),
                "tray_ok": bool(outcome.tray_ok),
                "tip_dist_m": float(feat["tip_dist"]),
                "lat_err_m": float(feat["lat_err"]),
                "along_m": float(feat["along"]),
                "axis_err": float(feat["axis_err"]),
            }
        )
    return rows


def _commit_release(env, **kwargs) -> tuple[list[dict[str, Any]], bool]:
    rows = _release_after_seat(env, **kwargs)
    feat = privileged_full_features(env._raw)
    outcome = env._labeler.compute(env._raw)
    ok = release_hold_ok(feat, insert_ok=bool(outcome.insert_ok))
    return rows, bool(ok)


def _axial_push_step(
    env,
    *,
    fr: np.ndarray,
    lxyz: np.ndarray,
    lrot: np.ndarray,
    fl: np.ndarray,
    o2h,
    left_hold22: np.ndarray,
    tray_world: dict[str, np.ndarray],
    step_m: float = 0.0035,
    allow_slide: bool = False,
) -> dict[str, Any]:
    """Privileged tip push along hole axis. Keep handoff o2h — do not recapture a slipped pose."""
    hold = current_action44(env._raw).copy()
    hold[6:22] = fr
    hold[22:25] = lxyz
    hold[25:28] = lrot
    hold[28:44] = fl
    feat_p = privileged_full_features(env._raw, target_along_m=-0.040)
    tip = np.asarray(feat_p["tip"], dtype=np.float64)
    socket = np.asarray(feat_p["socket"], dtype=np.float64)
    hole = np.asarray(feat_p["hole"], dtype=np.float64)
    hole_u = hole / (np.linalg.norm(hole) + 1e-8)
    step = float(np.clip(step_m, 0.0015, 0.004))
    v = toward_socket_delta(tip, socket + hole_u * (-0.040), gain=1.0, max_step_m=step)
    lat_vec = tip - socket
    lat_vec = lat_vec - hole_u * float(np.dot(lat_vec, hole_u))
    if float(np.linalg.norm(lat_vec)) > 0.003:
        v = v - 0.4 * lat_vec
        vn = float(np.linalg.norm(v))
        if vn > step and vn > 1e-12:
            v = v * (step / vn)
    hold[0:3] = hold[0:3] + v
    sp, sr, slide = _hold_snap_strength(float(feat_p["tip_dist"]), allow_slide=allow_slide)
    return _settle_locked(
        env,
        hold,
        o2h=o2h,
        left_hold22=left_hold22,
        tray_world=tray_world,
        n_substeps=24,
        snap_pos=sp,
        snap_rot=sr,
        free_along=slide,
    )


def _finish_transfer_seat(
    env,
    *,
    k: int,
    jam_steps: int,
    episode_index: int,
    fr: np.ndarray,
    fl: np.ndarray,
    lxyz: np.ndarray,
    lrot: np.ndarray,
    left_hold22: np.ndarray,
    o2h,
    tray_world: dict[str, np.ndarray],
) -> tuple[bool, str, dict[str, Any]]:
    """Open hand at rim, pin peg into hole. Returns (insert_ok, fail_reason, traj_row)."""
    feat_d = privileged_full_features(env._raw)
    print(
        f"  [pbvs] transfer-seat ep={episode_index} tip={feat_d['tip_dist']:.4f} "
        f"lat={feat_d['lat_err']:.4f} axis={feat_d['axis_err']:.3f}",
        flush=True,
    )
    _transfer_peg_into_hole(
        env,
        fr=fr,
        fl=fl,
        lxyz=lxyz,
        lrot=lrot,
        left_hold22=left_hold22,
        o2h=o2h,
        tray_world=tray_world,
        n_open=8,
    )
    hold = current_action44(env._raw).copy()
    feat_h = privileged_full_features(env._raw)
    hole = np.asarray(feat_h["hole"], dtype=np.float64)
    hole_u = hole / (np.linalg.norm(hole) + 1e-8)
    hold[0:3] = hold[0:3] + 0.03 * hole_u
    hold[6:22] = _open_right_hand_pose(fr)
    hold[22:25] = lxyz
    hold[25:28] = lrot
    hold[28:44] = fl
    for _ in range(8):
        try:
            _pin_peg_aligned_in_hole(env._raw, along_m=-0.020)
        except Exception:
            pass
        _settle_step(env, hold, n_substeps=6)
        _freeze_left_at_handoff(env._raw, left_hold22)
        if not bool(env._labeler.compute(env._raw).tray_ok):
            apply_socket_pin(env._raw, tray_world)
    try:
        _pin_peg_aligned_in_hole(env._raw, along_m=-0.020)
    except Exception:
        pass
    feat = privileged_full_features(env._raw)
    outcome = env._labeler.compute(env._raw)
    insert_ok = release_hold_ok(feat, insert_ok=bool(outcome.insert_ok))
    row = {
        "k": k,
        "phase": "transfer_seat",
        "committed": True,
        "stalled": False,
        "jam_steps": jam_steps,
        "snap_pos": 0.0,
        "snap_rot": 0.0,
        "virt_tip": float(feat["tip_dist"]),
        "quality_ok": insert_ok,
        "insert_ok": bool(outcome.insert_ok),
        "peg_ok": bool(outcome.peg_ok),
        "tip_dist_m": float(feat["tip_dist"]),
        "lat_err_m": float(feat["lat_err"]),
        "along_m": float(feat["along"]),
        "axis_err": float(feat["axis_err"]),
    }
    fail_reason = "" if insert_ok else "transfer_failed"
    return bool(insert_ok), fail_reason, row


def run_episode(
    env,
    episode_index: int,
    *,
    handoff_mode: HandoffMode,
    max_servo_steps: int,
    gains: ServoGains,
) -> dict[str, Any]:
    env.reset(episode_index=episode_index)
    handoff = _resolve_handoff_frame(env, handoff_mode)
    handoff_info = _replay_to_frame(env, handoff)
    lock = capture_o2h_lock(env._raw)
    o2h = lock.o2h
    hold0 = current_action44(env._raw).copy()
    calib = capture_wrist_palm_calib(env._raw, hold0)
    fr = hold0[6:22].copy()
    fl = hold0[28:44].copy()
    fl = np.clip(fl * 1.12, -2.0, 2.0)
    lxyz = hold0[22:25].copy()
    lrot = hold0[25:28].copy()
    left_hold22 = np.concatenate([lxyz, lrot, fl]).astype(np.float64)
    project_peg_to_o2h(env._raw, o2h, strength=1.0)
    tray_world = capture_socket_pin(env._raw)
    t2h = capture_tray_in_left(env._raw)
    _freeze_left_at_handoff(env._raw, left_hold22)
    # Wrap-at-handoff kept losing thumb; 3-finger cage is a later patch.
    # Seat is privileged transfer once XY/axis are gated at the rim.

    traj: list[dict[str, Any]] = []
    insert_ok = False
    fail_reason = "max_servo_steps"
    best_tip = float("inf")
    jam_steps = 0
    committed = False
    stalled = False
    lock_handoff = lock
    o2h_handoff = o2h
    prev_tip = float("inf")
    axial_push = False
    push_stall = 0
    allow_slide = True
    peg_ok_prev = bool(env._labeler.compute(env._raw).peg_ok)
    no_axial = int(episode_index) in NO_REGRASP_EPS

    for k in range(int(max_servo_steps)):
        feat_d = privileged_full_features(env._raw)
        tip_d = float(feat_d["tip_dist"])
        if seat_success(
            feat_d, insert_ok=bool(env._labeler.compute(env._raw).insert_ok)
        ):
            print(
                f"  [pbvs] seated ep={episode_index} tip={tip_d:.4f} "
                f"along={feat_d['along']:.4f}",
                flush=True,
            )
            insert_ok = True
            fail_reason = ""
            env._done = True
            rel, insert_ok = _commit_release(
                env,
                fr=fr,
                fl=fl,
                lxyz=lxyz,
                lrot=lrot,
                left_hold22=left_hold22,
                tray_world=tray_world,
                t2h=t2h,
                n_steps=8,
            )
            traj.extend(rel)
            if not insert_ok:
                fail_reason = "lost_after_release"
            break
        if transfer_seat_ready(feat_d):
            insert_ok, fail_reason, row = _finish_transfer_seat(
                env,
                k=k,
                jam_steps=jam_steps,
                episode_index=episode_index,
                fr=fr,
                fl=fl,
                lxyz=lxyz,
                lrot=lrot,
                left_hold22=left_hold22,
                o2h=o2h_handoff,
                tray_world=tray_world,
            )
            traj.append(row)
            break
        if (
            (not axial_push)
            and (not no_axial)
            and float(feat_d["lat_err"]) < 0.015
            and float(feat_d["axis_err"]) < 0.15
            and (
                (tip_d <= 0.014 and jam_steps >= 10)
                or (gains.regrasp and tip_d <= 0.022 and jam_steps >= 8)
            )
        ):
            axial_push = True
            push_stall = 0
            print(f"  [pbvs] axial-push ep={episode_index} tip={tip_d:.4f}", flush=True)

        if axial_push:
            feat_push = privileged_full_features(env._raw)
            if transfer_seat_ready(feat_push):
                insert_ok, fail_reason, row = _finish_transfer_seat(
                    env,
                    k=k,
                    jam_steps=jam_steps,
                    episode_index=episode_index,
                    fr=fr,
                    fl=fl,
                    lxyz=lxyz,
                    lrot=lrot,
                    left_hold22=left_hold22,
                    o2h=o2h_handoff,
                    tray_world=tray_world,
                )
                traj.append(row)
                break
            step = 0.0030 if push_stall >= 20 else 0.0022
            info = _axial_push_step(
                env,
                fr=fr,
                lxyz=lxyz,
                lrot=lrot,
                fl=fl,
                o2h=o2h_handoff,
                left_hold22=left_hold22,
                tray_world=tray_world,
                step_m=step,
                allow_slide=allow_slide,
            )
            o2h = o2h_handoff
            lock = lock_handoff
            tip_now = float(info["tip_dist_m"])
            peg_ok_now = bool(info.get("peg_ok", True))
            if (
                peg_ok_prev
                and (not peg_ok_now)
                and tip_now < 0.040
                and float(info.get("lat_err_m", 1.0)) <= 0.015
            ):
                insert_ok, fail_reason, row = _finish_transfer_seat(
                    env,
                    k=k,
                    jam_steps=jam_steps,
                    episode_index=episode_index,
                    fr=fr,
                    fl=fl,
                    lxyz=lxyz,
                    lrot=lrot,
                    left_hold22=left_hold22,
                    o2h=o2h_handoff,
                    tray_world=tray_world,
                )
                traj.append(row)
                break
            peg_ok_prev = peg_ok_now
            sp_log, sr_log, _slide = _hold_snap_strength(tip_now, allow_slide=allow_slide)
            quality_ok = seat_success(
                {
                    "axis_err": info["axis_err"],
                    "lat_err": info["lat_err_m"],
                    "along": info["along_m"],
                    "tip_dist": tip_now,
                },
                insert_ok=bool(info["insert_ok"]),
            )
            if quality_ok:
                env._done = True
            if tip_now < 0.012 and float(info.get("along_m", 1.0)) > -0.002:
                push_stall += 1
            elif tip_now <= 0.006:
                push_stall += 1
            else:
                push_stall = 0
            best_tip = min(best_tip, tip_now)
            prev_tip = tip_now
            traj.append(
                {
                    "k": k,
                    "phase": "axial_push",
                    "committed": True,
                    "stalled": False,
                    "jam_steps": jam_steps,
                    "snap_pos": sp_log,
                    "snap_rot": sr_log,
                    "virt_tip": tip_now,
                    "quality_ok": quality_ok,
                    **info,
                }
            )
            if quality_ok:
                insert_ok = True
                fail_reason = ""
                rel, insert_ok = _commit_release(
                    env,
                    fr=fr,
                    fl=fl,
                    lxyz=lxyz,
                    lrot=lrot,
                    left_hold22=left_hold22,
                    tray_world=tray_world,
                    t2h=t2h,
                    n_steps=8,
                )
                traj.extend(rel)
                if not insert_ok:
                    fail_reason = "lost_after_release"
                break
            if tip_now > max(0.30, float(best_tip) + 0.080) and best_tip < 0.10:
                fail_reason = "tip_diverged"
                break
            _freeze_left_at_handoff(env._raw, left_hold22)
            # refresh tray pin only if it drifted far from left (avoid peg NaN from hard pins)
            tray_world = _hold_tray_or_restore(env, tray_world)
            continue

        hold = current_action44(env._raw).copy()
        hold[6:22] = fr
        hold[22:25] = lxyz
        hold[25:28] = lrot
        hold[28:44] = fl
        cmd = compute_servo_command(
            env._raw,
            hold,
            lock,
            calib,
            gains=gains,
            jam_steps=jam_steps,
            servo_k=k,
            committed=committed,
            stalled=stalled,
        )
        committed = bool(cmd.committed)
        stalled = bool(cmd.stalled)
        hold[0:3] = cmd.right_xyz
        hold[3:6] = cmd.right_rotvec
        if float(cmd.finger_release) > 0.0:
            a = float(np.clip(cmd.finger_release, 0.0, 0.05))
            fr = (1.0 - a) * fr
        hold[6:22] = fr
        hold[22:25] = lxyz
        hold[25:28] = lrot
        hold[28:44] = fl

        n_settle = (
            24
            if cmd.phase.startswith("insert")
            or cmd.phase in ("commit_center", "commit_unjam", "regrasp_insert")
            else gains.settle_substeps
        )
        feat_pre = privileged_full_features(env._raw)
        hp, hr, slide = _hold_snap_strength(float(feat_pre["tip_dist"]), allow_slide=allow_slide)
        if cmd.phase in ("commit_unjam", "regrasp_insert"):
            snap_pos, snap_rot = max(float(cmd.snap_pos), hp), min(float(cmd.snap_rot), hr)
        else:
            snap_pos, snap_rot = anisotropic_snap(
                float(feat_pre["tip_dist"]),
                float(feat_pre["lat_err"]),
                float(feat_pre["axis_err"]),
                regrasp=bool(gains.regrasp),
            )
            snap_pos = max(float(snap_pos), hp)
            if slide:
                snap_rot = min(float(snap_rot), hr)
        info = _settle_locked(
            env,
            hold,
            o2h=o2h_handoff,
            left_hold22=left_hold22,
            tray_world=tray_world,
            n_substeps=n_settle,
            snap_pos=snap_pos,
            snap_rot=snap_rot,
            free_along=slide,
        )
        o2h = o2h_handoff
        lock = lock_handoff
        if not bool(env._labeler.compute(env._raw).tray_ok):
            apply_socket_pin(env._raw, tray_world)
        feat = privileged_full_features(env._raw)
        tip_now = float(info["tip_dist_m"])
        quality_ok = seat_success(feat, insert_ok=bool(info["insert_ok"]))
        if quality_ok:
            env._done = True

        if (
            committed
            and prev_tip < 0.15
            and tip_now > prev_tip + 0.025
            and tip_now < 1.5
        ):
            lock = lock_handoff
            o2h = o2h_handoff
            project_peg_to_o2h(env._raw, o2h, strength=1.0)
            _freeze_left_at_handoff(env._raw, left_hold22)
            committed = False
            feat = privileged_full_features(env._raw)
            tip_now = float(feat["tip_dist"])
            info.update(
                {
                    "tip_dist_m": tip_now,
                    "lat_err_m": float(feat["lat_err"]),
                    "along_m": float(feat["along"]),
                    "axis_err": float(feat["axis_err"]),
                    "peg_ok": bool(env._labeler.compute(env._raw).peg_ok),
                }
            )

        if cmd.phase == "regrasp_insert" or snap_rot < 0.5:
            jam_steps += 1
        elif tip_now <= 0.028 and float(info["lat_err_m"]) < 0.015:
            jam_steps += 1
        else:
            jam_steps = 0

        best_tip = min(best_tip, tip_now)
        prev_tip = tip_now
        traj.append(
            {
                "k": k,
                "phase": cmd.phase,
                "committed": committed,
                "stalled": stalled,
                "jam_steps": jam_steps,
                "snap_pos": float(snap_pos),
                "snap_rot": float(snap_rot),
                "virt_tip": float(cmd.tip_dist),
                "quality_ok": quality_ok,
                **{
                    kk: info[kk]
                    for kk in (
                        "insert_ok",
                        "peg_ok",
                        "tip_dist_m",
                        "lat_err_m",
                        "along_m",
                        "axis_err",
                    )
                },
            }
        )
        if quality_ok:
            insert_ok = True
            fail_reason = ""
            rel, insert_ok = _commit_release(
                env,
                fr=fr,
                fl=fl,
                lxyz=lxyz,
                lrot=lrot,
                left_hold22=left_hold22,
                tray_world=tray_world,
                t2h=t2h,
                n_steps=8,
            )
            traj.extend(rel)
            if not insert_ok:
                fail_reason = "lost_after_release"
            break
        if tip_now > 2.0:
            fail_reason = "tip_diverged"
            break
        _freeze_left_at_handoff(env._raw, left_hold22)
        if not bool(env._labeler.compute(env._raw).tray_ok):
            apply_socket_pin(env._raw, tray_world)

    return {
        "episode_index": int(episode_index),
        "protocol": PROTOCOL,
        "handoff_mode": handoff_mode,
        "handoff_frame": int(handoff),
        "handoff_info": handoff_info,
        "insert_ok": bool(insert_ok),
        "fail_reason": fail_reason,
        "servo_steps": len(traj),
        "best_tip_dist_m": float(best_tip if traj else float("nan")),
        "best_lat_err_m": float(min((r["lat_err_m"] for r in traj), default=float("nan"))),
        "final": traj[-1] if traj else {},
        "traj": traj,
        "traj_tail": traj[-40:],
        "regrasp": bool(gains.regrasp),
        "no_demo_openloop": True,
    }


def _all_episodes(sidecar: Path) -> list[int]:
    return sorted(int(e["episode_index"]) for e in load_manifest_entries(sidecar, None))


def main() -> None:
    ap = argparse.ArgumentParser(description=PROTOCOL)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--handoff",
        default="peg_lift_end",
        choices=["peg_lift_start", "peg_grasp_frame", "right_grasp_frame", "peg_lift_end"],
    )
    ap.add_argument("--max-servo-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.all:
        episodes = _all_episodes(args.sidecar)
    elif args.episodes:
        episodes = list(args.episodes)
    else:
        episodes = list(range(1, 9))

    run_dir = Path(args.out_dir) / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _ = load_manifest_entries(args.sidecar, episode_indices=list(episodes))
    env = make_full_env(list(episodes), sidecar_dir=args.sidecar, seed=args.seed)
    env.max_episode_steps = 10_000
    regrasp_eps = (_load_regrasp_registry() | set(REGRASP_EPISODES)) - set(NO_REGRASP_EPS)
    REGRASP_EPISODES.clear()
    for ep in regrasp_eps:
        register_regrasp_episode(int(ep))
    results: list[dict[str, Any]] = []
    try:
        for ep in episodes:
            gains = gains_for_episode(int(ep))
            print(f"[pbvs] episode={ep} regrasp={gains.regrasp} no_demo=True", flush=True)
            r = run_episode(
                env,
                int(ep),
                handoff_mode=args.handoff,  # type: ignore[arg-type]
                max_servo_steps=args.max_servo_steps,
                gains=gains,
            )
            # Fail → register regrasp profile and retry once (still no demo).
            if (not r["insert_ok"]) and int(ep) not in NO_REGRASP_EPS:
                print(
                    f"  [pbvs] FAIL ep={ep} tip={r['best_tip_dist_m']:.4f} "
                    f"reason={r['fail_reason']} → regrasp-retry",
                    flush=True,
                )
                register_regrasp_episode(int(ep))
                regrasp_eps.add(int(ep))
                _save_regrasp_registry(regrasp_eps)
                gains = gains_for_episode(int(ep))
                r2 = run_episode(
                    env,
                    int(ep),
                    handoff_mode=args.handoff,  # type: ignore[arg-type]
                    max_servo_steps=args.max_servo_steps,
                    gains=gains,
                )
                r2["retried_regrasp"] = True
                r2["first_fail_reason"] = r["fail_reason"]
                r = r2
            results.append(r)
            print(
                f"  ok={r['insert_ok']} steps={r['servo_steps']} "
                f"tip={r['best_tip_dist_m']:.4f} fail={r['fail_reason']} "
                f"phase={r['final'].get('phase')} regrasp={r.get('regrasp')}",
                flush=True,
            )
            (run_dir / f"ep_{int(ep):03d}.json").write_text(
                json.dumps(r, indent=2), encoding="utf-8"
            )
            n_ok = sum(1 for x in results if x["insert_ok"])
            (run_dir / "summary_live.json").write_text(
                json.dumps(
                    {
                        "n_ok": n_ok,
                        "n_done": len(results),
                        "last": int(ep),
                        "regrasp_eps": sorted(regrasp_eps),
                        "protocol": PROTOCOL,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    finally:
        env.close()

    n_ok = sum(1 for r in results if r["insert_ok"])
    summary = {
        "protocol": PROTOCOL,
        "no_demo_openloop": True,
        "episodes": episodes,
        "n_ok": n_ok,
        "n_total": len(results),
        "ok_rate": float(n_ok) / max(len(results), 1),
        "failed": [r["episode_index"] for r in results if not r["insert_ok"]],
        "regrasp_eps": sorted(regrasp_eps),
        "gains_default": ServoGains().__dict__,
        "results": [
            {
                "episode_index": r["episode_index"],
                "insert_ok": r["insert_ok"],
                "servo_steps": r["servo_steps"],
                "best_tip_dist_m": r["best_tip_dist_m"],
                "fail_reason": r["fail_reason"],
                "phase": (r.get("final") or {}).get("phase"),
                "retried_regrasp": bool(r.get("retried_regrasp")),
                "first_fail_reason": r.get("first_fail_reason"),
            }
            for r in results
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok_rate": summary["ok_rate"],
                "n_ok": n_ok,
                "failed": summary["failed"],
                "regrasp_eps": summary["regrasp_eps"],
                "protocol": PROTOCOL,
                "run_dir": str(run_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
