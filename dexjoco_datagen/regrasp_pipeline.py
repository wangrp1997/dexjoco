"""Failure → MimicGen-style SE(3) regrasp (classic merge + demo segment).

Focus: reliable regrasp data. No hand-rolled collision planners in the hot path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state
from interaction_retarget.constants import PEG_BODY, TRAY_BODY
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.replay import make_assembly_env, raw_flat_to_dict
from interaction_retarget.sim.settle import read_arm_action, vec_to_arm_action
from interaction_retarget.sim.video import DexEnvVideoRecorder

from dexjoco_datagen.pose_se3 import quat_wxyz_to_mat, transform_action23_segment
from dexjoco_datagen.traj_smooth import min_jerk

Mode = Literal["drop", "grasp_fail"]
_MODE_ALIASES = {
    "drop": "drop",
    "grasp_fail": "grasp_fail",
    "drop_mid_lift": "drop",
    "lift_no_follow": "grasp_fail",
}


def normalize_mode(mode: str) -> Mode:
    key = str(mode).strip().lower()
    if key not in _MODE_ALIASES:
        raise ValueError(f"unknown mode {mode!r}; use drop or grasp_fail")
    return _MODE_ALIASES[key]  # type: ignore[return-value]


@dataclass
class RegraspResult:
    episode_index: int
    success: bool
    peg_z_after_fail: float
    peg_z_final: float
    mode: str
    video_path: str
    message: str
    diagnostics: dict[str, Any]


def _ori_delta_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    r0 = quat_wxyz_to_mat(q0)
    r1 = quat_wxyz_to_mat(q1)
    return float(np.degrees(R.from_matrix(r0.T @ r1).magnitude()))


def _axis_delta_deg(q0: np.ndarray, q1: np.ndarray, *, axis: int = 2) -> float:
    a0 = quat_wxyz_to_mat(q0)[:, int(axis)]
    a1 = quat_wxyz_to_mat(q1)[:, int(axis)]
    c = float(np.clip(np.dot(a0, a1), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _pose_for_se3(
    src_pos: np.ndarray,
    src_quat: np.ndarray,
    cur_pos: np.ndarray,
    cur_quat: np.ndarray,
    *,
    axisymmetric: bool = True,
    axis_tol_deg: float = 30.0,
    prefer_src_quat: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Round peg: prefer demo/on-table quat so a tumble doesn't warp EE into the table."""
    full = _ori_delta_deg(src_quat, cur_quat)
    axis = _axis_delta_deg(src_quat, cur_quat, axis=2)
    info = {"ori_delta_deg": full, "axis_delta_deg": axis}
    del src_pos
    if axisymmetric and (prefer_src_quat or axis <= axis_tol_deg):
        return np.asarray(cur_pos, dtype=np.float64), np.asarray(src_quat, dtype=np.float64), info
    return np.asarray(cur_pos, dtype=np.float64), np.asarray(cur_quat, dtype=np.float64), info


def _classify_fail(
    *,
    success: bool,
    peg_z_fail: float,
    peg_z_final: float,
    peg_z_ref: float,
    ori_delta_deg: float,
    axis_delta_deg: float,
    peg_tray_xy: float,
    wrist_xy_reach: float,
    success_lift_m: float,
    regrasp_attempted: bool = True,
) -> str:
    if success:
        return "ok"
    dz = float(peg_z_final - peg_z_fail)
    # If we already ran regrasp, prefer contact/lift labels over pre-fail tumble.
    if not regrasp_attempted and axis_delta_deg >= 35.0:
        return "ori_tumble_se3_mismatch"
    if peg_tray_xy < 0.08:
        return "peg_too_close_to_tray"
    if wrist_xy_reach > 0.55:
        return "wrist_overreach_near_singularity"
    if dz < 0.01 and peg_z_final < peg_z_ref + 0.01:
        if axis_delta_deg >= 35.0:
            return "regrasp_fail_after_tumble"
        return "grasp_no_lift_contact_fail"
    if 0.01 <= dz < success_lift_m:
        return "weak_lift_slip"
    return "unknown_fail"


def _body_pose(raw_env, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    bid = raw_env._model.body(body_name).id
    data = raw_env._data
    return data.xpos[bid].copy(), data.xquat[bid].copy()


def _step_dict(raw_env, right23: np.ndarray, left23: np.ndarray, video: DexEnvVideoRecorder | None) -> None:
    raw_env.step(
        {
            "right": np.asarray(right23, dtype=np.float64).reshape(23),
            "left": np.asarray(left23, dtype=np.float64).reshape(23),
        }
    )
    if video is not None:
        video.capture()


def _open_hand(action23: np.ndarray, *, scale: float = 0.0) -> np.ndarray:
    out = np.asarray(action23, dtype=np.float64).reshape(23).copy()
    out[7:23] = float(scale)
    return out


def _with_underclose(action23: np.ndarray, *, scale: float = 0.28) -> np.ndarray:
    out = vec_to_arm_action(action23)
    out[7:23] = float(scale) * out[7:23]
    return out


def _slerp_quat_wxyz_shortest(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    q0_xyzw = q0[[1, 2, 3, 0]]
    q1_xyzw = q1[[1, 2, 3, 0]]
    slerp = Slerp([0.0, 1.0], R.from_quat([q0_xyzw, q1_xyzw]))
    out_xyzw = slerp(float(np.clip(t, 0.0, 1.0))).as_quat()
    return np.asarray([out_xyzw[3], out_xyzw[0], out_xyzw[1], out_xyzw[2]], dtype=np.float64)


def _interp_action23(a23: np.ndarray, b23: np.ndarray, t: float) -> np.ndarray:
    a23 = vec_to_arm_action(a23)
    b23 = vec_to_arm_action(b23)
    t = float(np.clip(t, 0.0, 1.0))
    pos = (1.0 - t) * a23[0:3] + t * b23[0:3]
    quat = _slerp_quat_wxyz_shortest(a23[3:7], b23[3:7], t)
    hand = (1.0 - t) * a23[7:23] + t * b23[7:23]
    return np.concatenate([pos, quat, hand], axis=0)


def _unit_xy(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        fb = np.asarray(fallback, dtype=np.float64).reshape(2)
        fn = float(np.linalg.norm(fb))
        return fb / fn if fn > 1e-6 else np.array([0.0, -1.0], dtype=np.float64)
    return v / n


def _midair_slip_release(
    raw_env,
    *,
    left_hold: np.ndarray,
    video: DexEnvVideoRecorder | None,
    open_scale: float = 0.05,
    steps_per_finger: int = 36,
    settle_steps: int = 48,
) -> list[int]:
    """Proven mid-air slip: ff→mf→rf→th (thumb last). Wrist fixed, left hold."""
    order = [1, 2, 3, 0]
    right0 = read_arm_action(raw_env, "right")
    hand = right0[7:23].copy()
    for finger in order:
        s = finger * 4
        start = hand[s : s + 4].copy()
        for i in range(max(int(steps_per_finger), 1)):
            t = (i + 1) / float(steps_per_finger)
            t = t * t * (3.0 - 2.0 * t)
            hand = hand.copy()
            hand[s : s + 4] = (1.0 - t) * start + t * open_scale
            cmd = right0.copy()
            cmd[7:23] = hand
            _step_dict(raw_env, cmd, left_hold, video)
    hold = right0.copy()
    hold[7:23] = open_scale
    for _ in range(max(int(settle_steps), 1)):
        _step_dict(raw_env, hold, left_hold, video)
    return order


def _execute_mimicgen_regrasp(
    raw_env,
    *,
    transformed_right: np.ndarray,
    left_hold: np.ndarray,
    video: DexEnvVideoRecorder | None,
    z_min: float,
    merge_step_m: float = 0.01,
    squeeze_hold: int = 44,
    lift_m: float = 0.12,
    lift_steps: int = 40,
) -> dict[str, Any]:
    """Classic MimicGen: merge (keep demo hand) → replay SE(3) → hold → lift.

    Critical: do NOT force-open during merge. Demo approach already has fingers
    closing; an open-hand merge leaves fingers too late and misses the grasp.
    Left stays on the post-fail tray hold (stable).
    """
    right_seg = np.asarray(transformed_right, dtype=np.float64).reshape(-1, 23)
    z_floor = float(z_min)
    left_hold = vec_to_arm_action(left_hold).copy()

    # Keep demo hand on first frame (not open).
    first_right = vec_to_arm_action(right_seg[0]).copy()
    if float(first_right[2]) < z_floor:
        first_right[2] = z_floor

    start_r = read_arm_action(raw_env, "right")
    dist = float(np.linalg.norm(first_right[0:3] - start_r[0:3]))
    # Fixed 80 steps: fingers need time to catch up from post-fail open hand.
    n_m = 80
    del dist, merge_step_m
    for i in range(n_m):
        t = (i + 1) / float(n_m)
        cmd_r = _interp_action23(start_r, first_right, t)
        if float(cmd_r[2]) < z_floor:
            cmd_r[2] = z_floor
        _step_dict(raw_env, cmd_r, left_hold, video)

    for a_r in right_seg:
        a_r = vec_to_arm_action(a_r)
        # Do not raise wrist with z_floor during demo replay — it breaks grasp geometry.
        _step_dict(raw_env, a_r, left_hold, video)

    hold = vec_to_arm_action(right_seg[-1]).copy()
    for _ in range(max(int(squeeze_hold), 1)):
        _step_dict(raw_env, hold, left_hold, video)

    lift = hold.copy()
    lift[2] = float(hold[2] + lift_m)
    for i in range(max(int(lift_steps), 1)):
        t = (i + 1) / float(lift_steps)
        _step_dict(raw_env, _interp_action23(hold, lift, t), left_hold, video)

    return {
        "recipe": "classic_mimicgen_closed_merge",
        "merge_steps": int(n_m),
        "z_floor": z_floor,
    }


def run_perturb_regrasp_one(
    entry: dict[str, Any],
    *,
    video_path: Path,
    seed: int = 0,
    mode: Mode | str = "drop",
    pre_grasp_frames: int = 35,
    mid_lift_frames: int = 6,
    settle_after_drop: int = 70,
    no_follow_lift_frames: int = 16,
    post_lift_frames: int = 45,
    success_lift_m: float = 0.05,
    record_video: bool = True,
    underclose_scale: float = 0.28,
    drop_lift_m: float | None = None,
    drop_max_lift_frames: int | None = None,
    open_hold_steps: int | None = None,
    retreat_steps: int | None = None,
    max_ori_delta_deg: float | None = None,
    _allow_pg_fallback: bool = True,
) -> RegraspResult:
    del drop_lift_m, drop_max_lift_frames, open_hold_steps, retreat_steps, max_ori_delta_deg
    mode = normalize_mode(mode)

    # grasp_fail: some eps need short approach (35), some need longer (50).
    if mode == "grasp_fail" and _allow_pg_fallback:
        winners: list[int] = []
        last: RegraspResult | None = None
        for pg in (35, 50):
            last = run_perturb_regrasp_one(
                entry,
                video_path=video_path,
                seed=seed,
                mode=mode,
                pre_grasp_frames=pg,
                mid_lift_frames=mid_lift_frames,
                settle_after_drop=settle_after_drop,
                no_follow_lift_frames=no_follow_lift_frames,
                post_lift_frames=post_lift_frames,
                success_lift_m=success_lift_m,
                record_video=False,
                underclose_scale=underclose_scale,
                _allow_pg_fallback=False,
            )
            if last.success:
                winners.append(pg)
                break
        pg_use = winners[0] if winners else 35
        if record_video:
            out = run_perturb_regrasp_one(
                entry,
                video_path=video_path,
                seed=seed,
                mode=mode,
                pre_grasp_frames=pg_use,
                mid_lift_frames=mid_lift_frames,
                settle_after_drop=settle_after_drop,
                no_follow_lift_frames=no_follow_lift_frames,
                post_lift_frames=post_lift_frames,
                success_lift_m=success_lift_m,
                record_video=True,
                underclose_scale=underclose_scale,
                _allow_pg_fallback=False,
            )
            if out.diagnostics is not None:
                out.diagnostics["pre_grasp_frames"] = int(pg_use)
                out.diagnostics["pg_fallback_probed"] = True
            return out
        assert last is not None
        if last.diagnostics is not None:
            last.diagnostics["pre_grasp_frames"] = int(pg_use)
            last.diagnostics["pg_fallback_probed"] = True
        return last

    timing = entry["timing"]
    ep = int(entry["episode_index"])
    right_grasp = int(timing["right_grasp_frame"])
    peg_lift = int(timing["peg_lift_start"])
    if right_grasp < 0 or peg_lift <= right_grasp:
        raise ValueError(f"ep{ep}: bad peg timing grasp={right_grasp} lift={peg_lift}")

    actions, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    n = len(actions)
    seg_start = int(np.clip(right_grasp - pre_grasp_frames, 0, n - 1))
    seg_end = int(np.clip(peg_lift + post_lift_frames, seg_start + 1, n))

    env = make_assembly_env(seed=seed, randomize=False, render_mode="rgb_array")
    raw = env.unwrapped
    config = CONFIG_MAPPING["bimanual_assembly"]()
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video = DexEnvVideoRecorder(env, video_path, fps=30) if record_video else None

    try:
        env.reset()
        if initial_state is not None and has_restorer("bimanual_assembly"):
            restore_initial_state(env, "bimanual_assembly", config, initial_state)

        src_seg_right: list[np.ndarray] = []
        src_obj_pos = None
        src_obj_quat = None
        left_hold = None
        peg_z0, _ = _body_pose(raw, PEG_BODY)
        peg_z_ref = float(peg_z0[2])
        slip_fingers: list[int] = []
        planner_info: dict[str, Any] = {}

        if mode == "drop":
            # Proven timing: drop at peg_lift + mid_lift_frames (default 6).
            drop_at = int(np.clip(peg_lift + mid_lift_frames, peg_lift + 1, n - 1))
            src_ref_t = int(np.clip(right_grasp - 8, 0, drop_at))
            for t in range(drop_at + 1):
                act = raw_flat_to_dict(actions[t])
                right = np.asarray(act["right"], dtype=np.float64)
                left = np.asarray(act["left"], dtype=np.float64)
                if t == src_ref_t:
                    src_obj_pos, src_obj_quat = _body_pose(raw, PEG_BODY)
                _step_dict(raw, right, left, video)
                left_hold = left.copy()

            assert left_hold is not None
            if src_obj_pos is None:
                src_obj_pos, src_obj_quat = _body_pose(raw, PEG_BODY)

            src_seg_right = [
                np.asarray(raw_flat_to_dict(actions[t])["right"], dtype=np.float64).copy()
                for t in range(seg_start, seg_end)
            ]
            slip_fingers = _midair_slip_release(
                raw,
                left_hold=left_hold,
                video=video,
                steps_per_finger=36,
                settle_steps=48,
            )

            hold = read_arm_action(raw, "right")
            tray_now, _ = _body_pose(raw, TRAY_BODY)
            away_xy = _unit_xy(hold[0:2] - tray_now[0:2], fallback=np.array([0.0, -1.0]))
            away = hold.copy()
            away[0:2] = away[0:2] + away_xy * 0.03
            away[2] = float(away[2] + 0.02)
            start = hold.copy()
            for i in range(18):
                t = min_jerk((i + 1) / 18.0)
                _step_dict(raw, _interp_action23(start, away, t), left_hold, video)
            hold = read_arm_action(raw, "right")
            for _ in range(settle_after_drop):
                _step_dict(raw, hold, left_hold, video)
            planner_info = {"drop_at": drop_at, "slip_order": slip_fingers}

            peg_pos_cur, peg_quat_cur = _body_pose(raw, PEG_BODY)
            peg_z_fail = float(peg_pos_cur[2])

        elif mode == "grasp_fail":
            # Underclose only around grasp; open-hand lift so peg stays on table
            # without dragging (full underclose-through-lift breaks some episodes).
            pre_weak = int(np.clip(right_grasp - 12, 0, n - 1))
            for t in range(pre_weak):
                act = raw_flat_to_dict(actions[t])
                right = np.asarray(act["right"], dtype=np.float64)
                left = np.asarray(act["left"], dtype=np.float64)
                _step_dict(raw, right, left, video)
                left_hold = left.copy()

            assert left_hold is not None
            src_obj_pos, src_obj_quat = _body_pose(raw, PEG_BODY)
            src_seg_right = [
                np.asarray(raw_flat_to_dict(actions[t])["right"], dtype=np.float64).copy()
                for t in range(seg_start, seg_end)
            ]

            weak_end = int(np.clip(right_grasp + 5, pre_weak + 1, n - 1))
            lift_end = int(np.clip(peg_lift + no_follow_lift_frames, weak_end + 1, n - 1))
            # Underclose near grasp, then open-hand lift (peg stays, less drag than full underclose).
            for t in range(pre_weak, weak_end + 1):
                act = raw_flat_to_dict(actions[t])
                right = _with_underclose(act["right"], scale=underclose_scale)
                left = np.asarray(act["left"], dtype=np.float64)
                _step_dict(raw, right, left, video)
                left_hold = left.copy()
            for t in range(weak_end + 1, lift_end + 1):
                act = raw_flat_to_dict(actions[t])
                right = _open_hand(act["right"], scale=0.05)
                right[0:7] = np.asarray(act["right"], dtype=np.float64)[0:7]
                left = np.asarray(act["left"], dtype=np.float64)
                _step_dict(raw, right, left, video)
                left_hold = left.copy()

            cur = read_arm_action(raw, "right")
            away = _open_hand(cur, scale=0.05)
            away[2] = float(cur[2] + 0.03)
            for i in range(24):
                t = min_jerk((i + 1) / 24.0)
                _step_dict(raw, _interp_action23(cur, away, t), left_hold, video)
            hold = read_arm_action(raw, "right")
            for _ in range(36):
                _step_dict(raw, hold, left_hold, video)
            planner_info = {
                "grasp_fail_retreat": "underclose_then_open_lift",
                "weak_end": weak_end,
                "lift_end": lift_end,
            }

            peg_pos_cur, peg_quat_cur = _body_pose(raw, PEG_BODY)
            peg_z_fail = float(peg_pos_cur[2])

        else:
            raise ValueError(f"unknown mode {mode}")

        assert src_seg_right and src_obj_pos is not None and src_obj_quat is not None
        assert left_hold is not None

        tray_pos, _ = _body_pose(raw, TRAY_BODY)
        se3_pos, se3_quat, ori_info = _pose_for_se3(
            src_obj_pos,
            src_obj_quat,
            peg_pos_cur,
            peg_quat_cur,
            axisymmetric=True,
            prefer_src_quat=True,
        )
        ori_delta = float(ori_info["ori_delta_deg"])
        axis_delta = float(ori_info["axis_delta_deg"])
        peg_tray_xy = float(np.linalg.norm(peg_pos_cur[:2] - tray_pos[:2]))
        z_min = float(peg_z_ref - 0.03)

        src_seg = np.stack(src_seg_right, axis=0)
        # If the peg barely moved, SE(3) warps absolute demo mocap and kills the grasp.
        # Use identity replay; only SE(3) when the object really translated/tilted.
        peg_shift = float(np.linalg.norm(peg_pos_cur - src_obj_pos))
        # drop: always SE3. grasp_fail: SE3 only if peg really shifted (ep14 ~3cm needs it;
        # ep10 ~7mm must stay identity — SE3 warps sliding-grasp demos).
        use_se3 = mode == "drop" or peg_shift >= 0.025 or axis_delta >= 20.0
        if use_se3:
            transformed = transform_action23_segment(
                src_seg,
                src_obj_pos,
                src_obj_quat,
                se3_pos,
                se3_quat,
            )
        else:
            transformed = src_seg.copy()
        first_wrist = vec_to_arm_action(transformed[0])
        wrist_xy_reach = float(np.linalg.norm(first_wrist[0:2]))

        print(
            f"[diag] ep{ep} {mode} before_regrasp: "
            f"oriΔ={ori_delta:.1f}° axisΔ={axis_delta:.1f}° shift={peg_shift:.3f} se3={use_se3} "
            f"peg_tray_xy={peg_tray_xy:.3f} wrist_xy_reach={wrist_xy_reach:.3f} "
            f"peg_z_fail={peg_z_fail:.3f} slip={slip_fingers} info={planner_info}",
            flush=True,
        )

        regrasp_info = _execute_mimicgen_regrasp(
            raw,
            transformed_right=transformed,
            left_hold=left_hold,
            video=video,
            z_min=z_min,
        )
        regrasp_info = {**regrasp_info, "used_se3": bool(use_se3), "peg_shift_m": round(peg_shift, 4)}
        peg_pos_final, _ = _body_pose(raw, PEG_BODY)
        peg_z_final = float(peg_pos_final[2])
        success = (peg_z_final >= peg_z_fail + success_lift_m) or (
            peg_z_final >= peg_z_ref + success_lift_m
        )
        fail_reason = _classify_fail(
            success=success,
            peg_z_fail=peg_z_fail,
            peg_z_final=peg_z_final,
            peg_z_ref=peg_z_ref,
            ori_delta_deg=ori_delta,
            axis_delta_deg=axis_delta,
            peg_tray_xy=peg_tray_xy,
            wrist_xy_reach=wrist_xy_reach,
            success_lift_m=success_lift_m,
            regrasp_attempted=True,
        )
        diagnostics = {
            "fail_reason": fail_reason,
            "ori_delta_deg": round(ori_delta, 2),
            "axis_delta_deg": round(axis_delta, 2),
            "peg_tray_xy_m": round(peg_tray_xy, 4),
            "wrist_xy_reach_m": round(wrist_xy_reach, 4),
            "peg_z_ref": round(peg_z_ref, 4),
            "peg_z_fail": round(peg_z_fail, 4),
            "peg_z_final": round(peg_z_final, 4),
            "delta_z_m": round(float(peg_z_final - peg_z_fail), 4),
            "slip_fingers": slip_fingers,
            "planner": planner_info,
            "regrasp_planner": regrasp_info,
        }
        msg = (
            f"ep{ep} mode={mode} regrasp {'OK' if success else 'FAIL'} "
            f"reason={fail_reason}: peg_z fail={peg_z_fail:.3f} final={peg_z_final:.3f} "
            f"oriΔ={ori_delta:.1f}° axisΔ={axis_delta:.1f}° tray_xy={peg_tray_xy:.3f}"
        )
        print(f"[diag] {msg}", flush=True)
        return RegraspResult(
            episode_index=ep,
            success=bool(success),
            peg_z_after_fail=peg_z_fail,
            peg_z_final=peg_z_final,
            mode=mode,
            video_path=str(video_path) if record_video else "",
            message=msg,
            diagnostics=diagnostics,
        )
    finally:
        if video is not None:
            video.close()
        env.close()


def load_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_entry(manifest: dict[str, Any], episode_index: int | None) -> dict[str, Any]:
    episodes = manifest["episodes"]
    if episode_index is not None:
        for e in episodes:
            if int(e["episode_index"]) == int(episode_index):
                return e
        raise KeyError(f"episode {episode_index} not in manifest")
    for e in episodes:
        t = e["timing"]
        if (
            t.get("right_grasp_frame") is not None
            and t.get("peg_lift_start") is not None
            and not t.get("right_grasp_fallback", False)
            and int(e["episode_index"]) >= 10
        ):
            return e
    return episodes[0]

