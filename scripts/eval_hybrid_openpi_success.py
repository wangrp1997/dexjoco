#!/usr/bin/env python3
"""Eval hybrid_insert with DexJoCo openpi success only (30-step insert contact)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO = Path(__file__).resolve().parents[1]
_LAI = Path("/home/wangrenpeng/lai")
for p in (_REPO, _REPO / "dexjoco", _LAI):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from hybrid_insert.assembly_contacts import AssemblyContactLabeler
from hybrid_insert.config import HybridInsertConfig
from hybrid_insert.controller import HybridInsertController
from hybrid_insert.geometry import (
    height_along_axis,
    insert_along_hole_delta,
    lateral_error,
    toward_socket_delta,
)
from interaction_retarget.constants import default_sidecar_dir
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.replay import make_assembly_env, rotvec_dual_arm_to_policy
from interaction_retarget.sim.settle import read_arm_action
from interaction_retarget.skill_replay.insert import (
    _insert_geometry,
    dual_arm23_to_action44,
    demo_replay_to_pre_insert,
)
from pose_insert.pre_insert import resolve_peg_lift_end_frame

DEFAULT_OUT = Path("/mnt/hdd/dexjoco/outputs/hybrid_insert_openpi_success")
SIDECAR = default_sidecar_dir("bimanual_assembly")
MAX_STEPS = 1500
APPROACH_CAP = 500


def _manifest(episodes: list[int] | None) -> list[dict]:
    manifest = json.loads((SIDECAR / "manifest.json").read_text(encoding="utf-8"))
    out = []
    for entry in manifest["episodes"]:
        ep = int(entry["episode_index"])
        if episodes is not None and ep not in episodes:
            continue
        timing = entry.get("timing", {})
        if timing.get("peg_lift_start") is None or timing.get("right_grasp_frame") is None:
            continue
        out.append(entry)
    return out


def _hybrid_config(insert_budget: int) -> HybridInsertConfig:
    return HybridInsertConfig(
        freeze_left_arm_at_handoff=False,
        left_share_xy=0.45,
        left_share_rot=0.40,
        tray_z_up_enable=False,
        pbvs_enter_tip_m=0.100,
        pbvs_spiral_min_tip_m=0.120,
        pbvs_rel_axis_min_tip_m=0.060,
        handoff_settle_frames=8,
        peg_lost_abort_frames=12,
        max_align_steps=insert_budget,
        max_insert_steps=insert_budget,
        align_debug_interval=99999,
        insert_debug_interval=99999,
        release_confirm_frames=2,
        pos_tol_m=0.004,
        angle_tol_rad=0.14,
        axis_align_max_lat_m=0.008,
        insert_align_confirm_frames=8,
        wrist_tip_scale=0.95,
        insert_wrist_tip_scale=0.9,
        max_wrist_step_m=0.004,
        max_left_wrist_step_m=0.0035,
        align_pos_gain=0.55,
        align_rot_gain=0.20,
        max_wrist_rot_step_rad=0.012,
        insert_along_step_m=0.003,
        pbvs_lambda_xy=0.7,
        pbvs_lambda_z=0.45,
        pbvs_lambda_rot=0.22,
        pbvs_standoff_m=0.055,
        pbvs_insert_target_along_m=0.02,
        pbvs_stall_frames=35,
        pbvs_retreat_frames=20,
        pbvs_retreat_step_m=0.0025,
        stop_lateral_tip_m=0.050,
        soft_seat_tip_m=0.038,
        seated_along_m=0.038,
        seated_lat_m=0.010,
        tip_jam_frames=10,
        tip_jam_improve_m=0.0004,
        max_insert_z_step_m=0.0015,
        tray_z_up_enable_tip_m=0.100,
        tray_z_up_disable_tip_m=0.045,
        tray_z_up_tol_rad=0.10,
        tray_z_up_gain=0.28,
        max_left_wrist_rot_step_rad=0.015,
        release_insert_socket_dist_m=0.038,
        release_steps=50,
        release_retract_m=0.010,
    )


def _approach(
    env,
    raw,
    labeler,
    left_hold,
    right_hand_hold,
    *,
    video_cb=None,
) -> str | None:
    best_dist = float(_insert_geometry(raw)[3])
    stall = 0
    for a in range(APPROACH_CAP):
        outcome = labeler.compute(raw)
        tip, socket, hole, dist = _insert_geometry(raw)
        lat_n, _ = lateral_error(tip, socket, hole)
        along = height_along_axis(tip, socket, hole)
        if not bool(outcome.peg_ok):
            return "peg_lost_approach"
        if dist + 1e-4 < best_dist:
            best_dist = float(dist)
            stall = 0
        else:
            stall += 1
        if lat_n <= 0.010 and along <= 0.13:
            return None
        if stall >= 80 and lat_n <= 0.014 and along <= 0.14:
            return None
        right = np.asarray(read_arm_action(raw, "right"), dtype=np.float64).copy()
        right[7:23] = right_hand_hold
        if along > 0.14 or dist > 0.18:
            delta = toward_socket_delta(tip, socket, gain=0.45, max_step_m=0.0035)
        else:
            _, lat_v = lateral_error(tip, socket, hole)
            delta = -0.8 * lat_v
            if lat_n <= 0.012 and along > 0.06:
                delta = delta + insert_along_hole_delta(hole, step_m=0.0012)
            n = float(np.linalg.norm(delta))
            if n > 0.0035:
                delta = delta * (0.0035 / n)
        right[0:3] = right[0:3] + delta
        policy44 = dual_arm23_to_action44(left_hold, right)
        action46 = rotvec_dual_arm_to_policy(np.asarray(policy44, dtype=np.float64).reshape(44))
        obs = env.step(action46)[0]
        if video_cb is not None:
            video_cb(obs)
    return None


def run_episode(entry: dict, *, seed: int) -> dict:
    ep = int(entry["episode_index"])
    peg_lift_end = resolve_peg_lift_end_frame(entry, SIDECAR)
    env = make_assembly_env(seed=seed, randomize=False)
    raw = env.unwrapped
    labeler = AssemblyContactLabeler(raw)
    try:
        _, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
        demo_replay_to_pre_insert(
            env,
            raw,
            zarr_path=entry["zarr_path"],
            stop_frame=int(peg_lift_end),
            initial_state=initial_state,
            video_cb=None,
            labeler=labeler,
        )
        left_hold = np.asarray(read_arm_action(raw, "left"), dtype=np.float64).copy()
        right0 = np.asarray(read_arm_action(raw, "right"), dtype=np.float64).copy()
        right_hand_hold = right0[7:23].copy()
        fail = _approach(env, raw, labeler, left_hold, right_hand_hold)
        if fail:
            return {
                "episode_index": ep,
                "openpi_success": False,
                "fail_reason": fail,
                "steps": 0,
                "max_insert_streak": 0,
            }
        if not bool(labeler.compute(raw).peg_ok):
            return {
                "episode_index": ep,
                "openpi_success": False,
                "fail_reason": "peg_lost_before_hybrid",
                "steps": 0,
                "max_insert_streak": 0,
            }

        insert_budget = MAX_STEPS - int(getattr(raw, "env_step", 0))
        hctrl = HybridInsertController(_hybrid_config(insert_budget))
        hctrl.reset(raw)
        hctrl._peg_rest_z = float(labeler._peg_rest_z)  # noqa: SLF001
        if hctrl._labeler is not None:
            hctrl._labeler._tray_rest_z = float(labeler._tray_rest_z)  # noqa: SLF001
            hctrl._labeler._peg_rest_z = float(labeler._peg_rest_z)  # noqa: SLF001
        left = read_arm_action(raw, "left")
        right = read_arm_action(raw, "right")
        policy44 = dual_arm23_to_action44(left, right)
        hctrl._activate(policy44, raw)  # noqa: SLF001

        openpi_success = False
        streak = 0
        max_streak = 0
        steps = 0
        fail_reason = ""
        for _ in range(insert_budget):
            left = read_arm_action(raw, "left")
            right = read_arm_action(raw, "right")
            policy44 = dual_arm23_to_action44(left, right)
            action44 = hctrl.merge_right_arm(raw, policy44)
            action46 = rotvec_dual_arm_to_policy(np.asarray(action44, dtype=np.float64).reshape(44))
            _, _, terminated, truncated, info = env.step(action46)
            steps += 1
            outcome = labeler.compute(raw)
            if bool(outcome.insert_ok):
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
            if bool(info.get("succeed", False)):
                openpi_success = True
                break
            if terminated or truncated:
                openpi_success = bool(info.get("succeed", False))
                if not openpi_success:
                    fail_reason = "env_done_no_success"
                break
        else:
            fail_reason = fail_reason or "max_steps"
        return {
            "episode_index": ep,
            "openpi_success": bool(openpi_success),
            "fail_reason": fail_reason,
            "steps": steps,
            "max_insert_streak": max_streak,
            "final_phase": hctrl.phase_name,
            "final_peg_ok": bool(labeler.compute(raw).peg_ok),
        }
    finally:
        env.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    if args.all:
        episodes = None
    elif args.episodes:
        episodes = list(args.episodes)
    else:
        episodes = [0, 51, 98]
    entries = _manifest(episodes)
    run_dir = args.out_dir / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in entries:
        ep = int(entry["episode_index"])
        print(f"[openpi-hybrid] ep={ep}", flush=True)
        r = run_episode(entry, seed=args.seed)
        results.append(r)
        print(
            f"  ok={r['openpi_success']} streak={r['max_insert_streak']} "
            f"phase={r.get('final_phase')} reason={r.get('fail_reason')}",
            flush=True,
        )
    n_ok = sum(1 for r in results if r["openpi_success"])
    summary = {
        "protocol": "hybrid_insert_openpi_30step_insert_contact",
        "n_ok": n_ok,
        "n_total": len(results),
        "ok_rate": n_ok / max(1, len(results)),
        "results": results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"n_ok": n_ok, "n_total": len(results), "run_dir": str(run_dir)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
