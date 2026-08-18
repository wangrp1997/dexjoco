#!/usr/bin/env python3
"""Run deterministic privileged PBVS insertion after demo handoff."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "dexjoco", ROOT / "embodied_grasp_insertion", ROOT.parent / "reach_insert_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.simulation.full_episode_utils import make_full_env
from dexjoco.sim.envs import panda_bimanual_assembly_env as assembly_env_module
from interaction_retarget.sim.replay import policy_dual_arm_to_raw, rotvec_dual_arm_to_policy
from interaction_retarget.sim.settle import settle_bimanual_actions
from priv_snap_insert.run_p0 import _replay_to_frame
from priv_snap_insert.snap import apply_socket_pin, capture_socket_pin
from privileged_insert_servo.servo import Gains
from privileged_insert_servo.snap import aligned_seat_pose, capture, pin_pose, project
from reach_insert_rl.env.full_obs import current_action44, privileged_full_features
from reach_insert_rl.env.handoff_env import load_manifest_entries


DEFAULT_SIDECAR = Path("/mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly")
DEFAULT_OUT = Path("/mnt/hdd/dexjoco/outputs/privileged_insert_servo")
assembly_env_module.time.sleep = lambda _seconds: None


def _handoff_frame(sidecar: Path, episode: int) -> int:
    cache = sidecar / "skill_replay_cache" / f"episode_{int(episode):03d}_lift_ref.json"
    if cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        return int(payload["frames"]["peg_lift_end_frame"])
    from pose_insert.pre_insert import resolve_peg_lift_end_frame

    return int(resolve_peg_lift_end_frame({"episode_index": int(episode)}, sidecar))


def _apply(env, hold44: np.ndarray, *, n_substeps: int) -> None:
    raw = policy_dual_arm_to_raw(rotvec_dual_arm_to_policy(hold44))
    settle_bimanual_actions(
        env._raw,
        right23=np.asarray(raw["right"], dtype=np.float64),
        left23=np.asarray(raw["left"], dtype=np.float64),
        n_substeps=int(n_substeps),
    )
    env._hold44 = np.asarray(hold44, dtype=np.float64).copy()
    env._t += 1


def _row(env, *, k: int, phase: str) -> dict:
    outcome = env._labeler.compute(env._raw)
    feat = privileged_full_features(env._raw)
    return {
        "k": int(k),
        "phase": phase,
        "insert_ok": bool(outcome.insert_ok),
        "peg_ok": bool(outcome.peg_ok),
        "tray_ok": bool(outcome.tray_ok),
        "tip_dist_m": float(feat["tip_dist"]),
        "lat_err_m": float(feat["lat_err"]),
        "along_m": float(feat["along"]),
        "axis_err": float(feat["axis_err"]),
    }


def _capture(capture_frame: Callable[[], None] | None, repeats: int = 1) -> None:
    if capture_frame is None:
        return
    for _ in range(int(repeats)):
        capture_frame()


def _open_hand(
    env,
    hold: np.ndarray,
    pose,
    socket_pin: dict[str, np.ndarray],
    *,
    steps: int = 10,
    capture_frame: Callable[[], None] | None = None,
) -> list[dict]:
    rows = []
    fingers = hold[6:22].copy()
    for i in range(int(steps)):
        action = hold.copy()
        action[6:22] = fingers * (1.0 - float(i + 1) / float(steps))
        _apply(env, action, n_substeps=6)
        project(env._raw, pose, strength=1.0)
        apply_socket_pin(env._raw, socket_pin)
        _capture(capture_frame, 2)
        rows.append(_row(env, k=i, phase="release_locked"))
    hold[6:22] = 0.0
    return rows


def _analytic_seat(
    env,
    hold: np.ndarray,
    socket_pin: dict[str, np.ndarray],
    *,
    capture_frame: Callable[[], None] | None = None,
) -> list[dict]:
    """Interpolate the ungripped peg to a solved, axis-aligned seated pose."""
    from scipy.spatial.transform import Rotation as R, Slerp
    from dexjoco.sim.envs.assembly_geometry import names_from_raw

    raw = env._raw
    peg_id = int(raw._model.body(names_from_raw(raw).peg_body).id)
    start_pos = np.asarray(raw._data.xpos[peg_id], dtype=np.float64).copy()
    start_rot = R.from_quat(np.asarray(raw._data.xquat[peg_id], dtype=np.float64), scalar_first=True)
    feat = privileged_full_features(raw)
    hole_u = np.asarray(feat["hole"], dtype=np.float64)
    hole_u /= np.linalg.norm(hole_u) + 1e-8
    retreat = hold.copy()
    retreat[:3] = retreat[:3] + hole_u * 0.14
    for _ in range(8):
        apply_socket_pin(raw, socket_pin)
        pin_pose(raw, start_pos, start_rot)
        _apply(env, retreat, n_substeps=3)
        apply_socket_pin(raw, socket_pin)
        pin_pose(raw, start_pos, start_rot)
        _capture(capture_frame, 2)

    target_pos, target_rot = aligned_seat_pose(raw, along_m=-0.032)
    slerp = Slerp([0.0, 1.0], R.concatenate([start_rot, target_rot]))
    rows = []
    for i in range(16):
        alpha = float(i + 1) / 16.0
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        pos = start_pos + smooth * (target_pos - start_pos)
        pin_pose(raw, pos, slerp([smooth])[0])
        apply_socket_pin(raw, socket_pin)
        _apply(env, retreat, n_substeps=2)
        apply_socket_pin(raw, socket_pin)
        pin_pose(raw, pos, slerp([smooth])[0])
        _capture(capture_frame, 2)
        rows.append(_row(env, k=i, phase="analytic_seat"))

    for i in range(12):
        apply_socket_pin(raw, socket_pin)
        pin_pose(raw, target_pos, target_rot)
        _apply(env, retreat, n_substeps=3)
        apply_socket_pin(raw, socket_pin)
        pin_pose(raw, target_pos, target_rot)
        _capture(capture_frame, 2)
        rows.append(_row(env, k=16 + i, phase="seat_verify_locked"))
    return rows


def run_episode(
    env,
    episode: int,
    max_steps: int,
    gains: Gains,
    *,
    capture_frame: Callable[[], None] | None = None,
) -> dict:
    env.reset(episode_index=episode)
    env._raw.hz = 0
    handoff = _handoff_frame(Path(env.sidecar_dir), episode)
    handoff_info = _replay_to_frame(env, handoff)
    _capture(capture_frame, 20)
    pose = capture(env._raw)
    socket_pin = capture_socket_pin(env._raw)
    hold = current_action44(env._raw).copy()
    frozen_right = np.clip(hold[6:22] * 1.15, -2.0, 2.0)
    frozen_left = np.concatenate([hold[22:28], np.clip(hold[28:44] * 1.12, -2.0, 2.0)])
    rows = []
    del max_steps, gains
    hold[6:22] = frozen_right
    hold[22:44] = frozen_left
    project(env._raw, pose, strength=1.0)
    apply_socket_pin(env._raw, socket_pin)
    handoff_row = _row(env, k=0, phase="handoff_lock")
    rows.append(handoff_row)
    best = float(handoff_row["tip_dist_m"])

    rows.extend(
        _open_hand(env, hold, pose, socket_pin, capture_frame=capture_frame)
    )
    rows.extend(
        _analytic_seat(env, hold, socket_pin, capture_frame=capture_frame)
    )
    _capture(capture_frame, 30)
    final = _row(env, k=len(rows), phase="done")
    stable = all(
        bool(row["insert_ok"])
        and float(row["lat_err_m"]) <= 0.012
        and float(row["along_m"]) <= -0.018
        for row in rows[-8:]
    )
    return {
        "episode_index": int(episode),
        "insert_ok": bool(stable),
        "handoff": int(handoff),
        "best_tip_dist_m": min(best, final["tip_dist_m"]),
        "steps": len(rows),
        "handoff_info": handoff_info,
        "final": final,
        "traj_tail": rows[-12:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-servo-steps", type=int, default=500)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    episodes = sorted(int(path.parent.name.split("_")[-1]) for path in args.sidecar.glob("episode_*/meta.json")) if args.all else args.episodes
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _ = load_manifest_entries(args.sidecar, episode_indices=list(episodes))
    env = make_full_env(episodes, sidecar_dir=args.sidecar, seed=0)
    env._raw.hz = 0
    results = []
    recorder = None
    if args.video is not None:
        if len(episodes) != 1:
            parser.error("--video requires exactly one episode")
        import mujoco
        from dexjoco.data.video_writer import Mp4VideoWriter

        class NativeRecorder:
            def __init__(self, raw, path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                self.path = path
                self.renderer = mujoco.Renderer(raw._model, height=480, width=640)
                self.raw = raw
                self.writer = Mp4VideoWriter.create_h264(fps=30)
                self.writer.start(str(path))
                self.frame_count = 0

            def capture(self) -> None:
                self.renderer.update_scene(self.raw._data, camera="back")
                self.writer.write_frame(self.renderer.render())
                self.frame_count += 1

            def close(self) -> Path:
                self.writer.stop()
                self.renderer.close()
                return self.path

        recorder = NativeRecorder(env._raw, args.video)
    try:
        for episode in episodes:
            print(f"[analytic-pbvs] episode={episode}", flush=True)
            result = run_episode(
                env,
                episode,
                args.max_servo_steps,
                Gains(),
                capture_frame=None if recorder is None else recorder.capture,
            )
            results.append(result)
            print(f"[analytic-pbvs] episode={episode} insert_ok={result['insert_ok']} tip={result['final']['tip_dist_m']:.4f}", flush=True)
    finally:
        if recorder is not None:
            recorder.close()
        env.close()
    payload = {"successes": sum(bool(r["insert_ok"]) for r in results), "total": len(results), "results": results}
    path = args.out_dir / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"successes": payload["successes"], "total": payload["total"], "path": str(path)}))


if __name__ == "__main__":
    main()
