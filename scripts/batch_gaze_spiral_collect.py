#!/usr/bin/env python3
"""Batch: official 100 demos -> handoff -> ascending spiral -> ego JPEG + labels (no video)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
for p in (_REPO, _REPO / "dexjoco", _SCRIPTS, Path("/home/wangrenpeng/lai")):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from eval_hybrid_openpi_success import _approach, _manifest
from hybrid_insert.assembly_contacts import AssemblyContactLabeler
from interaction_retarget.constants import default_sidecar_dir
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.replay import make_assembly_env, rotvec_dual_arm_to_policy
from interaction_retarget.sim.settle import read_arm_action
from interaction_retarget.skill_replay.insert import (
    _insert_geometry,
    demo_replay_to_pre_insert,
    dual_arm23_to_action44,
)
from pose_insert.pre_insert import resolve_peg_lift_end_frame
from retrieval_cerebellum.spatial_visual_supervision import CameraCalibration
from smoke_gaze_scan_collect import (
    DEFAULT_HOLE_OPENING_OFFSET_M,
    _build_scan_targets,
    _hole_opening_world,
    _step_dual,
    _world_separation,
)

SIDECAR = default_sidecar_dir("bimanual_assembly")
DEFAULT_OUT = Path("/mnt/hdd/dexjoco/datasets/gaze_spiral_ego_100")


def _ego_u8(raw) -> np.ndarray:
    fr = raw.render()[0]
    arr = np.asarray(fr)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return arr


def _project_ego(raw, *, hole_opening_offset_m: float) -> dict:
    tip, socket, hole_axis, _ = _insert_geometry(raw)
    hole = _hole_opening_world(socket, hole_axis, hole_opening_offset_m)
    cam_id = int(raw._front_camera_id)
    ego = _ego_u8(raw)
    h, w = ego.shape[:2]
    cal = CameraCalibration.from_vertical_fov(
        width=w,
        height=h,
        vertical_fov_degrees=float(raw._model.cam_fovy[cam_id]),
        position_world=np.asarray(raw._data.cam_xpos[cam_id]),
        rotation_world_from_camera=np.asarray(raw._data.cam_xmat[cam_id]).reshape(3, 3),
    )
    uv, depth, in_frame = cal.project(np.stack([tip, hole]))
    axis_u = hole_axis / (np.linalg.norm(hole_axis) + 1e-12)
    along = float(np.dot(tip - hole, axis_u))
    tip_occluded = along < -0.005
    return {
        "tip_u": float(uv[0, 0]),
        "tip_v": float(uv[0, 1]),
        "hole_u": float(uv[1, 0]),
        "hole_v": float(uv[1, 1]),
        "tip_depth": float(depth[0]),
        "hole_depth": float(depth[1]),
        "tip_in_frame": bool(in_frame[0]),
        "hole_in_frame": bool(in_frame[1]),
        "tip_visible": bool(in_frame[0]) and not tip_occluded,
        "hole_visible": bool(in_frame[1]),
        "along_mm": along * 1000.0,
    }


def _stop_frame_candidates(entry: dict, sidecar_dir: Path) -> list[int]:
    """Primary peg_lift_end, then earlier lift frames if grasp is fragile."""
    primary = int(resolve_peg_lift_end_frame(entry, sidecar_dir))
    timing = entry.get("timing") or {}
    pls = timing.get("peg_lift_start")
    cands = [primary]
    if pls is not None:
        pls = int(pls)
        early = pls + 25
        mid = pls + max(20, (primary - pls) // 2)
        for frame in (early, mid):
            if 0 < frame < primary and frame not in cands:
                cands.append(frame)
    return cands


def _settle_hold(
    env,
    raw,
    left_hold: np.ndarray,
    right_hand_hold: np.ndarray,
    *,
    steps: int = 20,
) -> None:
    """Hold both arms (lock right fingers) before privileged approach."""
    right = np.asarray(read_arm_action(raw, "right"), dtype=np.float64).copy()
    right[7:23] = np.asarray(right_hand_hold, dtype=np.float64)
    for _ in range(int(steps)):
        policy44 = dual_arm23_to_action44(left_hold, right)
        action46 = rotvec_dual_arm_to_policy(np.asarray(policy44, dtype=np.float64).reshape(44))
        env.step(action46.astype(np.float32))


def _replay_approach(
    env,
    raw,
    labeler: AssemblyContactLabeler,
    entry: dict,
    initial_state,
    stop_frame: int,
) -> tuple[str | None, int | None]:
    demo_replay_to_pre_insert(
        env,
        raw,
        zarr_path=entry["zarr_path"],
        stop_frame=int(stop_frame),
        initial_state=initial_state,
        labeler=labeler,
    )
    if not bool(labeler.compute(raw).peg_ok):
        return "peg_lost_after_replay", None
    left0 = np.asarray(read_arm_action(raw, "left"), dtype=np.float64).copy()
    right0 = np.asarray(read_arm_action(raw, "right"), dtype=np.float64).copy()
    _settle_hold(env, raw, left0, right0[7:23].copy())
    fail = _approach(env, raw, labeler, left0, right0[7:23].copy())
    if fail:
        return fail, None
    return None, int(stop_frame)


def _episode_done(ep_dir: Path, expected_frames: int) -> bool:
    meta = ep_dir / "meta.json"
    labels = ep_dir / "labels.parquet"
    if not meta.exists() or not labels.exists():
        return False
    try:
        n = pq.read_table(labels).num_rows
        return n >= expected_frames
    except Exception:
        return False


def collect_episode(
    entry: dict,
    *,
    out_root: Path,
    hole_opening_offset_m: float,
    jpeg_quality: int,
    spiral_frames: int,
    spiral_along_max_m: float,
) -> dict:
    ep = int(entry["episode_index"])
    ep_dir = out_root / f"episode_{ep:02d}"
    frames_dir = ep_dir / "frames"
    expected = spiral_frames  # no separate open by default

    if _episode_done(ep_dir, expected):
        return {"episode_index": ep, "status": "skipped", "num_frames": expected}

    peg_lift_end = resolve_peg_lift_end_frame(entry, SIDECAR)
    stop_candidates = _stop_frame_candidates(entry, SIDECAR)
    seed = ep
    env = make_assembly_env(seed=seed, randomize=False, render_mode="rgb_array")
    raw = env.unwrapped
    labeler = AssemblyContactLabeler(raw)
    rows: list[dict] = []
    used_stop_frame: int | None = None
    approach_attempts: list[dict] = []

    try:
        _, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
        fail_reason = "peg_lost_approach"
        for stop_frame in stop_candidates:
            attempt_fail, ok_stop = _replay_approach(
                env, raw, labeler, entry, initial_state, stop_frame
            )
            approach_attempts.append(
                {"stop_frame": int(stop_frame), "fail": attempt_fail}
            )
            if ok_stop is not None:
                used_stop_frame = ok_stop
                fail_reason = ""
                break
            fail_reason = attempt_fail or fail_reason

        if used_stop_frame is None:
            return {
                "episode_index": ep,
                "status": "fail_approach",
                "reason": fail_reason,
                "attempts": approach_attempts,
            }

        left_now = np.asarray(read_arm_action(raw, "left"), dtype=np.float64).copy()
        right_now = np.asarray(read_arm_action(raw, "right"), dtype=np.float64).copy()
        left_quat_ref, left_hand = left_now[3:7].copy(), left_now[7:23].copy()
        right_quat_ref, right_hand = right_now[3:7].copy(), right_now[7:23].copy()
        tip0, _, hole_axis, _ = _insert_geometry(raw)
        tip_to_mocap = right_now[0:3] - tip0
        left_sep = _world_separation(left_now[0:3], right_now[0:3])

        targets = _build_scan_targets(
            left0=left_now[0:3],
            tip0=tip0,
            hole_axis=hole_axis,
            left_sep=left_sep,
            spiral_frames=spiral_frames,
            spiral_along_max_m=spiral_along_max_m,
        )
        if ep_dir.exists():
            import shutil
            shutil.rmtree(ep_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        for fi, (phase, left_tgt, tip_tgt, l_wob, r_wob) in enumerate(targets):
            tip_to_mocap = _step_dual(
                env,
                raw,
                left_quat_ref=left_quat_ref,
                left_hand=left_hand,
                right_quat_ref=right_quat_ref,
                right_hand=right_hand,
                left_pos_target=left_tgt,
                right_tip_target=tip_tgt,
                tip_to_mocap=tip_to_mocap,
                left_quat_wobble=l_wob,
                right_quat_wobble=r_wob,
            )
            ego = _ego_u8(raw)
            rel = f"frames/{fi:05d}.jpg"
            cv2.imwrite(
                str(ep_dir / rel),
                cv2.cvtColor(ego, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            lab = _project_ego(raw, hole_opening_offset_m=hole_opening_offset_m)
            rows.append({"frame": fi, "phase": phase, "image": rel, **lab})

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, ep_dir / "labels.parquet")
        meta = {
            "episode_index": ep,
            "seed": seed,
            "zarr_path": str(entry["zarr_path"]),
            "peg_lift_end": int(peg_lift_end),
            "stop_frame_used": int(used_stop_frame),
            "approach_attempts": approach_attempts,
            "num_frames": len(rows),
            "camera": "ego",
            "format": "jpeg",
            "no_video": True,
            "hole_opening_offset_m": hole_opening_offset_m,
            "spiral_along_max_m": spiral_along_max_m,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return {"episode_index": ep, "status": "ok", "num_frames": len(rows)}
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--hole-opening-offset-m", type=float, default=DEFAULT_HOLE_OPENING_OFFSET_M)
    parser.add_argument("--spiral-frames", type=int, default=220)
    parser.add_argument("--spiral-along-max-m", type=float, default=0.045)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    args = parser.parse_args()

    wanted = set(args.episodes) if args.episodes else None
    entries = _manifest(None if wanted is None else sorted(wanted))
    if args.max_episodes is not None:
        entries = entries[: int(args.max_episodes)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "batch.log"
    summary_path = args.out_dir / "summary.jsonl"

    ok = skip = fail = 0
    t0 = time.time()
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n=== batch start {datetime.now(timezone.utc).isoformat()} ===\n")
        for entry in entries:
            ep = int(entry["episode_index"])
            logf.write(f"episode {ep} ... ")
            logf.flush()
            try:
                result = collect_episode(
                    entry,
                    out_root=args.out_dir,
                    hole_opening_offset_m=args.hole_opening_offset_m,
                    jpeg_quality=args.jpeg_quality,
                    spiral_frames=args.spiral_frames,
                    spiral_along_max_m=args.spiral_along_max_m,
                )
            except Exception as exc:
                result = {
                    "episode_index": ep,
                    "status": "error",
                    "reason": str(exc),
                    "trace": traceback.format_exc(),
                }
            with summary_path.open("a", encoding="utf-8") as sf:
                sf.write(json.dumps(result) + "\n")
            st = result.get("status")
            if st == "ok":
                ok += 1
            elif st == "skipped":
                skip += 1
            else:
                fail += 1
            logf.write(json.dumps(result) + "\n")
            logf.flush()
            print(json.dumps(result), flush=True)

    elapsed = time.time() - t0
    print(
        json.dumps(
            {
                "done": True,
                "ok": ok,
                "skipped": skip,
                "fail": fail,
                "elapsed_s": elapsed,
                "out_dir": str(args.out_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
