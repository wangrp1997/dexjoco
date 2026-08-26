#!/usr/bin/env python3
"""Smoke: handoff -> approach -> dual small Cartesian scan + privileged tip/孔口 labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

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

CAMERA_NAMES = ("ego", "wrist_left", "wrist_right")
FPS = 10
SIDECAR = default_sidecar_dir("bimanual_assembly")

# socket_site 在孔底；视觉孔口沿 hole_axis 外偏约 10cm（见 XML 几何）。
DEFAULT_HOLE_OPENING_OFFSET_M = 0.10


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(axis, dtype=np.float64).reshape(3)
    z = z / (np.linalg.norm(z) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(tmp, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    return x, y, z


def _hole_opening_world(socket: np.ndarray, hole_axis: np.ndarray, offset_m: float) -> np.ndarray:
    axis = np.asarray(hole_axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return np.asarray(socket, dtype=np.float64) + axis * float(offset_m)


def _slerp_quat_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0_xyzw = q0[[1, 2, 3, 0]]
    q1_xyzw = q1[[1, 2, 3, 0]]
    out = Slerp([0.0, 1.0], R.from_quat([q0_xyzw, q1_xyzw]))(float(np.clip(t, 0.0, 1.0))).as_quat()
    return np.asarray([out[3], out[0], out[1], out[2]], dtype=np.float64)


def _camera_calibration(raw, camera_id: int, width: int, height: int) -> CameraCalibration:
    return CameraCalibration.from_vertical_fov(
        width=width,
        height=height,
        vertical_fov_degrees=float(raw._model.cam_fovy[camera_id]),
        position_world=np.asarray(raw._data.cam_xpos[camera_id]),
        rotation_world_from_camera=np.asarray(raw._data.cam_xmat[camera_id]).reshape(3, 3),
    )


def _project_tip_hole(
    raw,
    frames: list[np.ndarray],
    *,
    hole_opening_offset_m: float,
) -> dict[str, dict]:
    tip, socket, hole_axis, _ = _insert_geometry(raw)
    hole = _hole_opening_world(socket, hole_axis, hole_opening_offset_m)
    cam_ids = (
        int(raw._front_camera_id),
        int(raw._wrist_left_camera_id),
        int(raw._wrist_right_camera_id),
    )
    out: dict[str, dict] = {}
    axis_u = hole_axis / (np.linalg.norm(hole_axis) + 1e-12)
    along = float(np.dot(tip - hole, axis_u))
    tip_likely_occluded = along < -0.005
    for name, cam_id, frame in zip(CAMERA_NAMES, cam_ids, frames, strict=True):
        h, w = frame.shape[:2]
        cal = _camera_calibration(raw, cam_id, w, h)
        uv, depth, in_frame = cal.project(np.stack([tip, hole]))
        out[name] = {
            "tip_uv": uv[0].astype(np.float32),
            "hole_uv": uv[1].astype(np.float32),
            "tip_depth": float(depth[0]),
            "hole_depth": float(depth[1]),
            "tip_in_frame": bool(in_frame[0]),
            "hole_in_frame": bool(in_frame[1]),
            "tip_visible": bool(in_frame[0]) and not tip_likely_occluded,
            "hole_visible": bool(in_frame[1]),
            "along_hole_m": along,
        }
    return out


def _draw_overlay(frame: np.ndarray, label: dict, *, title: str, extra: str = "") -> np.ndarray:
    import cv2

    vis = np.asarray(frame, dtype=np.uint8).copy()
    if vis.max() <= 1:
        vis = (np.clip(vis, 0, 1) * 255).astype(np.uint8)
    tip = tuple(np.round(label["tip_uv"]).astype(int))
    hole = tuple(np.round(label["hole_uv"]).astype(int))
    tip_color = (0, 255, 0) if label["tip_visible"] else (0, 180, 255)
    hole_color = (0, 0, 255) if label["hole_visible"] else (255, 180, 0)
    if label["tip_in_frame"]:
        cv2.drawMarker(vis, tip, tip_color, cv2.MARKER_CROSS, 22, 2)
        cv2.circle(vis, tip, 10, tip_color, 2)
    if label["hole_in_frame"]:
        cv2.drawMarker(vis, hole, hole_color, cv2.MARKER_TILTED_CROSS, 22, 2)
        cv2.circle(vis, hole, 10, hole_color, 2)
    for pt, color, ok in (
        (tip, (0, 255, 0), label["tip_in_frame"]),
        (hole, (0, 0, 255), label["hole_in_frame"]),
    ):
        if not ok:
            continue
        overlay = vis.copy()
        cv2.circle(overlay, pt, 28, color, -1)
        vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
    cv2.putText(vis, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    line2 = (
        f"tip_vis={int(label['tip_visible'])} hole_vis={int(label['hole_visible'])} "
        f"along={label['along_hole_m']*1000:.0f}mm"
    )
    if extra:
        line2 = f"{line2} {extra}"
    cv2.putText(vis, line2, (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def _rate_limited_move(current: np.ndarray, target: np.ndarray, step_m: float) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(current, dtype=np.float64)
    n = float(np.linalg.norm(delta))
    if n <= step_m or n < 1e-9:
        return np.asarray(target, dtype=np.float64)
    return np.asarray(current, dtype=np.float64) + delta * (step_m / n)


def _world_separation(left_pos: np.ndarray, right_pos: np.ndarray) -> np.ndarray:
    sep = np.asarray(left_pos, dtype=np.float64) - np.asarray(right_pos, dtype=np.float64)
    sep[2] = 0.0
    n = float(np.linalg.norm(sep))
    if n < 1e-3:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return sep / n


def _build_scan_targets(
    *,
    left0: np.ndarray,
    tip0: np.ndarray,
    hole_axis: np.ndarray,
    left_sep: np.ndarray,
    separate_frames: int = 0,
    spiral_frames: int = 220,
    spiral_r_min_m: float = 0.005,
    spiral_r_max_m: float = 0.050,
    spiral_turns: float = 2.5,
    spiral_along_max_m: float = 0.045,
    symmetric_open_m: float = 0.025,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]]:
    """(phase, left_pos, tip_pos, left_quat_wobble, right_quat_wobble).

    Ascending spiral: radius grows 5->50mm while tip retreats along hole axis away from opening.
    """
    x, y, z = _orthonormal_basis(hole_axis)
    left_z0 = float(left0[2])
    tip_z0 = float(tip0[2])
    left_center = np.asarray(left0, dtype=np.float64).copy()
    tip_center = np.asarray(tip0, dtype=np.float64).copy()

    targets: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]] = []

    if separate_frames > 0 and symmetric_open_m > 0.0:
        left_end = left_center + left_sep * symmetric_open_m
        tip_end = tip_center - left_sep * symmetric_open_m
        for i in range(separate_frames):
            t = (i + 1) / separate_frames
            left = left_center + (left_end - left_center) * t
            left[2] = left_z0
            tip = tip_center + (tip_end - tip_center) * t
            tip[2] = tip_z0
            targets.append(("open", left, tip, None, None))
        left_center = left_end
        tip_center = tip_end

    wobble_axis = x * 0.6 + z * 0.4
    wobble_axis /= np.linalg.norm(wobble_axis) + 1e-12

    for i in range(spiral_frames):
        t = i / max(spiral_frames - 1, 1)
        r = spiral_r_min_m + (spiral_r_max_m - spiral_r_min_m) * t
        along = spiral_along_max_m * t  # tip retreats from hole along +hole_axis
        th = spiral_turns * 2.0 * np.pi * t
        c, s = np.cos(th), np.sin(th)
        left = left_center + x * (r * c) + y * (r * s) + z * (along * 0.15)
        tip = tip_center + x * (r * c) + y * (r * s) + z * along
        # Left: tiny wobble; right: clearer multi-axis tilt for wrist camera diversity.
        left_wobble = R.from_rotvec(wobble_axis * (0.018 * np.sin(th)))
        right_wobble = (
            R.from_rotvec(x * (0.14 * np.sin(th)))
            * R.from_rotvec(y * (0.11 * np.cos(0.7 * th)))
            * R.from_rotvec(z * (0.09 * np.sin(1.3 * th)))
        )
        targets.append(("spiral", left, tip, left_wobble, right_wobble))

    return targets


def _frames_u8(raw) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for fr in raw.render():
        arr = np.asarray(fr)
        if arr.dtype != np.uint8:
            arr = (
                (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                if arr.max() <= 1.0
                else arr.astype(np.uint8)
            )
        out.append(arr)
    return out


def _record_frame(
    raw,
    *,
    writers,
    raw_writers,
    records: list[dict],
    frame_idx: int,
    phase: str,
    hole_opening_offset_m: float,
    left_quat_ref: np.ndarray,
    right_quat_ref: np.ndarray,
) -> None:
    frames_u8 = _frames_u8(raw)
    labels = _project_tip_hole(raw, frames_u8, hole_opening_offset_m=hole_opening_offset_m)
    tip_w, socket_w, hole_w, dist = _insert_geometry(raw)
    hole_open = _hole_opening_world(socket_w, hole_w, hole_opening_offset_m)
    left = np.asarray(read_arm_action(raw, "left"), dtype=np.float64)
    right = np.asarray(read_arm_action(raw, "right"), dtype=np.float64)
    wrist_sep_mm = float(np.linalg.norm(left[0:3] - right[0:3]) * 1000.0)
    # peg tilt from vertical
    peg_body = int(raw._model.body("industreal_round_peg_8mm").id)
    peg_z = np.asarray(raw._data.xmat[peg_body]).reshape(3, 3)[:, 2]
    peg_tilt_deg = float(np.degrees(np.arccos(np.clip(abs(peg_z[2]), 0.0, 1.0))))
    records.append(
        {
            "frame": frame_idx,
            "phase": phase,
            "tip_world": tip_w.astype(np.float32),
            "socket_world": socket_w.astype(np.float32),
            "hole_opening_world": hole_open.astype(np.float32),
            "dist_m": float(dist),
            "wrist_sep_mm": wrist_sep_mm,
            "peg_tilt_deg": peg_tilt_deg,
            "cameras": labels,
        }
    )
    extra = f"sep={wrist_sep_mm:.0f}mm tilt={peg_tilt_deg:.0f}deg"
    for name, fr in zip(CAMERA_NAMES, frames_u8, strict=True):
        raw_writers[name].append_data(fr)
        writers[name].append_data(
            _draw_overlay(
                fr,
                labels[name],
                title=f"{name} {phase} #{frame_idx} G=tip R=hole",
                extra=extra,
            )
        )


def _step_dual(
    env,
    raw,
    *,
    left_quat_ref: np.ndarray,
    left_hand: np.ndarray,
    right_quat_ref: np.ndarray,
    right_hand: np.ndarray,
    left_pos_target: np.ndarray,
    right_tip_target: np.ndarray,
    tip_to_mocap: np.ndarray,
    left_quat_wobble: R | None = None,
    right_quat_wobble: R | None = None,
    left_step_m: float = 0.0025,
    right_step_m: float = 0.0025,
) -> np.ndarray:
    left = np.asarray(read_arm_action(raw, "left"), dtype=np.float64).copy()
    right = np.asarray(read_arm_action(raw, "right"), dtype=np.float64).copy()
    left[0:3] = _rate_limited_move(left[0:3], left_pos_target, left_step_m)
    right_mocap_tgt = np.asarray(right_tip_target, dtype=np.float64) + tip_to_mocap
    right[0:3] = _rate_limited_move(right[0:3], right_mocap_tgt, right_step_m)

    lq = left_quat_ref.copy()
    rq = right_quat_ref.copy()
    if left_quat_wobble is not None:
        lq = (R.from_quat(lq[[1, 2, 3, 0]], scalar_first=False) * left_quat_wobble).as_quat()
        lq = np.asarray([lq[3], lq[0], lq[1], lq[2]], dtype=np.float64)
    if right_quat_wobble is not None:
        rq = (R.from_quat(rq[[1, 2, 3, 0]], scalar_first=False) * right_quat_wobble).as_quat()
        rq = np.asarray([rq[3], rq[0], rq[1], rq[2]], dtype=np.float64)

    left[3:7] = lq
    left[7:23] = left_hand
    right[3:7] = rq
    right[7:23] = right_hand
    policy44 = dual_arm23_to_action44(left, right)
    action46 = rotvec_dual_arm_to_policy(np.asarray(policy44, dtype=np.float64).reshape(44))
    env.step(action46)
    tip_cur, _, _, _ = _insert_geometry(raw)
    right_cur = np.asarray(read_arm_action(raw, "right"), dtype=np.float64)
    return 0.9 * tip_to_mocap + 0.1 * (right_cur[0:3] - tip_cur)


def run_smoke(
    *,
    episode: int,
    seed: int,
    out_dir: Path,
    hole_opening_offset_m: float,
    symmetric_open_m: float = 0.0,
    open_frames: int = 30,
    spiral_frames: int = 220,
    spiral_r_min_m: float = 0.005,
    spiral_r_max_m: float = 0.050,
    spiral_turns: float = 2.5,
    spiral_along_max_m: float = 0.045,
) -> dict:
    entries = _manifest([episode])
    if not entries:
        raise RuntimeError(f"episode {episode} not in sidecar manifest")
    entry = entries[0]
    peg_lift_end = resolve_peg_lift_end_frame(entry, SIDECAR)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_assembly_env(seed=seed, randomize=False, render_mode="rgb_array")
    raw = env.unwrapped
    labeler = AssemblyContactLabeler(raw)
    writers = {name: imageio.get_writer(out_dir / f"{name}_overlay.mp4", fps=FPS) for name in CAMERA_NAMES}
    raw_writers = {name: imageio.get_writer(out_dir / f"{name}_raw.mp4", fps=FPS) for name in CAMERA_NAMES}
    records: list[dict] = []
    frame_idx = 0

    try:
        _, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
        demo_replay_to_pre_insert(
            env,
            raw,
            zarr_path=entry["zarr_path"],
            stop_frame=int(peg_lift_end),
            initial_state=initial_state,
            labeler=labeler,
        )
        left0 = np.asarray(read_arm_action(raw, "left"), dtype=np.float64).copy()
        right0 = np.asarray(read_arm_action(raw, "right"), dtype=np.float64).copy()
        fail = _approach(env, raw, labeler, left0, right0[7:23].copy())
        if fail:
            raise RuntimeError(f"approach failed: {fail}")

        left_now = np.asarray(read_arm_action(raw, "left"), dtype=np.float64).copy()
        right_now = np.asarray(read_arm_action(raw, "right"), dtype=np.float64).copy()
        left_quat_ref, left_hand = left_now[3:7].copy(), left_now[7:23].copy()
        right_quat_ref, right_hand = right_now[3:7].copy(), right_now[7:23].copy()
        tip0, socket, hole_axis, _ = _insert_geometry(raw)
        tip_to_mocap = right_now[0:3] - tip0
        left_sep = _world_separation(left_now[0:3], right_now[0:3])

        targets = _build_scan_targets(
            left0=left_now[0:3],
            tip0=tip0,
            hole_axis=hole_axis,
            left_sep=left_sep,
            separate_frames=open_frames if symmetric_open_m > 0.0 else 0,
            symmetric_open_m=symmetric_open_m,
            spiral_frames=spiral_frames,
            spiral_r_min_m=spiral_r_min_m,
            spiral_r_max_m=spiral_r_max_m,
            spiral_turns=spiral_turns,
            spiral_along_max_m=spiral_along_max_m,
        )
        print(
            f"targets={len(targets)} hole_offset={hole_opening_offset_m:.3f}m "
            f"spiral r={spiral_r_min_m*1000:.0f}->{spiral_r_max_m*1000:.0f}mm "
            f"along+={spiral_along_max_m*1000:.0f}mm turns={spiral_turns} fps={FPS}",
            flush=True,
        )

        for phase, left_tgt, tip_tgt, l_wob, r_wob in targets:
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
            _record_frame(
                raw,
                writers=writers,
                raw_writers=raw_writers,
                records=records,
                frame_idx=frame_idx,
                phase=phase,
                hole_opening_offset_m=hole_opening_offset_m,
                left_quat_ref=left_quat_ref,
                right_quat_ref=right_quat_ref,
            )
            if frame_idx % 40 == 0:
                ego = records[-1]["cameras"]["ego"]
                print(
                    f"  [{frame_idx}/{len(targets)}] {phase} "
                    f"tip_uv={ego['tip_uv']} hole_uv={ego['hole_uv']} "
                    f"sep={records[-1]['wrist_sep_mm']:.0f}mm tilt={records[-1]['peg_tilt_deg']:.0f}deg",
                    flush=True,
                )
            frame_idx += 1

        meta = {
            "episode_index": episode,
            "seed": seed,
            "zarr_path": str(entry["zarr_path"]),
            "num_frames": len(records),
            "fps": FPS,
            "hole_opening_offset_m": hole_opening_offset_m,
            "note": (
                "Ascending spiral: radius 5->50mm, tip retreats up to 45mm along hole axis "
                "away from opening; right wrist wobble stronger than left."
            ),
            "legend": "green=tip, red=hole_opening",
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        np.savez_compressed(
            out_dir / "labels.npz",
            frames=np.asarray([r["frame"] for r in records], dtype=np.int32),
            phases=np.asarray([r["phase"] for r in records], dtype=object),
            tip_world=np.stack([r["tip_world"] for r in records]),
            hole_opening_world=np.stack([r["hole_opening_world"] for r in records]),
            ego_tip_uv=np.stack([r["cameras"]["ego"]["tip_uv"] for r in records]),
            ego_hole_uv=np.stack([r["cameras"]["ego"]["hole_uv"] for r in records]),
            ego_tip_visible=np.asarray(
                [r["cameras"]["ego"]["tip_visible"] for r in records], dtype=bool
            ),
            wrist_sep_mm=np.asarray([r["wrist_sep_mm"] for r in records], dtype=np.float32),
            peg_tilt_deg=np.asarray([r["peg_tilt_deg"] for r in records], dtype=np.float32),
        )
        return {
            "ok": True,
            "num_frames": len(records),
            "out_dir": str(out_dir),
            "picks": [0, len(records) // 4, len(records) // 2, 3 * len(records) // 4, len(records) - 1],
        }
    finally:
        for w in writers.values():
            w.close()
        for w in raw_writers.values():
            w.close()
        env.close()


def _dump_stills(out_dir: Path, picks: list[int]) -> None:
    still_dir = out_dir / "stills"
    still_dir.mkdir(exist_ok=True)
    for name in CAMERA_NAMES:
        path = out_dir / f"{name}_overlay.mp4"
        if not path.exists():
            continue
        reader = imageio.get_reader(path)
        n = reader.count_frames()
        for i in picks:
            i = int(np.clip(i, 0, n - 1))
            imageio.imwrite(still_dir / f"{name}_{i:04d}.png", reader.get_data(i))
        reader.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/outputs/gaze_scan_smoke_ep00_v8"),
    )
    parser.add_argument(
        "--hole-opening-offset-m",
        type=float,
        default=DEFAULT_HOLE_OPENING_OFFSET_M,
        help="socket_site -> visual 孔口 along hole axis (m)",
    )
    parser.add_argument(
        "--symmetric-open-m",
        type=float,
        default=0.0,
        help="if >0, both arms open this much (m) apart before circling; 0=in-place",
    )
    parser.add_argument(
        "--open-frames",
        type=int,
        default=30,
        help="frames for symmetric open (only if symmetric-open-m > 0)",
    )
    parser.add_argument("--spiral-max-m", type=float, default=0.050, help="max spiral radius (m)")
    parser.add_argument("--spiral-min-m", type=float, default=0.005, help="start spiral radius (m)")
    parser.add_argument("--spiral-turns", type=float, default=2.5, help="spiral revolutions")
    parser.add_argument("--spiral-frames", type=int, default=220, help="frames for one spiral sweep")
    parser.add_argument(
        "--spiral-along-max-m",
        type=float,
        default=0.045,
        help="max tip retreat along hole axis away from opening (m)",
    )
    args = parser.parse_args()
    result = run_smoke(
        episode=args.episode,
        seed=args.seed,
        out_dir=args.out_dir,
        hole_opening_offset_m=args.hole_opening_offset_m,
        symmetric_open_m=args.symmetric_open_m,
        open_frames=args.open_frames,
        spiral_frames=args.spiral_frames,
        spiral_r_min_m=args.spiral_min_m,
        spiral_r_max_m=args.spiral_max_m,
        spiral_turns=args.spiral_turns,
        spiral_along_max_m=args.spiral_along_max_m,
    )
    _dump_stills(args.out_dir, result["picks"])
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
