#!/usr/bin/env python3
"""Eval PoseInsert: demo replay to pre-insert -> PoseInsert insert."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import imageio

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEXJOJO_ROOT = _REPO_ROOT / "dexjoco"
if str(_DEXJOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXJOJO_ROOT))

from dexjoco_openpi_client.eval_dexjoco_openpi import _append_video_frames
from interaction_retarget.constants import TASK_ID, default_sidecar_dir
from interaction_retarget.sim.replay import make_assembly_env
from interaction_retarget.sim.video import EVAL_FPS, EVAL_MAX_FRAMES
from interaction_retarget.skill_replay.insert import InsertReport, run_pose_insert_phase
from pose_insert.paths import default_eval_video_path, default_poseinsert_data_dir
from pose_insert.pre_insert import resolve_peg_lift_end_frame

# pi0.5 eval parity (30fps, max 50s)
EVAL_VIDEO_FPS = EVAL_FPS
EVAL_VIDEO_MAX_FRAMES = EVAL_MAX_FRAMES
EGO_CAM = "ego"


@dataclass
class EvalRow:
    episode_index: int
    success: bool
    insert_ok: bool
    fail_reason: str
    video_path: str | None = None
    video_frames: int = 0
    video_capped: bool = False


class EgoVideoRecorder:
    """Same as pi0.5 eval: imageio writer + _append_video_frames, ego only, 50s cap."""

    def __init__(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path = out_path
        self._writer = imageio.get_writer(str(out_path), fps=EVAL_VIDEO_FPS)
        self._writers = {EGO_CAM: self._writer}
        self.frame_count = 0
        self.capped = False

    def append_obs(self, obs: dict) -> None:
        if self.frame_count >= EVAL_VIDEO_MAX_FRAMES:
            self.capped = True
            return
        _append_video_frames(self._writers, {EGO_CAM: obs[EGO_CAM]})
        self.frame_count += 1

    def close(self) -> Path:
        self._writer.close()
        return self.out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar-dir", type=Path, default=None)
    p.add_argument("--ep", type=int, default=35)
    p.add_argument("--all", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ckpt",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/poseinsert_sim/checkpoints/policy_last.ckpt"),
    )
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument(
        "--video",
        action="store_true",
        help="Record ego mp4 (single --ep only; default path on HDD)",
    )
    p.add_argument(
        "--video-out",
        type=Path,
        default=None,
        help="ego mp4 path (implies --video); default: /mnt/hdd/dexjoco/outputs/poseinsert_sim/videos/",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print peg_tip/socket/wrist trace every 30 insert steps",
    )
    p.add_argument(
        "--bimanual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dual-wrist mode (default). --no-bimanual freezes left arm (9D only).",
    )
    p.add_argument(
        "--insert-mode",
        choices=("auto", "policy", "action44", "wrist12", "zarr_oracle"),
        default="auto",
        help="auto: pick action44/wrist12/pose9 from ckpt; zarr_oracle: demo insert open-loop",
    )
    return p.parse_args()


def _manifest_entries(sidecar_dir: Path, episode_indices: list[int] | None) -> list[dict]:
    manifest = json.loads((sidecar_dir / "manifest.json").read_text(encoding="utf-8"))
    out: list[dict] = []
    for entry in manifest["episodes"]:
        ep = int(entry["episode_index"])
        if episode_indices is not None and ep not in episode_indices:
            continue
        timing = entry.get("timing", {})
        if timing.get("peg_lift_start") is None or timing.get("right_grasp_frame") is None:
            continue
        out.append(entry)
    return out


def _eval_one(
    entry: dict,
    *,
    args: argparse.Namespace,
    sidecar_dir: Path,
    data_root: Path,
    video_out: Path | None,
) -> EvalRow:
    ep_idx = int(entry["episode_index"])
    peg_lift_end_frame = resolve_peg_lift_end_frame(entry, sidecar_dir)

    env = make_assembly_env(seed=int(args.seed), randomize=False)
    raw = env.unwrapped
    recorder: EgoVideoRecorder | None = None
    video_cb: Callable[[dict], None] | None = None
    if video_out is not None:
        recorder = EgoVideoRecorder(video_out)
        video_cb = recorder.append_obs

    ckpt_path = args.ckpt.expanduser() if args.insert_mode != "zarr_oracle" else None
    report: InsertReport | None = None
    try:
        report = run_pose_insert_phase(
            env,
            raw,
            reach_mode="demo_replay",
            max_steps=int(args.max_steps),
            poseinsert_ckpt=ckpt_path,
            poseinsert_data_root=data_root,
            manifest_entry=entry,
            sidecar_dir=sidecar_dir,
            peg_lift_end_frame=peg_lift_end_frame,
            video_cb=video_cb,
            debug=bool(args.debug),
            bimanual=bool(args.bimanual),
            insert_mode=str(args.insert_mode),
        )
    finally:
        saved_video = None
        if recorder is not None:
            saved_video = recorder.close()
        env.close()

    assert report is not None
    return EvalRow(
        episode_index=ep_idx,
        success=bool(report.success),
        insert_ok=bool(report.insert_ok),
        fail_reason=str(report.fail_reason),
        video_path=str(saved_video) if saved_video is not None else None,
        video_frames=int(recorder.frame_count) if recorder is not None else 0,
        video_capped=bool(recorder.capped) if recorder is not None else False,
    )


def main() -> int:
    args = _parse_args()
    sidecar_dir = args.sidecar_dir if args.sidecar_dir is not None else default_sidecar_dir(TASK_ID)
    data_root = args.data_root if args.data_root is not None else default_poseinsert_data_dir(TASK_ID)
    ckpt = args.ckpt.expanduser()
    if args.insert_mode != "zarr_oracle" and not ckpt.is_file():
        print(f"checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    record_video = bool(args.video or args.video_out is not None)
    if record_video and args.all:
        print("--video/--video-out only supported with single --ep (not --all)", file=sys.stderr)
        return 1

    eps = None if args.all else [int(args.ep)]
    entries = _manifest_entries(sidecar_dir, eps)
    if not entries:
        print("no entries", file=sys.stderr)
        return 1

    passed = 0
    rows: list[EvalRow] = []
    for entry in entries:
        ep = int(entry["episode_index"])
        video_out = None
        if record_video:
            video_out = (
                args.video_out.expanduser()
                if args.video_out is not None
                else default_eval_video_path(ep, seed=int(args.seed))
            )
        row = _eval_one(entry, args=args, sidecar_dir=sidecar_dir, data_root=data_root, video_out=video_out)
        rows.append(row)
        msg = f"ep{row.episode_index} ok={row.success} insert_ok={row.insert_ok} reason={row.fail_reason}"
        if row.video_path:
            dur_s = row.video_frames / EVAL_VIDEO_FPS
            cap_note = " capped" if row.video_capped else ""
            msg += f" video={row.video_path} frames={row.video_frames} ({dur_s:.1f}s{cap_note})"
        print(msg)
        if row.success:
            passed += 1

    out = data_root / "eval_summary.json"
    out.write_text(
        json.dumps({"passed": passed, "total": len(rows), "episodes": [asdict(r) for r in rows]}, indent=2),
        encoding="utf-8",
    )
    print(f"{passed}/{len(rows)} -> {out}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
