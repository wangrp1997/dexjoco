#!/usr/bin/env python3
"""Collect force-aligned insert trajectories from original 100 human bimanual_assembly demos.

Protocol:
  1. Open-loop zarr replay through peg_lift_end (demo_replay_to_pre_insert) = handoff
  2. Open-loop replay original human demo insert actions from zarr (peg_lift+1 -> end)
  3. Record ego/wrist mp4 + trajectory.npz with FingerForceLabeler fields

Success criterion (same as pi05 800 collect): env info['succeed'] / is_success only.
Failures are not written to disk.
Outputs episode_demo_XX_success under pi05 raw dir (append; never touch episode_XXXX_*).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO = Path(__file__).resolve().parents[1]
_REACH_RL = Path("/home/wangrenpeng/reach_insert_rl")
for path in (_REPO, _REPO / "dexjoco", _REPO / "scripts", _REACH_RL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dexquery.data.finger_contact_forces import FingerForceLabeler
from hybrid_insert.assembly_contacts import AssemblyContactLabeler
from interaction_retarget.constants import default_sidecar_dir
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.replay import make_assembly_env, zarr_action_to_policy46
from interaction_retarget.skill_replay.insert import demo_replay_to_pre_insert
from pose_insert.pre_insert import resolve_peg_lift_end_frame
from reach_insert_rl.env.full_obs import policy46_to_action44

from export_hybrid_insert_ft import _SegmentRecorder

DEFAULT_OUT = Path("/mnt/hdd/dexjoco/outputs/pi05_hybrid_insert_collect_raw_force")
LEROBOT_ROOT = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")
SIDECAR = default_sidecar_dir("bimanual_assembly")
PROMPT = (
    "Grasp the tray with the left hand and the peg with the right hand, "
    "then insert the peg into the hole."
)
CAMERA_NAMES = ("ego", "wrist_left", "wrist_right")
FPS = 30


def _write_summary(path: Path, summary: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def _episode_success_dir(out_root: Path, ep: int) -> Path:
    return out_root / f"episode_demo_{ep:02d}_success"


def _csv_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one episode index")
    return values


def collect_episode(
    entry: dict,
    *,
    seed: int,
    out_root: Path,
    skip_existing: bool = False,
) -> dict:
    ep = int(entry["episode_index"])
    existing = _episode_success_dir(out_root, ep)
    if skip_existing and (existing / "trajectory.npz").is_file():
        meta_path = existing / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        return {
            "episode_index": ep,
            "success": True,
            "skipped": True,
            "num_frames": int(meta.get("num_frames", 0)),
            "output_dir": str(existing),
        }

    peg_lift_end = int(resolve_peg_lift_end_frame(entry, SIDECAR))
    env = make_assembly_env(seed=seed, randomize=False, render_mode="rgb_array")
    raw = env.unwrapped
    labeler = AssemblyContactLabeler(raw)
    force_labeler = FingerForceLabeler(raw)
    temp_dir = out_root / f"episode_demo_{ep:02d}_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    result: dict = {
        "episode_index": ep,
        "success": False,
        "fail_reason": "",
        "handoff_env_step": -1,
        "insert_frames": 0,
        "max_insert_streak": 0,
        "peg_lift_end_frame": peg_lift_end,
    }
    recorder: _SegmentRecorder | None = None

    try:
        actions, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
        labeler.reset_reference(raw)
        force_labeler.reset_reference(raw)
        demo_replay_to_pre_insert(
            env,
            raw,
            zarr_path=entry["zarr_path"],
            stop_frame=peg_lift_end,
            initial_state=initial_state,
            video_cb=None,
            labeler=labeler,
        )
        handoff_env_step = int(getattr(raw, "env_step", peg_lift_end))
        result["handoff_env_step"] = handoff_env_step
        result["replay_end_step"] = handoff_env_step

        if not bool(labeler.compute(raw).peg_ok):
            result["fail_reason"] = "peg_lost_before_insert"
            return result

        insert_start = peg_lift_end + 1
        if insert_start >= len(actions):
            result["fail_reason"] = "peg_lift_end_at_or_past_end"
            return result
        insert_actions = actions[insert_start:]

        temp_dir.mkdir(parents=True, exist_ok=True)
        recorder = _SegmentRecorder(raw, temp_dir, force_labeler=force_labeler)

        streak = 0
        max_streak = 0
        sim_success = False
        fail_reason = ""

        for demo_action in insert_actions:
            action46 = zarr_action_to_policy46(demo_action).astype(np.float32)
            action44 = policy46_to_action44(action46)
            _, _, terminated, truncated, info = env.step(action46)
            recorder.capture(action44=action44, phase="human_demo_replay", labeler=labeler)

            outcome = labeler.compute(raw)
            if bool(outcome.insert_ok):
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
            if bool(info.get("succeed", False)):
                sim_success = True
            if terminated or truncated:
                sim_success = bool(info.get("succeed", False))
                if not sim_success:
                    fail_reason = "env_done_no_success"
                break
        else:
            if not sim_success:
                fail_reason = "max_demo_frames"

        n_frames = recorder.close()
        result["insert_frames"] = int(n_frames)
        result["max_insert_streak"] = int(max_streak)
        result["success"] = bool(sim_success)
        result["fail_reason"] = "" if sim_success else fail_reason

        if not (sim_success and n_frames > 0):
            shutil.rmtree(temp_dir, ignore_errors=True)
            return result

        final_dir = _episode_success_dir(out_root, ep)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        recorder.save_npz(temp_dir / "trajectory.npz")
        meta = {
            "episode_index": ep,
            "zarr_path": str(entry["zarr_path"]),
            "lerobot_root": str(LEROBOT_ROOT),
            "peg_lift_end_frame": peg_lift_end,
            "insert_start_frame": insert_start,
            "insert_end_frame": insert_start + n_frames - 1,
            "success": True,
            "fail_reason": "",
            "max_insert_streak": int(max_streak),
            "num_frames": int(n_frames),
            "replay_end_step": int(handoff_env_step),
            "handoff_env_step": int(handoff_env_step),
            "segment": "human_demo_handoff_then_human_demo_open_loop_insert_forcevla",
            "data_source": "human_demo_ssd",
            "prompt": PROMPT,
            "observation_state_dim": 46,
            "action_dim": 44,
            "camera_names": list(CAMERA_NAMES),
            "fps": FPS,
            "force_fields": {
                "wrist_ft_right": 6,
                "wrist_ft_left": 6,
                "right_finger_force": 12,
                "left_finger_force": 12,
                "force": 36,
            },
            "force_layout": "ForceVLA both = [wrist_r(6), wrist_l(6), finger_r(12), finger_l(12)]",
            "force_aligned": True,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        (temp_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        temp_dir.rename(final_dir)
        result["output_dir"] = str(final_dir)
        return result
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=_csv_ints, default=None, help="Comma-separated demo indices")
    parser.add_argument("--all", action="store_true", help="Collect all manifest episodes (100 demos)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.all:
        episodes = None
    elif args.episodes is not None:
        episodes = args.episodes
    else:
        episodes = [0]

    args.output.mkdir(parents=True, exist_ok=True)
    entries = _manifest(episodes)
    summary_path = args.output / "summary_demo_human.json"
    prior_results: list[dict] = []
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        prior_results = list(prior.get("results") or [])

    t0 = time.time()
    results_by_ep: dict[int, dict] = {int(r["episode_index"]): r for r in prior_results}
    for entry in entries:
        ep = int(entry["episode_index"])
        print(f"[collect-demo] ep={ep}", flush=True)
        row = collect_episode(
            entry,
            seed=args.seed,
            out_root=args.output,
            skip_existing=bool(args.skip_existing),
        )
        results_by_ep[ep] = row
        status = "SKIP" if row.get("skipped") else ("OK" if row.get("success") else "FAIL")
        print(
            f"  {status} frames={row.get('insert_frames', 0)} "
            f"streak={row.get('max_insert_streak', 0)} "
            f"reason={row.get('fail_reason', '')} dir={row.get('output_dir', '')}",
            flush=True,
        )
        merged = sorted(results_by_ep.values(), key=lambda r: int(r["episode_index"]))
        summary = {
            "protocol": "human_demo_replay_to_peg_lift_end_then_human_demo_open_loop_insert_forcevla",
            "success_criterion": "env_info_succeed_same_as_pi05_collect",
            "output_layout": "episode_demo_XX_success only (failures not saved)",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "lerobot_root": str(LEROBOT_ROOT),
            "sidecar_dir": str(SIDECAR),
            "seed": int(args.seed),
            "force_aligned": True,
            "force_layout": "ForceVLA both = [wrist_r(6), wrist_l(6), finger_r(12), finger_l(12)]",
            "n_episodes": len(merged),
            "n_success": sum(1 for r in merged if r.get("success")),
            "elapsed_s": round(time.time() - t0, 1),
            "results": merged,
        }
        _write_summary(summary_path, summary)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(
        f"[collect-demo] done success={summary['n_success']}/{summary['n_episodes']} -> {summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
