#!/usr/bin/env python3
"""Export human-demo insert segment (peg_lift_end+1 -> episode end) from LeRobot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import imageio
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO, _REPO / "dexjoco", _REPO / "scripts"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from interaction_retarget.constants import default_sidecar_dir
from interaction_retarget.grasp.lift_reference import (
    extract_demo_lift_reference,
    save_demo_lift_reference,
)
from interaction_retarget.skill_replay.library import SkillLibrary
from pose_insert.pre_insert import resolve_peg_lift_end_frame

SIDECAR = default_sidecar_dir("bimanual_assembly")
LEROBOT_ROOT = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")
PROMPT = (
    "Grasp the tray with the left hand and the peg with the right hand, "
    "then insert the peg into the hole."
)
CAMERA_NAMES = ("ego", "wrist_left", "wrist_right")
CAMERA_KEYS = (
    "observation.images.ego",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)
FPS = 30


def _prewarm_peg_lift_cache(entries: list[dict]) -> None:
    lib = SkillLibrary(SIDECAR)
    for entry in entries:
        ep = int(entry["episode_index"])
        path = lib._lift_cache_path(ep)
        if path.is_file():
            continue
        print(f"[cache] peg_lift_end ep={ep}", flush=True)
        ref = extract_demo_lift_reference(
            SIDECAR,
            episode_index=ep,
            pick_seed=0,
        )
        save_demo_lift_reference(ref, path)


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


def _frame_to_uint8_hwc(frame) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def export_episode(
    entry: dict,
    *,
    dataset,
    episode_meta: dict,
    out_root: Path,
    skip_existing: bool = False,
) -> dict:
    ep = int(entry["episode_index"])
    out_dir = out_root / f"episode_{ep:02d}"
    if skip_existing and (out_dir / "meta.json").is_file() and (out_dir / "trajectory.npz").is_file():
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        return {
            "episode_index": ep,
            "num_frames": int(meta.get("num_frames", 0)),
            "peg_lift_end_frame": int(meta.get("peg_lift_end_frame", -1)),
            "output_dir": str(out_dir),
            "skipped": True,
        }

    peg_lift_end = int(resolve_peg_lift_end_frame(entry, SIDECAR))
    ep_len = int(episode_meta["length"])
    start_local = peg_lift_end + 1
    if start_local >= ep_len:
        return {
            "episode_index": ep,
            "num_frames": 0,
            "fail_reason": "peg_lift_end_at_or_past_end",
        }

    from_idx = int(episode_meta["dataset_from_index"])
    end_local = ep_len
    global_start = from_idx + start_local
    global_end = from_idx + end_local

    out_dir = out_root / f"episode_{ep:02d}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writers = {
        name: imageio.get_writer(out_dir / f"{name}.mp4", fps=FPS)
        for name in CAMERA_NAMES
    }
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    frame_indices: list[int] = []

    try:
        for global_idx in range(global_start, global_end):
            row = dataset[global_idx]
            if int(row["episode_index"]) != ep:
                raise RuntimeError(
                    f"episode mismatch at global {global_idx}: "
                    f"expected {ep}, got {int(row['episode_index'])}"
                )
            for name, key, writer in zip(CAMERA_NAMES, CAMERA_KEYS, writers.values(), strict=True):
                writer.append_data(_frame_to_uint8_hwc(row[key]))
            states.append(np.asarray(row["observation.state"], dtype=np.float32).reshape(46))
            actions.append(np.asarray(row["action"], dtype=np.float32).reshape(44))
            frame_indices.append(int(row["frame_index"]))
    finally:
        for writer in writers.values():
            writer.close()

    n_frames = len(states)
    np.savez(
        out_dir / "trajectory.npz",
        observation_state=np.stack(states, axis=0),
        action=np.stack(actions, axis=0),
        frame_index=np.asarray(frame_indices, dtype=np.int64),
        fps=np.int32(FPS),
    )
    meta = {
        "episode_index": ep,
        "zarr_path": str(entry["zarr_path"]),
        "lerobot_root": str(LEROBOT_ROOT),
        "peg_lift_end_frame": peg_lift_end,
        "insert_start_frame": start_local,
        "insert_end_frame": end_local - 1,
        "num_frames": int(n_frames),
        "segment": "human_demo_insert_only",
        "data_source": "lerobot_aligned",
        "prompt": PROMPT,
        "observation_state_dim": 46,
        "action_dim": 44,
        "camera_names": list(CAMERA_NAMES),
        "fps": FPS,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "episode_index": ep,
        "num_frames": int(n_frames),
        "peg_lift_end_frame": peg_lift_end,
        "output_dir": str(out_dir),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/outputs/demo_insert_ft_raw"),
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        episodes = None
    else:
        episodes = args.episodes if args.episodes is not None else [0]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("bimanual_assembly", root=str(LEROBOT_ROOT))
    ep_meta_by_idx = {int(e["episode_index"]): e for e in dataset.meta.episodes}

    entries = _manifest(episodes)
    _prewarm_peg_lift_cache(entries)
    results = []
    for entry in entries:
        ep = int(entry["episode_index"])
        if ep not in ep_meta_by_idx:
            results.append(
                {
                    "episode_index": ep,
                    "num_frames": 0,
                    "fail_reason": "missing_in_lerobot",
                }
            )
            continue
        print(f"[export] ep={ep}", flush=True)
        row = export_episode(
            entry,
            dataset=dataset,
            episode_meta=ep_meta_by_idx[ep],
            out_root=args.out_dir,
            skip_existing=bool(args.skip_existing),
        )
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    total_frames = sum(int(r.get("num_frames", 0)) for r in results)
    summary = {
        "protocol": "human_demo_insert_segment",
        "segment": "peg_lift_end+1 -> episode_end",
        "data_source": "lerobot_aligned",
        "n_episodes": len(results),
        "n_frames_total": int(total_frames),
        "results": results,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
