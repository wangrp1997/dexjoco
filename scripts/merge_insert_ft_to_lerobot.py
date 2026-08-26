#!/usr/bin/env python3
"""Merge demo + hybrid insert FT raw episodes into one LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import imageio
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

PROMPT = (
    "Grasp the tray with the left hand and the peg with the right hand, "
    "then insert the peg into the hole."
)
CAMERA_NAMES = ("ego", "wrist_left", "wrist_right")
FPS = 30
IMAGE_SHAPE = (640, 640, 3)


def _discover_episodes(root: Path, *, pattern: str) -> list[Path]:
    rx = re.compile(pattern)
    eps = [p for p in root.iterdir() if p.is_dir() and rx.fullmatch(p.name)]
    return sorted(eps, key=lambda p: p.name)


def _build_features() -> dict:
    features: dict = {}
    for cam in CAMERA_NAMES:
        features[f"{OBS_IMAGES}.{cam}"] = {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        }
    features[ACTION] = {"dtype": "float32", "shape": (44,)}
    features[OBS_STATE] = {"dtype": "float32", "shape": (46,)}
    return features


def _load_episode(ep_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    traj = np.load(ep_dir / "trajectory.npz")
    states = np.asarray(traj["observation_state"], dtype=np.float32)
    actions = np.asarray(traj["action"], dtype=np.float32)
    if actions.shape[0] != states.shape[0]:
        raise ValueError(
            f"{ep_dir}: state/action length mismatch "
            f"{states.shape[0]} vs {actions.shape[0]}"
        )
    return states, actions


def _export_episode(dataset: LeRobotDataset, ep_dir: Path, *, source: str) -> dict:
    states, actions = _load_episode(ep_dir)
    n = int(states.shape[0])
    readers = {cam: imageio.get_reader(ep_dir / f"{cam}.mp4") for cam in CAMERA_NAMES}
    try:
        for t in range(n):
            frame_data = {
                "task": PROMPT,
                ACTION: actions[t],
                OBS_STATE: states[t],
            }
            for cam in CAMERA_NAMES:
                frame = np.asarray(readers[cam].get_data(t))
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                frame_data[f"{OBS_IMAGES}.{cam}"] = frame
            dataset.add_frame(frame_data)
        dataset.save_episode()
    finally:
        for reader in readers.values():
            reader.close()
    meta_path = ep_dir / "meta.json"
    src_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return {
        "source": source,
        "input_dir": str(ep_dir),
        "num_frames": n,
        "segment": src_meta.get("segment", ""),
        "episode_index": src_meta.get("episode_index"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--demo-root",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/outputs/demo_insert_ft_raw"),
    )
    p.add_argument(
        "--hybrid-root",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/outputs/hybrid_insert_ft_raw"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/datasets/bimanual_assembly_insert_ft_mix"),
    )
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()

    demo_eps = _discover_episodes(args.demo_root, pattern=r"episode_\d{2}")
    hybrid_eps = _discover_episodes(args.hybrid_root, pattern=r"episode_\d{2}_success")
    episodes = [("demo", ep) for ep in demo_eps] + [("hybrid", ep) for ep in hybrid_eps]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    dataset = LeRobotDataset.create(
        repo_id="bimanual_assembly_insert_ft_mix",
        fps=FPS,
        features=_build_features(),
        root=args.out_dir,
        image_writer_threads=4,
        streaming_encoding=True,
        encoder_queue_maxsize=0,
    )

    results = []
    total_frames = 0
    for i, (source, ep_dir) in enumerate(episodes):
        print(f"[lerobot] {i + 1}/{len(episodes)} {source} {ep_dir.name}", flush=True)
        row = _export_episode(dataset, ep_dir, source=source)
        results.append(row)
        total_frames += int(row["num_frames"])
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        "dataset": "bimanual_assembly_insert_ft_mix",
        "lerobot_version": "v3",
        "prompt": PROMPT,
        "n_demo_episodes": sum(1 for r in results if r["source"] == "demo"),
        "n_hybrid_episodes": sum(1 for r in results if r["source"] == "hybrid"),
        "n_episodes": len(results),
        "n_frames_total": total_frames,
        "demo_root": str(args.demo_root),
        "hybrid_root": str(args.hybrid_root),
        "out_dir": str(args.out_dir),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "episodes": results,
    }
    (args.out_dir / "meta" / "insert_ft_mix_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        f"done: {args.out_dir} episodes={len(results)} frames={total_frames}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
