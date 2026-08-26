#!/usr/bin/env python3
"""Export force-aligned collect successes to LeRobot v3 + ForceVLA forces.parquet."""

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

from dexquery.data.force_label_store import (  # noqa: E402
    ForceLabelManifest,
    merge_episode_shards,
    utc_now_iso,
    write_episode_force_parquet,
    write_manifest,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402

PROMPT = (
    "Grasp the tray with the left hand and the peg with the right hand, "
    "then insert the peg into the hole."
)
CAMERA_NAMES = ("ego", "wrist_left", "wrist_right")
FPS = 30
IMAGE_SHAPE = (640, 640, 3)


def _discover_successes(root: Path) -> list[Path]:
    rx = re.compile(r"episode_\d+_success")
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


def _export_episode(
    dataset: LeRobotDataset,
    ep_dir: Path,
    *,
    episode_index: int,
    global_index_start: int,
    label_dir: Path,
) -> dict:
    traj = np.load(ep_dir / "trajectory.npz", allow_pickle=True)
    states = np.asarray(traj["observation_state"], dtype=np.float32)
    actions = np.asarray(traj["action"], dtype=np.float32)
    n = int(states.shape[0])
    if actions.shape[0] != n:
        raise ValueError(f"{ep_dir}: state/action mismatch {states.shape[0]} vs {actions.shape[0]}")
    for key in ("wrist_ft_right", "wrist_ft_left", "right_finger_force", "left_finger_force"):
        if key not in traj.files:
            raise KeyError(f"{ep_dir}: missing force key {key}")

    wrist_r = np.asarray(traj["wrist_ft_right"], dtype=np.float32)
    wrist_l = np.asarray(traj["wrist_ft_left"], dtype=np.float32)
    finger_r = np.asarray(traj["right_finger_force"], dtype=np.float32)
    finger_l = np.asarray(traj["left_finger_force"], dtype=np.float32)
    insert_ok = np.asarray(traj["insert_ok"], dtype=np.float32) if "insert_ok" in traj.files else np.zeros(n, np.float32)

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

    global_index = np.arange(global_index_start, global_index_start + n, dtype=np.int64)
    frame_index = np.arange(n, dtype=np.int64)
    write_episode_force_parquet(
        label_dir,
        episode_index,
        global_index=global_index,
        frame_index=frame_index,
        right_finger_force=finger_r,
        left_finger_force=finger_l,
        wrist_ft_right=wrist_r,
        wrist_ft_left=wrist_l,
        insert_ok=insert_ok,
    )

    meta_path = ep_dir / "meta.json"
    src_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return {
        "input_dir": str(ep_dir),
        "episode_index": episode_index,
        "num_frames": n,
        "global_index_start": int(global_index_start),
        "seed": src_meta.get("seed"),
        "force_aligned": bool(src_meta.get("force_aligned", True)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/outputs/pi05_hybrid_insert_collect_raw_force"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/datasets/bimanual_assembly_force_smoke"),
    )
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()

    episodes = _discover_successes(args.raw_root)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise SystemExit(f"no episode_*_success under {args.raw_root}")

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    label_dir = args.out_dir / "force_labels"

    dataset = LeRobotDataset.create(
        repo_id="bimanual_assembly_force_smoke",
        fps=FPS,
        features=_build_features(),
        root=args.out_dir,
        image_writer_threads=4,
        streaming_encoding=True,
        encoder_queue_maxsize=0,
    )

    results = []
    global_index = 0
    for i, ep_dir in enumerate(episodes):
        print(f"[export] {i + 1}/{len(episodes)} {ep_dir.name}", flush=True)
        row = _export_episode(
            dataset,
            ep_dir,
            episode_index=i,
            global_index_start=global_index,
            label_dir=label_dir,
        )
        results.append(row)
        global_index += int(row["num_frames"])
        print(json.dumps(row, ensure_ascii=False), flush=True)

    merged = merge_episode_shards(label_dir)
    write_manifest(
        label_dir,
        ForceLabelManifest(
            task="bimanual_assembly",
            source_dataset=str(args.raw_root),
            created_at=utc_now_iso(),
            num_frames=global_index,
            num_episodes=len(results),
            columns=[
                "index",
                "episode_index",
                "frame_index",
                "right_finger_force",
                "left_finger_force",
                "wrist_ft_right",
                "wrist_ft_left",
                "insert_ok",
            ],
            label_file=str(merged) if merged is not None else "",
            seed_base=-1,
            randomize=False,
            notes="Force-aligned pi05+hybrid insert collect; sidecar from trajectory.npz (not replay).",
        ),
    )

    summary = {
        "dataset": "bimanual_assembly_force_smoke",
        "lerobot_version": "v3",
        "prompt": PROMPT,
        "n_episodes": len(results),
        "n_frames_total": global_index,
        "raw_root": str(args.raw_root),
        "out_dir": str(args.out_dir),
        "force_labels": str(merged) if merged is not None else "",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "episodes": results,
    }
    (args.out_dir / "meta" / "force_smoke_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"done: {args.out_dir} episodes={len(results)} frames={global_index}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
