#!/usr/bin/env python3
"""Replay demo to peg_lift_end (handoff) and write ego mp4. No policy / no insert."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO = Path(__file__).resolve().parents[1]
for path in (_REPO, _REPO / "dexjoco"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from interaction_retarget.constants import default_sidecar_dir
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.replay import make_assembly_env, rotvec_dual_arm_to_policy
from interaction_retarget.sim.settle import read_arm_action
from interaction_retarget.sim.video import DexEnvVideoRecorder
from interaction_retarget.skill_replay.insert import (
    demo_replay_to_pre_insert,
    dual_arm23_to_action44,
)
from pose_insert.pre_insert import resolve_peg_lift_end_frame


def _csv_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one episode index")
    return values


def _manifest_entries(sidecar_dir: Path, episodes: list[int]) -> list[dict]:
    manifest = json.loads((sidecar_dir / "manifest.json").read_text(encoding="utf-8"))
    by_ep = {int(e["episode_index"]): e for e in manifest["episodes"]}
    missing = sorted(set(episodes) - set(by_ep))
    if missing:
        raise KeyError(f"episodes missing from manifest: {missing}")
    return [by_ep[ep] for ep in episodes]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=_csv_ints, required=True)
    p.add_argument("--output", type=Path, required=True, help="Output directory for ego mp4")
    p.add_argument("--sidecar-dir", type=Path, default=default_sidecar_dir("bimanual_assembly"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hold-frames", type=int, default=60, help="Hold handoff pose after replay")
    p.add_argument("--camera-index", type=int, default=0)
    args = p.parse_args()

    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    sidecar = args.sidecar_dir.expanduser().resolve()
    entries = _manifest_entries(sidecar, args.episodes)

    for entry in entries:
        ep = int(entry["episode_index"])
        peg_lift_end = int(resolve_peg_lift_end_frame(entry, sidecar))
        zarr_path = entry["zarr_path"]
        _, _, initial_state = load_zarr_episode(Path(zarr_path))
        env = make_assembly_env(seed=args.seed + ep, randomize=False)
        raw = env.unwrapped
        mp4 = out / f"ep{ep:02d}_handoff_ego.mp4"
        rec = DexEnvVideoRecorder(
            env,
            mp4,
            camera_index=int(args.camera_index),
            max_frames=int(peg_lift_end) + int(args.hold_frames) + 50,
        )

        def video_cb(_obs=None):
            rec.capture()

        demo_replay_to_pre_insert(
            env,
            raw,
            zarr_path=zarr_path,
            stop_frame=peg_lift_end,
            initial_state=initial_state,
            video_cb=video_cb,
        )
        hold44 = dual_arm23_to_action44(
            read_arm_action(raw, "left"),
            read_arm_action(raw, "right"),
        )
        hold46 = rotvec_dual_arm_to_policy(hold44).astype("float32")
        for _ in range(max(0, int(args.hold_frames))):
            env.step(hold46)
            rec.capture()
        rec.close()
        env.close()
        print(f"ep{ep:02d}: handoff frame={peg_lift_end} -> {mp4}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
