#!/usr/bin/env python3
"""Replay perfect demos and export interaction sidecars (phase-1 step 1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEXJOJO_ROOT = _REPO_ROOT / "dexjoco"
if str(_DEXJOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXJOJO_ROOT))

from interaction_retarget.constants import TASK_ID, default_sidecar_dir
from interaction_retarget.sim.grasp_timing import timing_warnings
from interaction_retarget.sim.replay import make_assembly_env, replay_episode
from interaction_retarget.io.sidecar import build_episode_sidecar, save_episode_sidecar
from interaction_retarget.io.zarr_io import discover_zarr_demos, load_zarr_episode


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr-root", type=Path, required=True, help="Root dir containing */replay.zarr demos")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Sidecar output dir (default: {default_sidecar_dir(TASK_ID)})",
    )
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--episodes", type=int, nargs="*", default=None, help="Optional episode index subset")
    p.add_argument("--randomize", action="store_true", help="Enable visual randomization (default: off)")
    p.add_argument("--no-trim-static", action="store_true")
    p.add_argument(
        "--vis",
        type=int,
        default=None,
        metavar="EP",
        help="MuJoCo viewer: replay episode EP and mark grasp/lift frames",
    )
    return p.parse_args()


def _write_manifest(out_dir: Path, entries: list[dict]) -> None:
    manifest = {
        "task": "bimanual_assembly",
        "num_episodes": len(entries),
        "episodes": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _run_vis(episode_index: int, zarr_path: Path, seed: int, initial_state, no_trim: bool) -> None:
    import mujoco.viewer

    from interaction_retarget.sim.replay import raw_flat_to_dict, replay_episode
    from interaction_retarget.io.sidecar import timing_from_trace

    actions, _, initial_state_loaded = load_zarr_episode(zarr_path, trim_static=not no_trim)
    if initial_state is None:
        initial_state = initial_state_loaded
    trace = replay_episode(actions, seed=seed, initial_state=initial_state, randomize=False)
    timing = timing_from_trace(trace)

    env = make_assembly_env(seed=seed, render_mode="human", randomize=False)
    raw = env.unwrapped
    from dexjoco.tasks import CONFIG_MAPPING
    from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

    config = CONFIG_MAPPING["bimanual_assembly"]()
    env.reset()
    if initial_state is not None and has_restorer("bimanual_assembly"):
        restore_initial_state(env, "bimanual_assembly", config, initial_state)

    mark_frames = {
        timing.left_grasp_frame: "L grasp",
        timing.right_grasp_frame: "R grasp",
        timing.tray_lift_start: "tray lift",
        timing.peg_lift_start: "peg lift",
    }

    print(f"[vis] episode {episode_index}: {zarr_path}")
    print(f"[vis] timing: {timing}")
    with mujoco.viewer.launch_passive(raw._model, raw._data) as viewer:
        for t, action in enumerate(actions):
            raw.step(raw_flat_to_dict(action))
            label = mark_frames.get(t)
            if label:
                print(f"  frame {t}: {label}")
            viewer.sync()
            if viewer.is_running():
                import time

                time.sleep(0.03)
            else:
                break
    env.close()


def main() -> None:
    args = _parse_args()
    demos = discover_zarr_demos(args.zarr_root)

    if args.vis is not None:
        if args.vis < 0 or args.vis >= len(demos):
            raise SystemExit(f"--vis episode index out of range: {args.vis} (found {len(demos)} demos)")
        zarr_path = demos[args.vis]
        actions, _, initial_state = load_zarr_episode(zarr_path, trim_static=not args.no_trim_static)
        _run_vis(args.vis, zarr_path, args.seed_base + args.vis, initial_state, args.no_trim_static)
        return

    out_dir = args.out_dir if args.out_dir is not None else default_sidecar_dir(TASK_ID)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.episodes) if args.episodes else None
    manifest_entries: list[dict] = []
    processed = 0

    for ep_idx, zarr_path in enumerate(tqdm(demos, desc="sidecar")):
        if selected is not None and ep_idx not in selected:
            continue
        if args.max_episodes is not None and processed >= args.max_episodes:
            break

        actions, action_key, initial_state = load_zarr_episode(
            zarr_path, trim_static=not args.no_trim_static
        )
        trace = replay_episode(
            actions,
            seed=args.seed_base + ep_idx,
            initial_state=initial_state,
            randomize=args.randomize,
        )
        sidecar = build_episode_sidecar(
            trace,
            episode_index=ep_idx,
            zarr_path=zarr_path,
            seed=args.seed_base,
        )
        npz_path = save_episode_sidecar(sidecar, out_dir)
        manifest_entries.append(
            {
                "episode_index": ep_idx,
                "zarr_path": str(zarr_path),
                "action_key": action_key,
                "npz_path": str(npz_path),
                "timing": {
                    "left_grasp_frame": sidecar.timing.left_grasp_frame,
                    "right_grasp_frame": sidecar.timing.right_grasp_frame,
                    "tray_lift_start": sidecar.timing.tray_lift_start,
                    "peg_lift_start": sidecar.timing.peg_lift_start,
                    "left_grasp_fallback": sidecar.timing.left_grasp_fallback,
                    "right_grasp_fallback": sidecar.timing.right_grasp_fallback,
                },
                "timing_warnings": timing_warnings(sidecar.timing),
                "has_tray": sidecar.tray is not None,
                "has_peg": sidecar.peg is not None,
            }
        )
        processed += 1

    _write_manifest(out_dir, manifest_entries)
    print(f"Wrote {processed} episode sidecars -> {out_dir}")


if __name__ == "__main__":
    main()
