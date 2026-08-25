from __future__ import annotations

import argparse
import json
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from dexjoco.tasks.bimanual_assembly.config import TaskConfig
from dexjoco.tasks.state_restorers import restore_initial_state
from interaction_retarget.io.zarr_io import discover_zarr_demos, load_zarr_episode
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.replay import zarr_action_to_policy46


STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def _capture(raw) -> dict[str, np.ndarray]:
    state = np.empty(mujoco.mj_stateSize(raw._model, STATE_SPEC), dtype=np.float64)
    mujoco.mj_getState(raw._model, raw._data, state, STATE_SPEC)
    return {
        "state": state,
        "table_pos": raw._model.body_pos[raw._table_body_id].copy(),
        "leg_sizes": raw._model.geom_size[raw._table_leg_geom_ids].copy(),
        "delta_h": np.asarray(raw.delta_h, dtype=np.float64),
    }


def build_root_bank(
    zarr_root: Path,
    manifest: Path | None,
    output: Path,
    episode_start: int,
    episode_count: int,
    offsets: tuple[int, ...],
    hold_steps: int,
    require_dual_grasp: bool = True,
) -> dict[str, int]:
    if episode_start < 0 or episode_count < 1:
        raise ValueError("episode_start must be non-negative and episode_count positive")
    offsets = tuple(sorted(set(int(offset) for offset in offsets)))
    if not offsets or offsets[0] <= 0:
        raise ValueError("offsets must contain positive step counts")

    if manifest is not None:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        demos = [
            (int(entry.get("episode_index", index)), Path(entry["zarr_path"]))
            for index, entry in enumerate(payload["episodes"])
        ]
    else:
        demos = list(enumerate(discover_zarr_demos(zarr_root)))
    demos = demos[episode_start : episode_start + episode_count]
    roots: list[dict[str, np.ndarray]] = []
    successful_demos = 0
    config = TaskConfig()

    for episode, zarr_path in demos:
        actions, _, initial_state = load_zarr_episode(zarr_path)
        if initial_state is None or not len(actions):
            continue
        env = config.get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            image_obs=False,
            randomize=False,
            randomize_dynamics=False,
            seed=episode,
        )
        raw = env.unwrapped
        raw.hz = 0
        raw._prime_rgb_array_renderer = lambda: None
        detector = AssemblyContactDetector(raw)
        ring: deque[tuple[int, dict[str, np.ndarray], bool]] = deque(
            maxlen=max(offsets) + 1
        )
        try:
            env.reset()
            restore_initial_state(env, "bimanual_assembly", config, initial_state)
            raw._success_started = False
            raw._success_counter = 0

            final_action = actions[-1]
            for frame in range(len(actions) + hold_steps):
                action = actions[frame] if frame < len(actions) else final_action
                _, _, terminated, truncated, info = env.step(
                    zarr_action_to_policy46(action)
                )
                contact = detector.compute(raw)
                both_contact = contact.tray_contact and contact.peg_contact
                ring.append((frame, _capture(raw), both_contact))
                if info.get("succeed", False):
                    successful_demos += 1
                    for offset in offsets:
                        if len(ring) <= offset:
                            continue
                        source_frame, root, root_both_contact = ring[-offset - 1]
                        if require_dual_grasp and not root_both_contact:
                            continue
                        reference = np.stack(
                            [
                                zarr_action_to_policy46(
                                    actions[future_frame]
                                    if future_frame < len(actions)
                                    else final_action
                                )
                                for future_frame in range(source_frame + 1, frame + 1)
                            ]
                        ).astype(np.float32)
                        reference = np.pad(
                            reference, ((0, max(offsets) - offset), (0, 0)), mode="edge"
                        )
                        root.update(
                            offset=np.asarray(offset, dtype=np.int32),
                            source_episode=np.asarray(episode, dtype=np.int32),
                            source_frame=np.asarray(source_frame, dtype=np.int32),
                            source_path=np.asarray(str(zarr_path)),
                            reference_action=reference,
                            reference_length=np.asarray(offset, dtype=np.int32),
                        )
                        roots.append(root)
                    break
                if terminated or truncated:
                    break
        finally:
            env.close()

    if not roots:
        raise RuntimeError("No native-success trajectories found; root bank not written")

    output.parent.mkdir(parents=True, exist_ok=True)
    keys = roots[0].keys()
    arrays = {key: np.stack([root[key] for root in roots]) for key in keys}
    np.savez_compressed(
        output,
        **arrays,
        state_spec=np.asarray(int(STATE_SPEC), dtype=np.int64),
        version=np.asarray(2, dtype=np.int32),
    )
    summary = {
        "demos_scanned": len(demos),
        "episode_slice": [episode_start, episode_start + len(demos)],
        "successful_demos": successful_demos,
        "roots": len(roots),
    }
    print(json.dumps({**summary, "output": str(output)}, indent=2))
    return summary


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Build native-success reverse-curriculum roots")
    command.add_argument(
        "--zarr-root",
        type=Path,
        default=Path(
            "/mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly"
        ),
    )
    command.add_argument(
        "--manifest",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly/manifest.json"),
    )
    command.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/hdd/dexjoco/outputs/bimanual_physics_rl/roots/near_insert_reference.npz"
        ),
    )
    command.add_argument("--episodes", type=int, default=20)
    command.add_argument("--episode-start", type=int, default=0)
    command.add_argument("--offsets", type=int, nargs="+", default=(30, 60, 120))
    command.add_argument("--hold-steps", type=int, default=200)
    command.add_argument("--allow-pregrasp", action="store_true")
    return command


if __name__ == "__main__":
    args = parser().parse_args()
    build_root_bank(
        args.zarr_root,
        args.manifest,
        args.output,
        args.episode_start,
        args.episodes,
        tuple(args.offsets),
        args.hold_steps,
        not args.allow_pregrasp,
    )
