#!/usr/bin/env python3
"""Privileged replay of dexjoco zarr demos -> OpenTrack Trajectory npz."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import zarr
from tqdm import tqdm

# dexjoco repo root (parent of dex_track_assembly/)
_DEXJOJO_ROOT = Path(__file__).resolve().parents[2]
_DEX_PKG = _DEXJOJO_ROOT / "dexjoco"
for p in (_DEXJOJO_ROOT, _DEX_PKG):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("MUJOCO_GL", "egl")

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state
from interaction_retarget.sim.replay import make_assembly_env, raw_flat_to_dict

from track_mj.paths import DEFAULT_MANIFEST, mocap_dir
from track_mj.utils.dataset.traj_class import Trajectory, TrajectoryData, TrajectoryInfo, TrajectoryModel

DEFAULT_HZ = 30.0


def load_zarr_episode(zarr_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    key = "action" if "action" in data else "action_rotvec"
    actions = np.asarray(data[key][:], dtype=np.float64)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    initial_state = None
    if "state" in data:
        states = np.asarray(data["state"][:], dtype=np.float64)
        start = 0
        for i in range(len(actions) - 1):
            if not np.array_equal(actions[i], actions[i + 1]):
                start = i
                break
        actions = actions[start:]
        initial_state = np.asarray(states[start], dtype=np.float64).ravel()
    return actions, initial_state


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ep = p.add_mutually_exclusive_group(required=True)
    ep.add_argument("--ep", type=int, help="single episode_index in manifest")
    ep.add_argument("--all", action="store_true", help="convert manifest episodes in [start, end]")
    p.add_argument("--start", type=int, default=0, help="first episode_index (inclusive, with --all)")
    p.add_argument("--end", type=int, default=None, help="last episode_index (inclusive, with --all)")
    p.add_argument("--skip-existing", action="store_true", help="skip if output npz already exists")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--segment",
        choices=("full", "grasp_lift", "insert"),
        default="full",
        help="full=entire zarr; grasp_lift=0..peg_lift_start+margin; insert=peg_lift..end",
    )
    p.add_argument("--out", type=Path, default=None, help="output path (single --ep only)")
    p.add_argument("--hz", type=float, default=DEFAULT_HZ)
    return p.parse_args()


def _frame_slice(entry: dict, segment: str, n: int) -> slice:
    timing = entry.get("timing") or {}
    if segment == "full":
        return slice(0, n)
    peg_lift = int(timing.get("peg_lift_start", max(n - 1, 0)))
    tray_lift = int(timing.get("tray_lift_start", 0))
    start = min(tray_lift, peg_lift)
    if segment == "grasp_lift":
        end = min(n, peg_lift + 120)
        return slice(0, max(end, start + 1))
    # insert
    return slice(max(peg_lift - 10, 0), n)


def _joint_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        names.append(name if name else f"joint_{i}")
    return names


def _trajectory_model(model: mujoco.MjModel) -> TrajectoryModel:
    return TrajectoryModel(
        njnt=model.njnt,
        jnt_type=np.array(model.jnt_type, dtype=np.int32),
        nbody=model.nbody,
        body_rootid=np.array(model.body_rootid, dtype=np.int32),
        body_weldid=np.array(model.body_weldid, dtype=np.int32),
        body_mocapid=np.array(model.body_mocapid, dtype=np.int32),
        body_pos=np.array(model.body_pos, dtype=np.float64),
        body_quat=np.array(model.body_quat, dtype=np.float64),
        body_ipos=np.array(model.body_ipos, dtype=np.float64),
        body_iquat=np.array(model.body_iquat, dtype=np.float64),
        nsite=model.nsite,
        site_bodyid=np.array(model.site_bodyid, dtype=np.int32),
        site_pos=np.array(model.site_pos, dtype=np.float64),
        site_quat=np.array(model.site_quat, dtype=np.float64),
    )


def _record_episode(
    actions: np.ndarray,
    *,
    seed: int,
    initial_state: np.ndarray | None,
    desc: str = "replay",
) -> tuple[np.ndarray, ...]:
    env = make_assembly_env(seed=int(seed), randomize=False)
    raw = env.unwrapped
    model = raw._model
    config = CONFIG_MAPPING["bimanual_assembly"]()
    n = len(actions)
    qpos = np.zeros((n, model.nq), dtype=np.float64)
    qvel = np.zeros((n, model.nv), dtype=np.float64)
    xpos = np.zeros((n, model.nbody, 3), dtype=np.float64)
    xquat = np.zeros((n, model.nbody, 4), dtype=np.float64)
    cvel = np.zeros((n, model.nbody, 6), dtype=np.float64)
    subtree_com = np.zeros((n, model.nbody, 3), dtype=np.float64)
    site_xpos = np.zeros((n, model.nsite, 3), dtype=np.float64)
    site_xmat = np.zeros((n, model.nsite, 9), dtype=np.float64)
    try:
        env.reset()
        if initial_state is not None and has_restorer("bimanual_assembly"):
            restore_initial_state(env, "bimanual_assembly", config, initial_state)
        for i, action in enumerate(tqdm(actions, desc=desc, leave=False, unit="frame")):
            raw.step(raw_flat_to_dict(action))
            data = raw._data
            qpos[i] = np.asarray(data.qpos, dtype=np.float64)
            qvel[i] = np.asarray(data.qvel, dtype=np.float64)
            xpos[i] = np.asarray(data.xpos, dtype=np.float64)
            xquat[i] = np.asarray(data.xquat, dtype=np.float64)
            cvel[i] = np.asarray(data.cvel, dtype=np.float64)
            subtree_com[i] = np.asarray(data.subtree_com, dtype=np.float64)
            site_xpos[i] = np.asarray(data.site_xpos, dtype=np.float64)
            site_xmat[i] = np.asarray(data.site_xmat, dtype=np.float64)
    finally:
        env.close()
    return qpos, qvel, xpos, xquat, cvel, subtree_com, site_xpos, site_xmat


def convert_episode(
    entry: dict,
    *,
    segment: str,
    seed: int,
    hz: float,
    out_path: Path,
) -> Path:
    ep = int(entry["episode_index"])
    actions, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    sl = _frame_slice(entry, segment, len(actions))
    actions = actions[sl]
    if len(actions) == 0:
        raise ValueError(f"empty action slice segment={segment} ep={ep}")

    qpos, qvel, xpos, xquat, cvel, subtree_com, site_xpos, site_xmat = _record_episode(
        actions,
        seed=int(seed),
        initial_state=initial_state,
        desc=f"ep{ep:03d} {segment}",
    )

    env = make_assembly_env(seed=int(seed), randomize=False)
    model = env.unwrapped._model
    env.close()

    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
        for i in range(model.nbody)
    ]
    site_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or f"site_{i}"
        for i in range(model.nsite)
    ]
    info = TrajectoryInfo(
        joint_names=_joint_names(model),
        model=_trajectory_model(model),
        frequency=float(hz),
        body_names=body_names,
        site_names=site_names,
        metadata={
            "episode_index": int(entry["episode_index"]),
            "segment": segment,
            "zarr_path": str(entry["zarr_path"]),
            "source": "dexjoco_zarr_replay",
        },
    )
    data = TrajectoryData(
        qpos=qpos,
        qvel=qvel,
        xpos=xpos,
        xquat=xquat,
        cvel=cvel,
        subtree_com=subtree_com,
        site_xpos=site_xpos,
        site_xmat=site_xmat,
        split_points=np.array([0, len(qpos)], dtype=np.int32),
    )
    traj = Trajectory(info=info, data=data)
    # Use sim-recorded qvel; OpenTrack recalc helpers assume single-root G1 layout.

    out_path.parent.mkdir(parents=True, exist_ok=True)
    traj.save(str(out_path))
    return out_path


def _select_entries(manifest: dict, args: argparse.Namespace) -> list[dict]:
    episodes = sorted(manifest["episodes"], key=lambda e: int(e["episode_index"]))
    if args.ep is not None:
        return [next(e for e in episodes if int(e["episode_index"]) == int(args.ep))]
    end = args.end
    if end is None:
        end = int(manifest.get("num_episodes", len(episodes))) - 1
    selected = [
        e for e in episodes if int(args.start) <= int(e["episode_index"]) <= int(end)
    ]
    if not selected:
        raise ValueError(f"no episodes in range [{args.start}, {end}]")
    return selected


def _default_out_path(ep: int, segment: str) -> Path:
    return mocap_dir() / f"ep{ep:03d}_{segment}.npz"


def main() -> int:
    args = _parse_args()
    if args.out is not None and args.ep is None:
        raise SystemExit("--out only valid with --ep")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = _select_entries(manifest, args)
    failures: list[str] = []

    for entry in tqdm(entries, desc=f"convert {args.segment}", unit="ep"):
        ep = int(entry["episode_index"])
        out = args.out if args.out is not None else _default_out_path(ep, args.segment)
        if args.skip_existing and out.exists():
            tqdm.write(f"skip ep{ep:03d} (exists)")
            continue
        try:
            path = convert_episode(
                entry,
                segment=str(args.segment),
                seed=int(args.seed),
                hz=float(args.hz),
                out_path=out,
            )
            loaded = Trajectory.load(str(path), backend=np)
            tqdm.write(
                f"saved {path.name} | frames={loaded.data.qpos.shape[0]} "
                f"nq={loaded.data.qpos.shape[1]} nv={loaded.data.qvel.shape[1]}"
            )
        except Exception as exc:
            failures.append(f"ep{ep:03d}: {exc}")
            tqdm.write(f"FAILED ep{ep:03d}: {exc}")

    if failures:
        print(f"\n{len(failures)}/{len(entries)} failed:", file=sys.stderr)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        return 1
    print(f"done: {len(entries)} episode(s), segment={args.segment}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
