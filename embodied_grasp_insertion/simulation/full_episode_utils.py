"""Shared helpers for FullEpisodeEnv demo replay and root selection."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEXJOCO_ROOT = PROJECT_ROOT.parent
REACH_ROOT = DEXJOCO_ROOT.parent / "reach_insert_rl"
LAI_ROOT = DEXJOCO_ROOT.parent / "lai"


def ensure_paths() -> None:
    for p in (
        str(PROJECT_ROOT),
        str(REACH_ROOT),
        str(DEXJOCO_ROOT),
        str(DEXJOCO_ROOT / "dexjoco"),
        str(LAI_ROOT),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)


ensure_paths()

from reach_insert_rl.env.full_env import FullEpisodeEnv  # noqa: E402
from reach_insert_rl.env.full_obs import current_action44, policy46_to_action44  # noqa: E402
from reach_insert_rl.env.handoff_env import load_manifest_entries  # noqa: E402
from interaction_retarget.sim.replay import zarr_action_to_policy46  # noqa: E402
from dexjoco.sim.envs.assembly_geometry import names_from_raw  # noqa: E402
from embodied_grasp_insertion.geometry.target_hole import (  # noqa: E402
    target_hole_from_raw,
)


WRIST_IDX = list(range(0, 6)) + list(range(22, 28))
RIGHT_FINGER_IDX = list(range(6, 22))
LEFT_FINGER_IDX = list(range(28, 44))
FINGER_IDX = RIGHT_FINGER_IDX + LEFT_FINGER_IDX


def geometry_family_id_from_env(env: FullEpisodeEnv) -> str:
    """Resolve family from underlying raw env (demo envs default round_8mm)."""
    raw = getattr(env, "_raw", None) or getattr(env, "_env", None)
    if raw is None:
        return "round_8mm"
    return names_from_raw(raw).family_id


def target_hole_info_from_env(env: FullEpisodeEnv) -> dict[str, Any]:
    raw = getattr(env, "_raw", None) or getattr(env, "_env", None)
    if raw is None:
        raise RuntimeError("env has no raw MuJoCo handle")
    return target_hole_from_raw(raw).to_dict()


@dataclass
class RootCandidate:
    episode_index: int
    frame: int
    phase: str
    reason: str
    tip_dist_m: float
    peg_ok: bool
    insert_ok: bool


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_full_env(
    episode_ids: list[int],
    *,
    sidecar_dir: Path | None = None,
    seed: int = 0,
) -> FullEpisodeEnv:
    from embodied_grasp_insertion.pilot.paths import assert_not_pilot_path_for_training

    if sidecar_dir is not None:
        assert_not_pilot_path_for_training(sidecar_dir, context="make_full_env.sidecar_dir")
    entries = load_manifest_entries(sidecar_dir, episode_indices=list(episode_ids))
    if not entries:
        raise RuntimeError(f"no manifest entries for episodes {episode_ids}")
    return FullEpisodeEnv(entries, sidecar_dir=sidecar_dir, seed=seed, use_force=True)


def load_action_rotvec(zarr_path: str | Path) -> np.ndarray:
    import zarr

    root = zarr.open(str(zarr_path), mode="r")
    if "data" in root and "action_rotvec" in root["data"]:
        return np.asarray(root["data"]["action_rotvec"][:], dtype=np.float64)
    # Fallback: convert stored actions.
    from interaction_retarget.io.zarr_io import load_zarr_episode

    actions, _, _ = load_zarr_episode(Path(zarr_path))
    out = []
    for a in actions:
        out.append(policy46_to_action44(zarr_action_to_policy46(a)))
    return np.asarray(out, dtype=np.float64)


def sync_python_after_demo_abs_step(env: FullEpisodeEnv) -> dict[str, Any]:
    """Update FullEpisodeEnv bookkeeping after an absolute demo env._env.step."""
    from reach_insert_rl.env.full_obs import privileged_full_features

    now44 = current_action44(env._raw)
    env._hold44 = now44.copy()
    env._t += 1
    outcome = env._labeler.compute(env._raw)
    feat = privileged_full_features(env._raw)
    if not outcome.peg_ok:
        env._peg_lost += 1
    else:
        env._peg_lost = 0
    # Mirror reward side effects that flip seen flags / prev geometry.
    success = env._is_success(outcome, feat)
    _ = env._shaped_reward(outcome, feat, success=success)
    peg_lost = env._peg_ok_seen and env._peg_lost >= env.peg_lost_abort
    timeout = env._t >= env.max_episode_steps
    terminated = bool(success or peg_lost)
    truncated = bool(timeout and not terminated)
    env._done = terminated or truncated
    return {
        "tray_ok": bool(outcome.tray_ok),
        "peg_ok": bool(outcome.peg_ok),
        "insert_ok": bool(outcome.insert_ok),
        "tip_dist_m": float(feat["tip_dist"]),
        "lat_err_m": float(feat["lat_err"]),
        "along_m": float(feat["along"]),
        "terminated": terminated,
        "truncated": truncated,
        "success": bool(success),
        "peg_lost_abort": bool(peg_lost),
        "geometry_family_id": geometry_family_id_from_env(env),
        "target_hole": target_hole_info_from_env(env),
    }


def replay_demo_one_step(env: FullEpisodeEnv) -> dict[str, Any]:
    assert env._actions is not None
    t = int(env._t)
    if t >= len(env._actions):
        raise RuntimeError("demo actions exhausted")
    if env._done:
        raise RuntimeError("episode already done")
    a = env._actions[t]
    try:
        a46 = zarr_action_to_policy46(a)
        env._env.step(a46)
    except Exception:
        from interaction_retarget.sim.replay import raw_flat_to_dict

        env._raw.step(raw_flat_to_dict(a))
    return sync_python_after_demo_abs_step(env)


def replay_demo_to_frame(env: FullEpisodeEnv, frame: int) -> list[dict[str, Any]]:
    """Replay absolute demo actions until env._t == frame."""
    infos = []
    target = int(frame)
    if target < int(env._t):
        raise RuntimeError(f"cannot rewind without restore: at {env._t}, want {target}")
    while int(env._t) < target:
        infos.append(replay_demo_one_step(env))
        if env._done and int(env._t) < target:
            raise RuntimeError(
                f"episode terminated at t={env._t} before target frame {target}"
            )
    return infos


def select_roots_for_episode(
    env: FullEpisodeEnv,
    *,
    early_offset: int = 5,
    transport_tip_min_m: float = 0.08,
    preinsert_tip_max_m: float = 0.06,
    max_scan_frames: int | None = None,
) -> list[RootCandidate]:
    """Generic root selection from demo replay (no episode-specific frame hardcodes)."""
    assert env._actions is not None and env._spec is not None
    ep = int(env._spec.episode_index)
    n = len(env._actions)
    limit = n if max_scan_frames is None else min(n, int(max_scan_frames))

    first_peg_ok = None
    transport_frame = None
    preinsert_frame = None
    history: list[dict[str, Any]] = []

    while int(env._t) < limit and not env._done:
        info = replay_demo_one_step(env)
        history.append({"frame": int(env._t), **info})
        f = int(env._t)
        if first_peg_ok is None and info["peg_ok"] and not info["insert_ok"]:
            first_peg_ok = f
        if (
            transport_frame is None
            and info["peg_ok"]
            and not info["insert_ok"]
            and first_peg_ok is not None
            and f >= first_peg_ok + 20
            and float(info["tip_dist_m"]) >= float(transport_tip_min_m)
        ):
            transport_frame = f
        if (
            preinsert_frame is None
            and info["peg_ok"]
            and not info["insert_ok"]
            and first_peg_ok is not None
            and float(info["tip_dist_m"]) <= float(preinsert_tip_max_m)
            and f >= first_peg_ok + 10
        ):
            preinsert_frame = f

    roots: list[RootCandidate] = []
    if first_peg_ok is not None:
        early = min(first_peg_ok + int(early_offset), history[-1]["frame"])
        # Ensure early still peg_ok.
        early_info = next(h for h in history if h["frame"] == early)
        if early_info["peg_ok"] and not early_info["insert_ok"]:
            roots.append(
                RootCandidate(
                    episode_index=ep,
                    frame=int(early),
                    phase="early_grasp",
                    reason=(
                        f"first peg_ok at {first_peg_ok} + offset {early_offset}; "
                        "peg grasped, not inserted"
                    ),
                    tip_dist_m=float(early_info["tip_dist_m"]),
                    peg_ok=True,
                    insert_ok=False,
                )
            )
    if transport_frame is not None:
        info = next(h for h in history if h["frame"] == transport_frame)
        roots.append(
            RootCandidate(
                episode_index=ep,
                frame=int(transport_frame),
                phase="transport",
                reason=(
                    f"peg_ok, tip_dist>={transport_tip_min_m}, "
                    f">=20 steps after first peg_ok"
                ),
                tip_dist_m=float(info["tip_dist_m"]),
                peg_ok=True,
                insert_ok=False,
            )
        )
    if preinsert_frame is not None:
        info = next(h for h in history if h["frame"] == preinsert_frame)
        # Avoid duplicate frame.
        if all(r.frame != preinsert_frame for r in roots):
            roots.append(
                RootCandidate(
                    episode_index=ep,
                    frame=int(preinsert_frame),
                    phase="pre_insert",
                    reason=(
                        f"peg_ok, tip_dist<={preinsert_tip_max_m}, not insert_ok"
                    ),
                    tip_dist_m=float(info["tip_dist_m"]),
                    peg_ok=True,
                    insert_ok=False,
                )
            )
    return roots


def abs44_from_demo(env: FullEpisodeEnv, frame: int) -> np.ndarray:
    assert env._actions is not None
    if frame < 0 or frame >= len(env._actions):
        raise IndexError(frame)
    return policy46_to_action44(zarr_action_to_policy46(env._actions[frame]))


def build_wrist_sequence(
    *,
    source: str,
    horizon: int,
    mild_transport_delta: np.ndarray,
    profile: str = "constant",
) -> np.ndarray:
    """Return (horizon, 44) deltas with only wrist components possibly nonzero.

    Profiles (task-agnostic, fixed, reproducible; no socket servo):
    - constant: same delta every step (legacy mild_transport)
    - shake: alternate +/- delta each step (net~0, inertial load)
    - impulse_hold: apply delta for first 2 steps, then zeros
    - go_return: +delta for first half, -delta for second half
    """
    out = np.zeros((int(horizon), 44), dtype=np.float64)
    if source == "hold":
        return out
    if source != "mild_transport":
        raise ValueError(f"unknown wrist source: {source}")
    d = np.asarray(mild_transport_delta, dtype=np.float64).reshape(44).copy()
    d[FINGER_IDX] = 0.0
    profile = str(profile or "constant")
    if profile == "constant":
        for t in range(horizon):
            out[t] = d
        return out
    if profile == "shake":
        for t in range(horizon):
            out[t] = d if (t % 2 == 0) else -d
        return out
    if profile == "impulse_hold":
        n = min(2, horizon)
        for t in range(n):
            out[t] = d
        return out
    if profile == "go_return":
        mid = max(1, horizon // 2)
        for t in range(horizon):
            out[t] = d if t < mid else -d
        return out
    raise ValueError(f"unknown wrist profile: {profile}")


def build_finger_sequence(
    *,
    name: str,
    horizon: int,
    env: FullEpisodeEnv,
    root_frame: int,
    mild_close_delta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (horizon, 44) deltas; wrist components left 0 (filled later)."""
    out = np.zeros((int(horizon), 44), dtype=np.float64)
    scale = env._scale()

    if name == "hold_fingers":
        return out

    if name == "mild_close":
        # Allegro: increasing joint commands typically close the hand.
        out[:, RIGHT_FINGER_IDX] = float(mild_close_delta)
        return out

    if name == "mild_open":
        out[:, RIGHT_FINGER_IDX] = -float(mild_close_delta)
        return out

    if name == "shuffled_or_random_finger":
        amp = float(mild_close_delta)
        budget = float(np.sqrt(len(RIGHT_FINGER_IDX)) * abs(amp))
        vec = rng.normal(size=len(RIGHT_FINGER_IDX))
        n = float(np.linalg.norm(vec)) + 1e-12
        vec = vec / n * budget
        for t in range(horizon):
            out[t, RIGHT_FINGER_IDX] = vec
        return out

    if name == "demo_finger_replay":
        assert env._hold44 is not None and env._actions is not None
        # Build deltas toward demo absolute finger poses frame-by-frame from root.
        # Uses successive targets demo[root_frame + k] relative to evolving hold estimate.
        hold = env._hold44.copy()
        for k in range(horizon):
            demo_idx = root_frame + k
            if demo_idx >= len(env._actions):
                break
            demo_abs = abs44_from_demo(env, demo_idx)
            delta = (demo_abs - hold) / (scale + 1e-12)
            delta = np.clip(delta, -1.0, 1.0)
            finger_only = np.zeros(44, dtype=np.float64)
            finger_only[FINGER_IDX] = delta[FINGER_IDX]
            out[k] = finger_only
            # Predict hold update as env.step would (wrist held → only fingers move in this seq).
            hold = hold + finger_only * scale
        return out

    raise ValueError(f"unknown finger intervention: {name}")


def merge_wrist_finger(wrist_seq: np.ndarray, finger_seq: np.ndarray) -> np.ndarray:
    w = np.asarray(wrist_seq, dtype=np.float64)
    f = np.asarray(finger_seq, dtype=np.float64)
    if w.shape != f.shape:
        raise ValueError(f"seq shape mismatch {w.shape} vs {f.shape}")
    out = w.copy()
    out[:, FINGER_IDX] = f[:, FINGER_IDX]
    # Ensure wrist comes only from wrist_seq.
    out[:, WRIST_IDX] = w[:, WRIST_IDX]
    return out


def action_finger_stats(seq: np.ndarray) -> dict[str, float]:
    f = np.asarray(seq, dtype=np.float64)[:, FINGER_IDX]
    return {
        "finger_l2_mean": float(np.linalg.norm(f, axis=1).mean()),
        "finger_l2_max": float(np.linalg.norm(f, axis=1).max()),
        "finger_abs_max": float(np.abs(f).max()),
    }


def action_wrist_equal(a: np.ndarray, b: np.ndarray, *, atol: float = 0.0) -> bool:
    return bool(np.allclose(a[:, WRIST_IDX], b[:, WRIST_IDX], atol=atol, rtol=0.0))
