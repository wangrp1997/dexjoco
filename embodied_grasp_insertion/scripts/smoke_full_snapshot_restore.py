#!/usr/bin/env python3
"""Snapshot restore determinism gate for FullEpisodeEnv (P0-C0)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.physics.grasp_metrics import object_in_hand_pose  # noqa: E402
from embodied_grasp_insertion.simulation.full_episode_snapshot import (  # noqa: E402
    FullEpisodeSnapshot,
)
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    make_full_env,
    replay_demo_to_frame,
    select_roots_for_episode,
)


def _pack_step(env, obs, reward, terminated, truncated, info) -> dict[str, Any]:
    raw = env._raw
    data = raw._data
    outcome = env._labeler.compute(raw)
    o2h = object_in_hand_pose(raw)
    return {
        "t": int(env._t),
        "qpos": np.asarray(data.qpos, dtype=np.float64).copy(),
        "qvel": np.asarray(data.qvel, dtype=np.float64).copy(),
        "obs": np.asarray(obs, dtype=np.float64).copy(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "tray_ok": bool(outcome.tray_ok),
        "peg_ok": bool(outcome.peg_ok),
        "insert_ok": bool(outcome.insert_ok),
        "peg_contact_count": int(outcome.peg_contact_count),
        "tray_contact_count": int(outcome.tray_contact_count),
        "object_pos": o2h.translation.copy(),
        "object_rotvec": o2h.rotvec.copy(),
        "hold44": np.asarray(env._hold44, dtype=np.float64).copy(),
        "info_peg_lost": bool(info.get("peg_lost", False)),
    }


def rollout(env, actions: np.ndarray) -> list[dict[str, Any]]:
    steps = []
    for a in actions:
        if env._done:
            break
        obs, reward, terminated, truncated, info = env.step(a)
        steps.append(_pack_step(env, obs, reward, terminated, truncated, info))
        if terminated or truncated:
            break
    return steps


def compare_rollouts(
    a: list[dict[str, Any]],
    b: list[dict[str, Any]],
    *,
    qpos_atol: float,
    qvel_atol: float,
    obs_atol: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "length_a": len(a),
        "length_b": len(b),
        "length_equal": len(a) == len(b),
        "passed": False,
        "max_abs": {},
        "failures": [],
    }
    if len(a) != len(b):
        out["failures"].append("length_mismatch")
        return out
    if len(a) == 0:
        out["failures"].append("empty_rollout")
        return out

    keys_bool = ["terminated", "truncated", "tray_ok", "peg_ok", "insert_ok", "info_peg_lost"]
    max_abs = {
        "qpos": 0.0,
        "qvel": 0.0,
        "obs": 0.0,
        "reward": 0.0,
        "object_pos": 0.0,
        "object_rotvec": 0.0,
        "hold44": 0.0,
        "peg_contact_count": 0.0,
    }
    for i, (sa, sb) in enumerate(zip(a, b)):
        for k in keys_bool:
            if bool(sa[k]) != bool(sb[k]):
                out["failures"].append(f"step{i}_{k}_mismatch:{sa[k]}!={sb[k]}")
        if int(sa["peg_contact_count"]) != int(sb["peg_contact_count"]):
            out["failures"].append(
                f"step{i}_peg_contact_count:{sa['peg_contact_count']}!={sb['peg_contact_count']}"
            )
        if int(sa["tray_contact_count"]) != int(sb["tray_contact_count"]):
            out["failures"].append(
                f"step{i}_tray_contact_count:{sa['tray_contact_count']}!={sb['tray_contact_count']}"
            )
        for key, atol in (
            ("qpos", qpos_atol),
            ("qvel", qvel_atol),
            ("obs", obs_atol),
            ("object_pos", qpos_atol),
            ("object_rotvec", qpos_atol),
            ("hold44", qpos_atol),
        ):
            err = float(np.max(np.abs(sa[key] - sb[key])))
            max_abs[key] = max(max_abs[key], err)
            if err > atol:
                out["failures"].append(f"step{i}_{key}_err={err}")
        rerr = abs(float(sa["reward"]) - float(sb["reward"]))
        max_abs["reward"] = max(max_abs["reward"], rerr)
        if rerr > 1e-7:
            out["failures"].append(f"step{i}_reward_err={rerr}")

    out["max_abs"] = max_abs
    out["passed"] = len(out["failures"]) == 0 and out["length_equal"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument(
        "--root-frames",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit frames; default: auto-select >=2 generic roots",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--sidecar-dir", type=str, default="/mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly")
    parser.add_argument("--qpos-atol", type=float, default=1e-8)
    parser.add_argument("--qvel-atol", type=float, default=1e-8)
    parser.add_argument("--obs-atol", type=float, default=1e-5)
    args = parser.parse_args()

    env = make_full_env([args.episode_id], sidecar_dir=Path(args.sidecar_dir), seed=args.seed)
    rng = np.random.default_rng(args.seed)
    results: list[dict[str, Any]] = []
    try:
        env.reset(episode_index=args.episode_id)
        if args.root_frames:
            frames = [int(x) for x in args.root_frames]
            root_meta = [
                {"frame": f, "phase": "user_specified", "reason": "cli --root-frames"} for f in frames
            ]
        else:
            roots = select_roots_for_episode(env)
            if len(roots) < 2:
                raise RuntimeError(
                    f"auto root selection found {len(roots)} roots; need >=2. "
                    "Pass --root-frames or choose another episode."
                )
            frames = [r.frame for r in roots[:2]]
            root_meta = [
                {
                    "frame": r.frame,
                    "phase": r.phase,
                    "reason": r.reason,
                    "tip_dist_m": r.tip_dist_m,
                }
                for r in roots[:2]
            ]

        for meta in root_meta:
            frame = int(meta["frame"])
            env.reset(episode_index=args.episode_id)
            replay_demo_to_frame(env, frame)
            if not env._labeler.compute(env._raw).peg_ok:
                raise RuntimeError(f"root frame {frame} has peg_ok=False")
            snap = FullEpisodeSnapshot.capture(env)
            # Immediate restore self-check before any rollout.
            qpos_before = np.asarray(env._raw._data.qpos, dtype=np.float64).copy()
            qvel_before = np.asarray(env._raw._data.qvel, dtype=np.float64).copy()
            obs_before = np.asarray(env._obs(), dtype=np.float64).copy()
            actions = rng.uniform(-0.2, 0.2, size=(args.horizon, 44)).astype(np.float64)
            roll_a = rollout(env, actions)
            obs0 = snap.restore(env)
            qpos_after = np.asarray(env._raw._data.qpos, dtype=np.float64)
            qvel_after = np.asarray(env._raw._data.qvel, dtype=np.float64)
            init_match = {
                "qpos_err": float(np.max(np.abs(qpos_after - snap.qpos))),
                "qvel_err": float(np.max(np.abs(qvel_after - snap.qvel))),
                "qpos_vs_pre_capture_err": float(np.max(np.abs(qpos_after - qpos_before))),
                "qvel_vs_pre_capture_err": float(np.max(np.abs(qvel_after - qvel_before))),
                "obs_vs_pre_capture_err": float(
                    np.max(np.abs(np.asarray(obs0, dtype=np.float64) - obs_before))
                ),
            }
            obs_check = env._obs()
            init_obs_err = float(np.max(np.abs(np.asarray(obs0) - np.asarray(obs_check))))
            roll_b = rollout(env, actions)
            cmp = compare_rollouts(
                roll_a,
                roll_b,
                qpos_atol=args.qpos_atol,
                qvel_atol=args.qvel_atol,
                obs_atol=args.obs_atol,
            )
            if init_match["qpos_err"] > args.qpos_atol or init_match["qvel_err"] > args.qvel_atol:
                cmp["passed"] = False
                cmp["failures"] = [
                    f"prestep_restore_mismatch:{init_match}",
                    *cmp.get("failures", []),
                ]
            results.append(
                {
                    "episode_index": args.episode_id,
                    "root": meta,
                    "horizon_requested": args.horizon,
                    "init_restore_match": init_match,
                    "init_obs_self_consistency_err": init_obs_err,
                    "snapshot_fields": {
                        "state_spec": snap.state_spec_name,
                        "qpos_dim": int(snap.qpos.size),
                        "qvel_dim": int(snap.qvel.size),
                        "has_act": snap.act is not None,
                        "ctrl_dim": int(snap.ctrl.size),
                        "python": [
                            "t",
                            "peg_lost",
                            "hold44",
                            "force_baseline",
                            "prev_tip",
                            "prev_lat",
                            "prev_along",
                            "tray_ok_seen",
                            "peg_ok_seen",
                            "done",
                            "labeler_rest_heights",
                        ],
                    },
                    "compare": cmp,
                }
            )
            if args.strict and not cmp["passed"]:
                break
    finally:
        env.close()

    passed = all(r["compare"]["passed"] for r in results) and len(results) >= 2
    payload = {
        "episode_id": args.episode_id,
        "n_roots_tested": len(results),
        "passed": passed,
        "strict": bool(args.strict),
        "results": [
            {
                **{k: v for k, v in r.items() if k != "compare"},
                "compare": {
                    **r["compare"],
                    # drop bulky arrays already summarized
                },
            }
            for r in results
        ],
    }
    # Make JSON serializable: compare already has no arrays.
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "n_roots": len(results), "output": str(out)}, ensure_ascii=False))
    if args.strict and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
