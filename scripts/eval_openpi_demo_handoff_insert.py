#!/usr/bin/env python3
"""Evaluate OpenPI insertion after replaying each demo to its hybrid handoff state."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
for _proxy_variable in (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
):
    os.environ.pop(_proxy_variable, None)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEXJOCO_ROOT = _REPO_ROOT / "dexjoco"
for path in (_REPO_ROOT, _DEXJOCO_ROOT, _REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid_insert.assembly_contacts import AssemblyContactLabeler
from interaction_retarget.constants import default_sidecar_dir
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.settle import read_arm_action
from interaction_retarget.skill_replay.insert import demo_replay_to_pre_insert
from openpi_client import websocket_client_policy
from pose_insert.pre_insert import resolve_peg_lift_end_frame

from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv
from eval_hybrid_openpi_success import _approach


PROMPT = "Grasp the tray with the left hand and the peg with the right hand, then insert the peg into the hole."
CAMERA_MAPPING = {"base": "ego", "wrist_left": "wrist_left", "wrist_right": "wrist_right"}


def _csv_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one episode index")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=_csv_ints, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--replan-ratio", type=float, default=0.8)
    parser.add_argument("--max-policy-steps", type=int, default=900)
    parser.add_argument("--sidecar-dir", type=Path, default=default_sidecar_dir("bimanual_assembly"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _manifest_entries(sidecar_dir: Path, episodes: list[int]) -> list[dict[str, Any]]:
    manifest = json.loads((sidecar_dir / "manifest.json").read_text(encoding="utf-8"))
    by_episode = {int(entry["episode_index"]): entry for entry in manifest["episodes"]}
    missing = sorted(set(episodes) - set(by_episode))
    if missing:
        raise KeyError(f"episodes missing from manifest: {missing}")
    return [by_episode[episode] for episode in episodes]


def _raw_env(env: Any) -> Any:
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current.unwrapped if hasattr(current, "unwrapped") else current


def _sync_wrapper(wrapper: DexJoCoOpenPIEnv) -> None:
    raw = _raw_env(wrapper.env)
    raw_observation = raw._compute_observation()  # noqa: SLF001
    adapted_observation = wrapper.env.observation(raw_observation)
    wrapper._done = False  # noqa: SLF001
    wrapper._success = False  # noqa: SLF001
    wrapper._update_raw_obs(adapted_observation)  # noqa: SLF001
    wrapper.obs = wrapper._process_obs(adapted_observation)  # noqa: SLF001


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prepare_handoff(
    wrapper: DexJoCoOpenPIEnv,
    entry: dict[str, Any],
    sidecar_dir: Path,
    *,
    video_cb: Callable[[dict], None] | None = None,
    direct_demo_handoff: bool = False,
) -> dict[str, Any]:
    raw = _raw_env(wrapper.env)
    labeler = AssemblyContactLabeler(raw)
    peg_lift_end = resolve_peg_lift_end_frame(entry, sidecar_dir)
    _, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    demo_replay_to_pre_insert(
        wrapper.env,
        raw,
        zarr_path=entry["zarr_path"],
        stop_frame=int(peg_lift_end),
        initial_state=initial_state,
        video_cb=video_cb,
        labeler=labeler,
    )
    replay_end_step = int(getattr(raw, "env_step", 0))
    setup_failure = None
    if not direct_demo_handoff:
        left_hold = np.asarray(read_arm_action(raw, "left"), dtype=np.float64).copy()
        right = np.asarray(read_arm_action(raw, "right"), dtype=np.float64).copy()
        setup_failure = _approach(
            wrapper.env,
            raw,
            labeler,
            left_hold,
            right[7:23].copy(),
            video_cb=video_cb,
        )
    outcome = labeler.compute(raw)
    _sync_wrapper(wrapper)
    return {
        "labeler": labeler,
        "replay_end_step": replay_end_step,
        "handoff_env_step": int(getattr(raw, "env_step", 0)),
        "setup_failure": setup_failure,
        "initial_tray_ok": bool(outcome.tray_ok),
        "initial_peg_ok": bool(outcome.peg_ok),
        "initial_insert_ok": bool(outcome.insert_ok),
    }


def _run_policy(
    wrapper: DexJoCoOpenPIEnv,
    client: websocket_client_policy.WebsocketClientPolicy,
    labeler: AssemblyContactLabeler,
    *,
    action_horizon: int,
    replan_ratio: float,
    max_policy_steps: int,
) -> dict[str, Any]:
    execute_steps = max(1, min(action_horizon, int(round(action_horizon * replan_ratio))))
    policy_steps = 0
    inference_calls = 0
    insert_streak = 0
    max_insert_streak = 0
    peg_lost_streak = 0
    tray_lost_streak = 0
    peg_lost = False
    tray_lost = False
    ever_insert_contact = False

    while policy_steps < max_policy_steps and not wrapper.is_done:
        result = client.infer(wrapper.get_obs())
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 44:
            raise ValueError(f"expected policy actions shaped (T, 44), got {actions.shape}")
        inference_calls += 1
        for action in actions[:execute_steps]:
            wrapper.step(action)
            policy_steps += 1
            outcome = labeler.compute(_raw_env(wrapper.env))
            if bool(outcome.insert_ok):
                insert_streak += 1
                max_insert_streak = max(max_insert_streak, insert_streak)
                ever_insert_contact = True
            else:
                insert_streak = 0
            peg_lost_streak = 0 if outcome.peg_ok else peg_lost_streak + 1
            tray_lost_streak = 0 if outcome.tray_ok else tray_lost_streak + 1
            peg_lost = peg_lost or peg_lost_streak >= 10
            tray_lost = tray_lost or tray_lost_streak >= 10
            if wrapper.is_done or policy_steps >= max_policy_steps:
                break

    final_outcome = labeler.compute(_raw_env(wrapper.env))
    if wrapper.is_success:
        failure_reason = "success"
    elif peg_lost:
        failure_reason = "peg_lost_during_insert"
    elif tray_lost:
        failure_reason = "tray_lost_during_insert"
    elif not ever_insert_contact:
        failure_reason = "alignment_or_hole_entry_failed"
    elif max_insert_streak < 30:
        failure_reason = "unstable_or_incomplete_insertion"
    elif wrapper.is_done:
        failure_reason = "environment_timeout"
    else:
        failure_reason = "policy_step_budget"
    return {
        "success": bool(wrapper.is_success),
        "failure_reason": failure_reason,
        "policy_steps": policy_steps,
        "inference_calls": inference_calls,
        "max_insert_streak": max_insert_streak,
        "ever_insert_contact": ever_insert_contact,
        "final_tray_ok": bool(final_outcome.tray_ok),
        "final_peg_ok": bool(final_outcome.peg_ok),
        "final_insert_ok": bool(final_outcome.insert_ok),
    }


def main() -> int:
    args = _parse_args()
    if args.action_horizon <= 0 or args.max_policy_steps <= 0:
        raise ValueError("action horizon and policy step budget must be positive")
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if args.overwrite:
            shutil.rmtree(output)
        else:
            raise FileExistsError(f"output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    sidecar_dir = args.sidecar_dir.expanduser().resolve()
    entries = _manifest_entries(sidecar_dir, args.episodes)
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    wrapper = DexJoCoOpenPIEnv(
        env_name="bimanual_assembly",
        camera_mapping=CAMERA_MAPPING,
        seed=args.seed,
        rand_full=False,
        randomize_dynamics=False,
        dual_arm=True,
        prompt=PROMPT,
        render_mode="rgb_array",
    )
    wrapper.start()
    rows: list[dict[str, Any]] = []
    try:
        for index, entry in enumerate(entries, start=1):
            episode = int(entry["episode_index"])
            print(f"Episode {index}/{len(entries)} demo={episode}: preparing handoff", flush=True)
            prepared = _prepare_handoff(wrapper, entry, sidecar_dir)
            setup_failure = prepared.pop("setup_failure")
            row = {
                "episode_index": episode,
                "zarr_path": str(entry["zarr_path"]),
                **prepared,
            }
            labeler = row.pop("labeler")
            if setup_failure or not row["initial_peg_ok"]:
                row.update(
                    {
                        "setup_ok": False,
                        "setup_failure": setup_failure or "peg_lost_before_policy",
                        "success": False,
                        "failure_reason": f"setup_failure:{setup_failure or 'peg_lost_before_policy'}",
                        "policy_steps": 0,
                        "inference_calls": 0,
                        "max_insert_streak": 0,
                    }
                )
                print(f"  setup failed: {row['setup_failure']}", flush=True)
            else:
                row["setup_ok"] = True
                row["setup_failure"] = ""
                row.update(
                    _run_policy(
                        wrapper,
                        client,
                        labeler,
                        action_horizon=args.action_horizon,
                        replan_ratio=args.replan_ratio,
                        max_policy_steps=args.max_policy_steps,
                    )
                )
                print(
                    f"  {'SUCCESS' if row['success'] else 'FAIL'} steps={row['policy_steps']} "
                    f"streak={row['max_insert_streak']} reason={row['failure_reason']}",
                    flush=True,
                )
            rows.append(row)
            _atomic_json(output / f"episode_{episode:02d}.json", row)
    finally:
        wrapper.close()

    evaluable = [row for row in rows if row["setup_ok"]]
    successes = sum(bool(row["success"]) for row in evaluable)
    failure_counts = dict(Counter(row["failure_reason"] for row in rows if not row["success"]))
    summary = {
        "protocol": "demo_replay_to_peg_lift_end_then_hybrid_approach_then_openpi",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "episodes_requested": args.episodes,
        "episodes_evaluable": len(evaluable),
        "setup_failures": len(rows) - len(evaluable),
        "successes": successes,
        "success_rate": successes / max(1, len(evaluable)),
        "failure_counts": failure_counts,
        "action_horizon": args.action_horizon,
        "replan_ratio": args.replan_ratio,
        "max_policy_steps": args.max_policy_steps,
        "episodes": rows,
    }
    _atomic_json(output / "summary.json", summary)
    print(
        f"Success rate: {successes}/{len(evaluable)}; setup failures: {len(rows) - len(evaluable)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
