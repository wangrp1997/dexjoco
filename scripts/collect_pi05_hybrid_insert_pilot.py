#!/usr/bin/env python3
"""Pilot: rand_obj pi0.5 rollout -> hybrid handoff -> privileged insert; save successes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO = Path(__file__).resolve().parents[1]
_DEXJOco = _REPO / "dexjoco"
for path in (_REPO, _DEXJOco, _REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dexquery.data.finger_contact_forces import FingerForceLabeler
from hybrid_insert import EvalHybridInsert, get_raw_env
from hybrid_insert.assembly_contacts import AssemblyContactLabeler
from openpi_client import websocket_client_policy

from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv
from export_hybrid_insert_ft import _SegmentRecorder

DEFAULT_CONFIG = _REPO / "configs/rand_obj/bimanual_assembly.yaml"
DEFAULT_OUT = Path("/mnt/hdd/dexjoco/outputs/pi05_hybrid_insert_collect_raw_force")
PROMPT = (
    "Grasp the tray with the left hand and the peg with the right hand, "
    "then insert the peg into the hole."
)
CAMERA_NAMES = ("ego", "wrist_left", "wrist_right")
FPS = 30


def _load_eval_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_summary(path: Path, summary: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _abort_reason(
    labeler: AssemblyContactLabeler,
    raw,
    *,
    policy_steps: int,
    had_tray: bool,
    had_peg: bool,
    tray_lost_streak: int,
    peg_lost_streak: int,
    hybrid_active: bool,
    max_policy_steps: int,
    grasp_deadline: int,
) -> str | None:
    if hybrid_active:
        return None
    outcome = labeler.compute(raw)
    if policy_steps >= grasp_deadline and not (had_tray and had_peg):
        return "never_grasped_both"
    if had_tray and tray_lost_streak >= 12:
        return "tray_lost"
    if had_peg and peg_lost_streak >= 12:
        return "peg_lost"
    if policy_steps >= max_policy_steps:
        return "handoff_timeout"
    return None


def run_rollout(
    *,
    rollout_index: int,
    seed: int,
    cfg: dict,
    client: websocket_client_policy.WebsocketClientPolicy,
    out_root: Path,
    action_horizon: int,
    replan_ratio: float,
    max_policy_steps: int,
    grasp_deadline: int,
    rand_full: bool,
) -> dict:
    env_name = cfg["env_name"]
    wrapper = DexJoCoOpenPIEnv(
        env_name=env_name,
        camera_mapping=cfg["camera_mapping"],
        seed=seed,
        rand_full=rand_full,
        randomize_dynamics=False,
        dual_arm=cfg["robot_type"] == "dual_arm",
        prompt=cfg["prompt"],
        render_mode="rgb_array",
    )
    wrapper.start()
    hybrid = EvalHybridInsert(task=env_name, enabled=True)
    raw = get_raw_env(wrapper.env)
    labeler = AssemblyContactLabeler(raw)
    force_labeler = FingerForceLabeler(raw)
    hybrid.on_reset(wrapper.env)

    execute_steps = max(1, min(action_horizon, int(round(action_horizon * replan_ratio))))
    policy_steps = 0
    had_tray = False
    had_peg = False
    tray_lost_streak = 0
    peg_lost_streak = 0
    recorder: _SegmentRecorder | None = None
    temp_dir = out_root / f"rollout_{rollout_index:04d}_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    result = {
        "rollout_index": rollout_index,
        "seed": seed,
        "success": False,
        "fail_reason": "",
        "policy_steps": 0,
        "insert_frames": 0,
        "handoff": False,
        "handoff_env_step": -1,
        "max_insert_streak": 0,
    }
    insert_streak = 0
    max_insert_streak = 0

    try:
        wrapper.reset()
        labeler.reset_reference(raw)
        force_labeler.reset_reference(raw)
        action_buffer: deque[np.ndarray] = deque()
        while not wrapper.is_done:
            if not action_buffer:
                infer = client.infer(wrapper.get_obs())
                infer_actions = np.asarray(infer["actions"], dtype=np.float32)
                if infer_actions.ndim != 2 or infer_actions.shape[1] != 44:
                    raise ValueError(f"expected (T,44) actions, got {infer_actions.shape}")
                for row in infer_actions[:execute_steps]:
                    action_buffer.append(row)

            policy_action = action_buffer.popleft()
            if not hybrid.active:
                hybrid.observe(wrapper.env, policy_action)
            merged = hybrid.merge(wrapper.env, policy_action)

            if hybrid.active and recorder is None:
                temp_dir.mkdir(parents=True, exist_ok=True)
                recorder = _SegmentRecorder(raw, temp_dir, force_labeler=force_labeler)
                result["handoff"] = True
                result["handoff_env_step"] = int(getattr(raw, "env_step", policy_steps))
                print(f"  rollout {rollout_index}: handoff at policy_step={policy_steps}", flush=True)

            wrapper.step(merged)
            policy_steps += 1
            outcome = labeler.compute(raw)
            if outcome.tray_ok:
                had_tray = True
                tray_lost_streak = 0
            else:
                tray_lost_streak += 1 if had_tray else 0
            if outcome.peg_ok:
                had_peg = True
                peg_lost_streak = 0
            else:
                peg_lost_streak += 1 if had_peg else 0
            if outcome.insert_ok:
                insert_streak += 1
                max_insert_streak = max(max_insert_streak, insert_streak)
            else:
                insert_streak = 0

            if recorder is not None:
                recorder.capture(action44=merged, phase=hybrid.controller.phase_name, labeler=labeler)

            abort = _abort_reason(
                labeler,
                raw,
                policy_steps=policy_steps,
                had_tray=had_tray,
                had_peg=had_peg,
                tray_lost_streak=tray_lost_streak,
                peg_lost_streak=peg_lost_streak,
                hybrid_active=hybrid.active,
                max_policy_steps=max_policy_steps,
                grasp_deadline=grasp_deadline,
            )
            if abort is not None:
                result["fail_reason"] = abort
                break

            if wrapper.is_done:
                result["success"] = bool(wrapper.is_success)
                if not result["success"]:
                    result["fail_reason"] = result["fail_reason"] or "env_done_no_success"
                break

        result["policy_steps"] = policy_steps
        result["max_insert_streak"] = int(max_insert_streak)
        if recorder is not None:
            result["insert_frames"] = recorder.close()

        if result["success"] and recorder is not None and result["insert_frames"] > 0:
            final_dir = out_root / f"episode_{rollout_index:04d}_success"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            recorder.save_npz(temp_dir / "trajectory.npz")
            meta = {
                "episode_index": int(rollout_index),
                "seed": int(seed),
                "openpi_success": True,
                "fail_reason": "",
                "max_insert_streak": int(max_insert_streak),
                "num_frames": int(result["insert_frames"]),
                "handoff_env_step": int(result["handoff_env_step"]),
                "segment": "pi05_policy_then_hybrid_pbvs_insert_forcevla",
                "prompt": PROMPT,
                "observation_state_dim": 46,
                "action_dim": 44,
                "camera_names": list(CAMERA_NAMES),
                "fps": FPS,
                "force_fields": {
                    "wrist_ft_right": 6,
                    "wrist_ft_left": 6,
                    "right_finger_force": 12,
                    "left_finger_force": 12,
                    "force": 36,
                },
                "force_layout": "ForceVLA both = [wrist_r(6), wrist_l(6), finger_r(12), finger_l(12)]",
                "force_aligned": True,
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }
            (temp_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            temp_dir.rename(final_dir)
            result["output_dir"] = str(final_dir)
        elif temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return result
    finally:
        wrapper.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-rollouts", type=int, default=15)
    parser.add_argument("--target-successes", type=int, default=3)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--replan-ratio", type=float, default=0.8)
    parser.add_argument("--max-policy-steps", type=int, default=1200)
    parser.add_argument("--grasp-deadline", type=int, default=700)
    parser.add_argument("--rand-full", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cfg = _load_eval_cfg(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    results: list[dict] = []
    failures: Counter[str] = Counter()
    successes = 0
    prior_elapsed_s = 0.0
    t0 = time.time()
    summary_path = args.output / "summary.json"
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        results = list(prior.get("results") or [])
        failures = Counter(prior.get("failure_counts") or {})
        successes = int(prior.get("successes") or 0)
        prior_elapsed_s = float(prior.get("elapsed_s") or 0.0)
        if results:
            last_idx = max(int(r["rollout_index"]) for r in results)
            # Continue after the last finished rollout unless user set a later start.
            args.start_index = max(int(args.start_index), last_idx + 1)
            # Keep seed aligned with original formula: seed = seed_base + (rollout_index - first_start).
            # With seed_base=251400 and start_index growing, prefer seed = seed_base + rollout_index
            # when resuming from a prior run that used seed=seed_base+i with start_index=0.
            print(
                f"[collect] resume successes={successes} attempts={len(results)} "
                f"next_episode={args.start_index:04d}",
                flush=True,
            )

    def snapshot() -> dict:
        return {
            "protocol": "rand_obj_pi05_policy_to_hybrid_handoff_then_privileged_insert_forcevla",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": str(args.config),
            "rand_full": bool(args.rand_full),
            "force_aligned": True,
            "force_layout": "ForceVLA both = [wrist_r(6), wrist_l(6), finger_r(12), finger_l(12)]",
            "max_rollouts": args.max_rollouts,
            "target_successes": args.target_successes,
            "successes": successes,
            "attempts": len(results),
            "success_rate": successes / len(results) if results else 0.0,
            "failure_counts": dict(failures),
            "elapsed_s": round(prior_elapsed_s + (time.time() - t0), 1),
            "results": results,
        }

    for i in range(args.max_rollouts):
        if successes >= args.target_successes:
            break
        rollout_index = args.start_index + i
        # Original launch: start_index=0, seed=seed_base+i == seed_base+rollout_index.
        seed = args.seed_base + rollout_index
        print(
            f"[collect] rollout {i + 1}/{args.max_rollouts} "
            f"episode={rollout_index:04d} seed={seed}",
            flush=True,
        )
        row = run_rollout(
            rollout_index=rollout_index,
            seed=seed,
            cfg=cfg,
            client=client,
            out_root=args.output,
            action_horizon=args.action_horizon,
            replan_ratio=args.replan_ratio,
            max_policy_steps=args.max_policy_steps,
            grasp_deadline=args.grasp_deadline,
            rand_full=args.rand_full,
        )
        results.append(row)
        if row["success"]:
            successes += 1
            print(f"  SUCCESS frames={row['insert_frames']} dir={row.get('output_dir', '')}", flush=True)
        else:
            failures[row.get("fail_reason") or "unknown"] += 1
            print(f"  FAIL reason={row.get('fail_reason')}", flush=True)
        _write_summary(summary_path, snapshot())

    summary = snapshot()
    _write_summary(summary_path, summary)
    print(
        f"[collect] done successes={successes}/{len(results)} "
        f"({100 * summary['success_rate']:.1f}%) -> {summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
