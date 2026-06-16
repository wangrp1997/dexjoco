"""Standalone DexQuery evaluation in DexJoCo simulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

_DEXQUERY_ROOT = Path(__file__).resolve().parents[1]
_DEXJOCo_ROOT = _DEXQUERY_ROOT.parent
if str(_DEXJOCo_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXJOCo_ROOT))

from dexjoco.tasks import CONFIG_MAPPING  # noqa: E402
from dexquery.inference.phase_controller import PhaseControllerConfig  # noqa: E402
from dexquery.policy.dexquery_policy import DexQueryPolicy, load_checkpoint  # noqa: E402


def _policy_action_to_env(action44: np.ndarray) -> np.ndarray:
    r_xyz = action44[:3]
    r_rot = action44[3:6]
    r_hand = action44[6:22]
    l_xyz = action44[22:25]
    l_rot = action44[25:28]
    l_hand = action44[28:44]
    r_quat = R.from_rotvec(r_rot).as_quat(scalar_first=True)
    l_quat = R.from_rotvec(l_rot).as_quat(scalar_first=True)
    return np.concatenate([r_xyz, r_quat, l_xyz, l_quat, r_hand, l_hand]).astype(np.float64)


def _observation_from_env(obs: dict) -> dict:
    r_arm = obs["state"][:7]
    l_arm = obs["state"][7:14]
    r_hand = obs["state"][14:30]
    l_hand = obs["state"][30:46]
    state = np.concatenate([r_arm, l_arm, r_hand, l_hand]).astype(np.float32)
    return {
        "state": state,
        "ego": obs["ego"],
        "wrist_left": obs["wrist_left"],
        "wrist_right": obs["wrist_right"],
    }


def eval_dexquery(
    *,
    task: str,
    checkpoint: Path,
    episodes: int,
    seed: int,
    device: str,
    replan_ratio: float,
    phase_controller: PhaseControllerConfig | None,
    output_dir: Path | None,
    randomize: bool,
) -> float:
    policy = load_checkpoint(
        checkpoint,
        task=task,
        device=device,
        replan_ratio=replan_ratio,
        phase_controller=phase_controller,
    )
    config = CONFIG_MAPPING[task]()
    env = config.get_environment(
        policy_mode=True,
        seed=seed,
        randomize=randomize,
        render_mode=None,
    )

    successes: list[bool] = []
    traces: list[dict] = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset()
            policy.reset()
            done = False
            success = False
            step = 0
            episode_trace = {"episode": episode, "steps": []}

            while not done:
                policy_obs = _observation_from_env(obs)
                action44, info = policy.select_action(policy_obs)
                env_action = _policy_action_to_env(action44)
                obs, _reward, terminated, truncated, info_out = env.step(env_action)
                done = bool(terminated or truncated)
                success = bool(info_out.get("succeed", False))
                episode_trace["steps"].append(
                    {
                        "step": step,
                        "tray_prob": info.tray_prob,
                        "peg_prob": info.peg_prob,
                        "tray_ok": info.tray_ok,
                        "peg_ok": info.peg_ok,
                        "subtask_phase": info.subtask_phase,
                        "replanned": info.replanned,
                    }
                )
                step += 1

            successes.append(success)
            episode_trace["success"] = success
            traces.append(episode_trace)
            print(
                f"Episode {episode + 1}/{episodes}: success={success} steps={step} "
                f"final_phase={episode_trace['steps'][-1]['subtask_phase'] if episode_trace['steps'] else -1}",
                flush=True,
            )
    finally:
        env.close()

    rate = float(np.mean(successes)) if successes else 0.0
    print(f"Success rate: {rate:.1%} ({sum(successes)}/{len(successes)})", flush=True)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "task": task,
            "checkpoint": str(checkpoint),
            "episodes": episodes,
            "seed": seed,
            "success_rate": rate,
            "successes": successes,
        }
        with open(output_dir / "eval_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(output_dir / "phase_traces.json", "w", encoding="utf-8") as f:
            json.dump(traces, f, indent=2)
        print(f"Wrote eval outputs to {output_dir}", flush=True)
    return rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DexQuery checkpoint on DexJoCo.")
    parser.add_argument("--task", required=True, help="Task name, e.g. bimanual_assembly")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replan-ratio", type=float, default=0.8)
    parser.add_argument("--threshold-high", type=float, default=0.6)
    parser.add_argument("--threshold-low", type=float, default=0.4)
    parser.add_argument("--confirm-frames", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--randomize", action="store_true", default=True)
    parser.add_argument("--no-randomize", action="store_false", dest="randomize")
    args = parser.parse_args()

    phase_controller = PhaseControllerConfig(
        threshold_high=args.threshold_high,
        threshold_low=args.threshold_low,
        confirm_frames=args.confirm_frames,
    )
    eval_dexquery(
        task=args.task,
        checkpoint=args.checkpoint.expanduser(),
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        replan_ratio=args.replan_ratio,
        phase_controller=phase_controller,
        output_dir=args.output_dir,
        randomize=args.randomize,
    )


if __name__ == "__main__":
    main()
