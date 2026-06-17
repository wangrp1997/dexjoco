"""Standalone DexQuery evaluation in DexJoCo simulation."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import imageio
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

_DEXQUERY_ROOT = Path(__file__).resolve().parents[1]
_DEXJOCo_ROOT = _DEXQUERY_ROOT.parent
if str(_DEXJOCo_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXJOCo_ROOT))
_DEXJOCo_PKG = _DEXJOCo_ROOT / "dexjoco"
if str(_DEXJOCo_PKG) not in sys.path:
    sys.path.insert(0, str(_DEXJOCo_PKG))

from dexjoco.tasks import CONFIG_MAPPING  # noqa: E402
from dexjoco_lerobot_client.eval_config import (  # noqa: E402
    default_eval_output_dir,
    default_replan_ratio,
    load_eval_yaml,
    video_camera_names,
)
from dexquery.inference.phase_controller import PhaseControllerConfig  # noqa: E402
from dexquery.policy.dexquery_policy import load_checkpoint  # noqa: E402
from hybrid_insert import EvalHybridInsert  # noqa: E402

SIM_ORACLE_TASKS = frozenset({"bimanual_assembly"})

EVAL_FPS = 30
VIDEO_CAMERA_KEYS: tuple[str, ...] = ("ego", "wrist_left", "wrist_right")


def _load_dexquery_inference_config(task: str) -> dict:
    path = _DEXQUERY_ROOT / "configs" / f"{task}.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("inference", {})


def _default_replan_ratio(task: str, robot_type: str) -> float:
    inference_cfg = _load_dexquery_inference_config(task)
    if "replan_ratio" in inference_cfg:
        return float(inference_cfg["replan_ratio"])
    return default_replan_ratio(robot_type)


def _phase_controller_from_task(task: str, overrides: dict) -> PhaseControllerConfig:
    inference_cfg = _load_dexquery_inference_config(task)
    phase_cfg = dict(inference_cfg.get("phase_controller", {}))
    phase_cfg.update({k: v for k, v in overrides.items() if v is not None})
    return PhaseControllerConfig(**phase_cfg)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


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


def _frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    frame = np.asarray(frame)
    if frame.max() <= 1.0:
        frame = (frame * 255.0).clip(0, 255)
    return frame.astype(np.uint8)


def _append_video_frame(writer: imageio.core.format.Writer, frame: np.ndarray) -> None:
    writer.append_data(_frame_to_uint8(frame))


def _write_episode_videos(
    video_out_root: Path,
    episode: int,
    obs: dict,
    *,
    save_attn: bool = False,
) -> dict[str, imageio.core.format.Writer]:
    episode_out_path = video_out_root / f"episode_{episode:02d}_temp"
    episode_out_path.mkdir(exist_ok=True, parents=True)
    writers = {
        camera: imageio.get_writer(episode_out_path / f"{camera}.mp4", fps=EVAL_FPS)
        for camera in VIDEO_CAMERA_KEYS
        if camera in obs
    }
    if save_attn:
        for camera in VIDEO_CAMERA_KEYS:
            if camera in obs:
                writers[f"attn_{camera}"] = imageio.get_writer(
                    episode_out_path / f"attn_{camera}.mp4",
                    fps=EVAL_FPS,
                )
    return writers


def _record_plain_camera_frames(
    writers: dict[str, imageio.core.format.Writer],
    obs: dict,
) -> None:
    for camera in VIDEO_CAMERA_KEYS:
        writer_key = f"attn_{camera}"
        if writer_key in writers and camera in obs:
            _append_video_frame(writers[writer_key], obs[camera])


def _record_attn_frames(
    writers: dict[str, imageio.core.format.Writer],
    attn_overlays: dict[str, np.ndarray] | None,
) -> None:
    if not attn_overlays:
        return
    for camera, frame in attn_overlays.items():
        writer_key = f"attn_{camera}"
        if writer_key in writers:
            _append_video_frame(writers[writer_key], frame)


def _record_observation_frames(
    writers: dict[str, imageio.core.format.Writer],
    obs: dict,
) -> None:
    for camera, writer in writers.items():
        if camera in obs:
            _append_video_frame(writer, obs[camera])


def eval_dexquery(
    *,
    task: str,
    checkpoint: Path,
    episodes: int,
    seed: int,
    device: str,
    replan_ratio: float,
    phase_controller: PhaseControllerConfig | None,
    output_dir: Path,
    randomize: bool,
    save_attn_videos: bool = False,
    hybrid_insert: bool = False,
) -> float:
    policy = load_checkpoint(
        checkpoint,
        task=task,
        device=device,
        replan_ratio=replan_ratio,
        phase_controller=phase_controller,
        save_attn_videos=save_attn_videos,
    )
    config = CONFIG_MAPPING[task]()
    env = config.get_environment(
        policy_mode=True,
        seed=seed,
        randomize=randomize,
        render_mode="rgb_array",
    )
    raw_env = env.unwrapped
    contact_labeler = None
    if task in SIM_ORACLE_TASKS:
        from dexquery.data.assembly_contacts import AssemblyContactLabeler

        contact_labeler = AssemblyContactLabeler(raw_env)

    hybrid = EvalHybridInsert(task=task, enabled=hybrid_insert)
    if hybrid.enabled:
        print("hybrid_insert: enabled for bimanual_assembly insert phase", flush=True)

    successes: list[bool] = []
    traces: list[dict] = []
    try:
        for episode in range(episodes):
            print(f"Episode {episode + 1}/{episodes}", flush=True)
            obs, _ = env.reset()
            policy.reset()
            hybrid.on_reset(env)
            if contact_labeler is not None:
                contact_labeler.reset_reference(raw_env)
            done = False
            success = False
            step = 0
            episode_trace = {"episode": episode, "steps": []}
            video_writers = _write_episode_videos(
                output_dir,
                episode,
                obs,
                save_attn=save_attn_videos,
            )
            _record_observation_frames(video_writers, obs)
            if save_attn_videos:
                _record_plain_camera_frames(video_writers, obs)

            try:
                while not done:
                    tray_ok_sim: bool | None = None
                    peg_ok_sim: bool | None = None
                    if contact_labeler is not None:
                        sim_outcome = contact_labeler.compute(raw_env)
                        tray_ok_sim = sim_outcome.tray_ok
                        peg_ok_sim = sim_outcome.peg_ok

                    policy_obs = _observation_from_env(obs)
                    state46 = policy_obs["state"]
                    action44, info = policy.select_action(
                        policy_obs,
                        tray_ok_sim=tray_ok_sim,
                        peg_ok_sim=peg_ok_sim,
                    )
                    if hybrid.enabled and not hybrid.active:
                        hybrid.observe(env, action44)
                    action44 = hybrid.merge(env, action44)
                    env_action = _policy_action_to_env(action44)
                    obs, _reward, terminated, truncated, info_out = env.step(env_action)
                    done = bool(terminated or truncated)
                    success = bool(info_out.get("succeed", False))
                    _record_observation_frames(video_writers, obs)
                    if save_attn_videos and info is not None:
                        _record_attn_frames(video_writers, info.attn_overlays)
                    step_record = {
                        "step": step,
                        "hybrid_insert_active": hybrid.active,
                    }
                    if info is not None:
                        step_record.update(
                            {
                                "tray_prob": info.tray_prob,
                                "peg_prob": info.peg_prob,
                                "tray_ok": info.tray_ok,
                                "peg_ok": info.peg_ok,
                                "subtask_phase": info.subtask_phase,
                                "replanned": info.replanned,
                            }
                        )
                    if tray_ok_sim is not None:
                        step_record["tray_ok_sim"] = tray_ok_sim
                        step_record["peg_ok_sim"] = peg_ok_sim
                    episode_trace["steps"].append(step_record)
                    step += 1
            finally:
                for writer in video_writers.values():
                    writer.close()

            successes.append(success)
            episode_trace["success"] = success
            traces.append(episode_trace)
            result_suffix = "success" if success else "failure"
            temp_dir = output_dir / f"episode_{episode:02d}_temp"
            final_dir = output_dir / f"episode_{episode:02d}_{result_suffix}"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            temp_dir.rename(final_dir)
            print(
                f"Episode {episode + 1}/{episodes}: success={success} steps={step} "
                f"final_phase={episode_trace['steps'][-1]['subtask_phase'] if episode_trace['steps'] else -1}",
                flush=True,
            )
    finally:
        env.close()

    rate = float(np.mean(successes)) if successes else 0.0
    num_success = int(sum(successes))
    print(f"Success rate: {rate:.1%} ({num_success}/{len(successes)})", flush=True)

    (output_dir / f"success_rate_{num_success}_{episodes}.txt").touch()
    summary = {
        "task": task,
        "checkpoint": str(checkpoint),
        "episodes": episodes,
        "seed": seed,
        "success_rate": rate,
        "successes": successes,
        "replan_ratio": replan_ratio,
        "phase_controller": {
            "threshold_high": phase_controller.threshold_high if phase_controller else None,
            "threshold_low": phase_controller.threshold_low if phase_controller else None,
            "confirm_frames": phase_controller.confirm_frames if phase_controller else None,
            "insert_min_prob": phase_controller.insert_min_prob if phase_controller else None,
            "use_sim_guard": phase_controller.use_sim_guard if phase_controller else None,
        },
        "save_attn_videos": save_attn_videos,
        "hybrid_insert": hybrid_insert,
    }
    with open(output_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "phase_traces.json", "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2)
    print(f"Videos and eval outputs saved under: {output_dir.resolve()}", flush=True)
    return rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DexQuery checkpoint on DexJoCo.")
    parser.add_argument("--task", required=True, help="Task name, e.g. bimanual_assembly")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="Eval yaml under configs/rand_obj/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replan-ratio", type=float, default=None)
    parser.add_argument("--threshold-high", type=float, default=None)
    parser.add_argument("--threshold-low", type=float, default=None)
    parser.add_argument("--confirm-frames", type=int, default=None)
    parser.add_argument("--insert-min-prob", type=float, default=None)
    parser.add_argument(
        "--no-sim-guard",
        action="store_true",
        help="Disable sim contact guard even when task yaml enables it",
    )
    parser.add_argument("--output", "--output-dir", type=Path, default=None, dest="output_dir")
    parser.add_argument("--rand-full", action="store_true", help="Enable full scene randomization")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-attn-videos",
        action="store_true",
        help="Save per-camera cross-attention heatmap videos under each episode directory",
    )
    parser.add_argument(
        "--hybrid-insert",
        action="store_true",
        help="Use privileged geometry controller for insert after grasp (bimanual_assembly)",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    _set_seed(args.seed)

    eval_cfg_path = args.config or (_DEXJOCo_ROOT / "configs/rand_obj" / f"{args.task}.yaml")
    eval_cfg = load_eval_yaml(eval_cfg_path)
    env_name = eval_cfg["env_name"]
    robot_type = eval_cfg.get("robot_type", "dual_arm")
    replan_ratio = (
        float(args.replan_ratio)
        if args.replan_ratio is not None
        else _default_replan_ratio(args.task, robot_type)
    )

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if args.output_dir is None:
        output_dir = default_eval_output_dir(
            "dexquery",
            env_name,
            args.seed,
            checkpoint,
            rand_full=args.rand_full,
            hybrid_insert=args.hybrid_insert,
        )
    else:
        output_dir = args.output_dir.expanduser()
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output path {output_dir} already exists and is not empty. "
                "Remove it, pass --overwrite, or choose another --output path."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_controller = _phase_controller_from_task(
        args.task,
        {
            "threshold_high": args.threshold_high,
            "threshold_low": args.threshold_low,
            "confirm_frames": args.confirm_frames,
            "insert_min_prob": args.insert_min_prob,
        },
    )
    if args.no_sim_guard:
        phase_controller = PhaseControllerConfig(
            threshold_high=phase_controller.threshold_high,
            threshold_low=phase_controller.threshold_low,
            confirm_frames=phase_controller.confirm_frames,
            insert_min_prob=phase_controller.insert_min_prob,
            use_sim_guard=False,
        )
    print(
        f"Eval policy=dexquery | checkpoint={checkpoint} | replan_ratio={replan_ratio} | "
        f"phase={phase_controller} | output={output_dir.resolve()}",
        flush=True,
    )
    print(f"Camera keys: {video_camera_names(eval_cfg)}", flush=True)

    eval_dexquery(
        task=args.task,
        checkpoint=checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        replan_ratio=replan_ratio,
        phase_controller=phase_controller,
        output_dir=output_dir,
        randomize=args.rand_full,
        save_attn_videos=args.save_attn_videos,
        hybrid_insert=args.hybrid_insert,
    )


if __name__ == "__main__":
    main()
