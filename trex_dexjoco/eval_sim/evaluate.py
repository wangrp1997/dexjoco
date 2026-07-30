"""Evaluate T-Rex post-train on DexJoCo sim (same video / success format as pi0.5).

Output example (cwd = dexjoco repo root, ``outputs`` → ``/mnt/hdd/dexjoco/outputs``)::

    outputs/trex/bimanual_assembly_seed0_ckpt000013/
      episode_00_success/
      episode_01_failure/
      ...
      success_rate_12_50.txt
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import imageio
import numpy as np
import yaml

_TREX_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _TREX_ROOT.parent
_DEXJOCO_PKG = _REPO_ROOT / "dexjoco"
for p in (str(_TREX_ROOT), str(_REPO_ROOT), str(_DEXJOCO_PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_sim.policy import TrexPolicy, resolve_trex_ckpt_label  # noqa: E402

# Match pi0.5 / ForceVLA eval protocol.
EVAL_MAX_VIDEO_FRAMES = 1500


@dataclass
class TimedAction:
    action: np.ndarray
    timestamp: int


def _set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def _append_video_frames(video_writers: dict, raw_images: dict, frame_count: list[int]) -> bool:
    if frame_count[0] >= EVAL_MAX_VIDEO_FRAMES:
        return False
    for cam_name, writer in video_writers.items():
        writer.append_data(raw_images[cam_name])
    frame_count[0] += 1
    return frame_count[0] < EVAL_MAX_VIDEO_FRAMES


def _default_output_dir(
    env_name: str,
    seed: int,
    checkpoint: Path,
    *,
    rand_full: bool = False,
    hybrid_insert: bool = False,
    skill_graph_recovery: bool = False,
) -> Path:
    suffix = "_rand_full" if rand_full else ""
    hybrid_suffix = "_hybrid" if hybrid_insert else ""
    skill_graph_suffix = "_skill_graph" if skill_graph_recovery else ""
    ckpt_label = resolve_trex_ckpt_label(checkpoint)
    return (
        Path("outputs")
        / "trex"
        / f"{env_name}{suffix}{hybrid_suffix}{skill_graph_suffix}_seed{seed}_{ckpt_label}"
    )


def main(
    config: Path = Path("configs/rand_obj/bimanual_assembly.yaml"),
    checkpoint: Path = Path(
        "/mnt/hdd/checkpoints/trex_dexjoco_ckpt/bimanual_assembly/"
        "trex_posttrain_bimanual_assembly/"
        "trex_posttrain_bimanual_assembly_0728_2128/checkpoint-13-44646"
    ),
    seed: int = 0,
    rand_full: bool = False,
    randomize_dynamics: bool = False,
    output: Path | None = None,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    replan_ratio: float = 0.25,
    episodes: int = 50,
    hybrid_insert: bool = False,
    overwrite: bool = False,
    skill_graph_recovery: bool = False,
    cuda: int = 0,
    image_w: int = 384,
    image_h: int = 288,
    base_model_path: str = "/mnt/hdd/checkpoints/trex/Qwen3-VL-2B-Instruct",
    progress_every: int = 50,
):
    """Run DexJoCo sim eval for a T-Rex checkpoint.

    Videos / success marker match ``dexjoco-openpi-eval`` (pi0.5 / ForceVLA).
    Requires ``force_mode=finger`` (sim contact → tactile [8,3]).

    Note: inference is synchronous (unlike pi0.5 websocket async). Keep
    ``replan_ratio`` low (default 0.25) so most of the 16-step chunk is used
    before the next slow_and_fast call; 0.8 would replan every ~4 steps and
    make one episode take tens of minutes.
    """
    if render_mode == "rgb_array":
        os.environ.setdefault("MUJOCO_GL", "egl")
    else:
        os.environ.setdefault("MUJOCO_GL", "glfw")
    _set_seed(seed)

    # Prefer repo-relative config when cwd is trex_dexjoco.
    if not config.is_file():
        alt = _REPO_ROOT / config
        if alt.is_file():
            config = alt
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    env_name = cfg["env_name"]
    camera_mapping = cfg["camera_mapping"]
    robot_type = cfg["robot_type"]
    dual_arm = robot_type == "dual_arm"
    if not dual_arm:
        raise ValueError("T-Rex DexJoCo eval currently supports dual_arm only")
    prompt = cfg["prompt"]

    if output is None:
        output_dir = _default_output_dir(
            env_name, seed, checkpoint,
            rand_full=rand_full,
            hybrid_insert=hybrid_insert,
            skill_graph_recovery=skill_graph_recovery,
        )
    else:
        output_dir = output
    # Resolve relative to repo root so results land next to pi0.5 / forcevla.
    if not output_dir.is_absolute():
        output_dir = (_REPO_ROOT / output_dir).resolve()
    print(f"Eval output: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output path {output_dir} already exists and is not empty. "
                "Remove it, pass --overwrite, or choose another --output path."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy before MuJoCo env so import/weight errors don't leave EGL viewers.
    policy = TrexPolicy(
        str(checkpoint),
        cuda=cuda,
        image_size=(image_w, image_h),
        base_model_path=base_model_path,
        prompt=prompt,
    )
    action_horizon = policy.action_chunk

    from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv
    from hybrid_insert import EvalHybridInsert, state_to_dual_arm_action44

    env = DexJoCoOpenPIEnv(
        env_name=env_name,
        camera_mapping=camera_mapping,
        seed=seed,
        rand_full=rand_full,
        randomize_dynamics=randomize_dynamics,
        dual_arm=dual_arm,
        prompt=prompt,
        render_mode=render_mode,
        force_mode="finger",
    )
    env.start()
    print("force_mode=finger: sim finger contact → T-Rex tactile [8,3]", flush=True)

    hybrid = EvalHybridInsert(task=env_name, enabled=hybrid_insert)
    if hybrid.enabled:
        print("hybrid_insert: enabled", flush=True)

    regrasp_hook = None
    if skill_graph_recovery:
        if env_name != "bimanual_assembly":
            raise ValueError("skill_graph_recovery only supports bimanual_assembly")
        from skill_graph.hooks.vla_recovery import RegraspRecoveryHook

        regrasp_hook = RegraspRecoveryHook()
        print("skill_graph: regrasp recovery enabled", flush=True)

    video_writers = None

    try:
        num_success = 0
        for ep in range(episodes):
            print(f"Episode {ep + 1}/{episodes}", flush=True)
            video_dir = output_dir / f"episode_{ep:02d}_temp"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_writers = {
                cam_name: imageio.get_writer(video_dir / f"{cam_name}.mp4", fps=30)
                for cam_name in camera_mapping.values()
            }

            env.reset()
            hybrid.on_reset(env.env)
            if regrasp_hook is not None:
                regrasp_hook.reset_episode(env.env)
            policy.reset()

            timestamp = 0
            actions_buffer: deque[TimedAction] = deque()
            video_frame_count = [0]
            in_stay_state = False

            raw_images = env.get_raw_images()
            _append_video_frames(video_writers, raw_images, video_frame_count)

            # Seed first chunk.
            t_infer0 = time.time()
            chunk = policy.infer_chunk(env.get_obs())
            print(
                f"  first infer: {chunk.shape[0]} actions, "
                f"{(time.time() - t_infer0):.1f}s",
                flush=True,
            )
            for i, a in enumerate(chunk):
                actions_buffer.append(TimedAction(action=a, timestamp=timestamp + i))

            while True:
                if regrasp_hook is not None and regrasp_hook.is_busy():
                    recovery_done = regrasp_hook.step_recovery(env, timestamp=timestamp)
                    timestamp += 1
                    raw_images = env.get_raw_images()
                    _append_video_frames(video_writers, raw_images, video_frame_count)
                    if recovery_done:
                        actions_buffer.clear()
                        chunk = policy.infer_chunk(env.get_obs())
                        for i, a in enumerate(chunk):
                            actions_buffer.append(
                                TimedAction(action=a, timestamp=timestamp + i)
                            )
                    if video_frame_count[0] >= EVAL_MAX_VIDEO_FRAMES:
                        env._done = True
                    if env.is_done:
                        if env.is_success:
                            num_success += 1
                            print("Success!")
                        else:
                            print("Failed")
                        break
                    continue

                # Drop expired.
                while actions_buffer and actions_buffer[0].timestamp < timestamp:
                    actions_buffer.popleft()

                state46 = env.obs["state"]
                if actions_buffer:
                    policy_action = actions_buffer.popleft().action
                    if hybrid.enabled and not hybrid.active:
                        hybrid.observe(env.env, policy_action)
                    action = (
                        hybrid.merge(env.env, policy_action)
                        if hybrid.enabled
                        else policy_action
                    )
                    env.step(action)
                    in_stay_state = False
                elif hybrid.active:
                    hold_action = state_to_dual_arm_action44(state46)
                    action = hybrid.merge(env.env, hold_action)
                    env.step(action)
                    in_stay_state = False
                else:
                    env.stay(continue_stay=in_stay_state)
                    in_stay_state = True

                timestamp += 1
                if progress_every > 0 and timestamp % progress_every == 0:
                    print(
                        f"  t={timestamp} frames={video_frame_count[0]} "
                        f"buf={len(actions_buffer)}",
                        flush=True,
                    )
                raw_images = env.get_raw_images()
                _append_video_frames(video_writers, raw_images, video_frame_count)

                if regrasp_hook is not None and timestamp > 0 and (timestamp % 10 == 0):
                    regrasp_hook.log_status(env.env, timestamp=timestamp)
                    if regrasp_hook.maybe_start_recovery(env.env, timestamp=timestamp):
                        actions_buffer.clear()
                        chunk = policy.infer_chunk(env.get_obs())
                        for i, a in enumerate(chunk):
                            actions_buffer.append(
                                TimedAction(action=a, timestamp=timestamp + i)
                            )

                if len(actions_buffer) < replan_ratio * action_horizon:
                    t_infer0 = time.time()
                    chunk = policy.infer_chunk(env.get_obs())
                    # Replace future buffer with fresh chunk (sync replan).
                    actions_buffer.clear()
                    for i, a in enumerate(chunk):
                        actions_buffer.append(
                            TimedAction(action=a, timestamp=timestamp + i)
                        )
                    if timestamp < 5 or timestamp % max(progress_every, 1) == 0:
                        print(
                            f"  replan@{timestamp}: {(time.time() - t_infer0):.1f}s "
                            f"chunk={chunk.shape[0]}",
                            flush=True,
                        )

                if video_frame_count[0] >= EVAL_MAX_VIDEO_FRAMES:
                    env._done = True
                if env.is_done:
                    if env.is_success:
                        num_success += 1
                        print("Success!")
                    else:
                        print("Failed")
                    if hybrid.enabled:
                        print(f"  hybrid_insert: {hybrid.episode_summary()}", flush=True)
                    if regrasp_hook is not None:
                        print(f"  skill_graph: {regrasp_hook.episode_summary()}", flush=True)
                    break

            for writer in video_writers.values():
                writer.close()
            video_writers = None

            result_suffix = "success" if env.is_success else "failure"
            final_video_dir = output_dir / f"episode_{ep:02d}_{result_suffix}"
            video_dir.rename(final_video_dir)

        print(
            f"\nSuccess rate: {num_success}/{episodes} "
            f"({100 * num_success / episodes:.1f}%)"
        )
        (output_dir / f"success_rate_{num_success}_{episodes}.txt").touch()
    finally:
        env.close()
        if video_writers is not None:
            for writer in video_writers.values():
                writer.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
