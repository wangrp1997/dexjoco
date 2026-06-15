"""Evaluate LeRobot policies (ACT, Diffusion, Multi-Task DiT, GR00T) on DexJoCo simulation."""

from __future__ import annotations

import os
import random
import shutil
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Literal

import imageio
import numpy as np
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.transport import services_pb2

from .async_observation_robot_client import AsyncObservationRobotClient
from .config_dexjoco_robot import DexJoCoRobotConfig  # noqa: F401 — registers robot config
from .dexjoco_robot import DexJoCoRobot
from .eval_config import (
    default_eval_output_dir,
    default_replan_ratio,
    load_eval_yaml,
    resolve_actions_per_chunk,
    video_camera_names,
    write_robot_config_yaml,
)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def reset_client_runtime_state(client: AsyncObservationRobotClient) -> None:
    with client.action_queue_lock:
        client.action_queue = Queue()
    client.clear_pending_observations()
    with client.latest_action_lock:
        client.latest_action = -1
    client.action_chunk_size = -1
    client.must_go.set()
    client.fps_tracker.reset()


def _model_env_video_keys(robot: DexJoCoRobot) -> list[tuple[str, str]]:
    """Return (model_image_key, env_video_filename) pairs for rollout recording."""
    assert robot.model_env_image_map is not None
    return list(robot.model_env_image_map.items())


def eval_n_episodes(
    client: AsyncObservationRobotClient,
    n_episodes: int,
    task: str,
    video_out_root: Path,
) -> float:
    assert isinstance(client.robot, DexJoCoRobot), (
        "client.robot must be an instance of DexJoCoRobot"
    )

    video_keys = _model_env_video_keys(client.robot)
    successes = []

    for episode in range(n_episodes):
        print(f"Episode {episode + 1}/{n_episodes}")

        episode_success = False
        episode_out_path = video_out_root / f"episode_{episode:02d}_temp"
        episode_out_path.mkdir(exist_ok=True, parents=True)
        video_writers = {
            env_name: imageio.get_writer(episode_out_path / f"{env_name}.mp4", fps=30)
            for _, env_name in video_keys
        }

        try:
            client.wait_for_all_observations_sent()
            reset_client_runtime_state(client)
            client.stub.Ready(services_pb2.Empty())  # type: ignore[attr-defined]
            client.robot.reset()

            if client.robot.exp_name == "click_mouse":
                for _ in range(30):
                    obs, *_ = client.robot.env.step(
                        action=np.array([
                            -4.4294e-01,
                            1.3729e-06,
                            1.5170e00,
                            1.3860e-05,
                            -1.0000e00,
                            -2.2014e-05,
                            -4.4665e-04,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0.263,
                            0,
                            0,
                            0,
                        ])
                    )
                    client.robot.observation = client.robot._process_observation(obs)

            in_stay_state = False
            episode_start_time = time.time()

            client.must_go.set()
            client.control_loop_observation(task=task)

            while True:
                if client.actions_available():
                    with client.action_queue_lock:
                        action = client.action_queue.queue[0]
                    if action.get_timestamp() >= episode_start_time:
                        action_legal = True
                    else:
                        with client.action_queue_lock:
                            client.action_queue.get_nowait()
                        time.sleep(client.config.environment_dt)
                        continue
                else:
                    action_legal = False

                if action_legal:
                    client.control_loop_action()
                    in_stay_state = False
                else:
                    client.robot.stay(in_stay_state)
                    in_stay_state = True
                    time.sleep(client.config.environment_dt)

                if client._ready_to_send_observation():
                    client.must_go.set()
                    client.control_loop_observation(task=task)

                obs = client.robot.get_observation()
                for model_key, env_name in video_keys:
                    video_writers[env_name].append_data(obs[model_key])

                if client.robot.is_done:
                    break

            successes.append(client.robot.is_success)
            episode_success = client.robot.is_success
        finally:
            for writer in video_writers.values():
                writer.close()

        result_suffix = "success" if episode_success else "failure"
        episode_out_path.rename(video_out_root / f"episode_{episode:02d}_{result_suffix}")

    num_success = sum(successes)
    (video_out_root / f"success_rate_{num_success}_{n_episodes}.txt").touch()
    return num_success / n_episodes


def main(
    config: Path,
    checkpoint: Path,
    seed: int = 0,
    rand_full: bool = False,
    randomize_dynamics: bool = False,
    host: str = "127.0.0.1",
    port: int = 8080,
    output: Path | None = None,
    overwrite: bool = False,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    replan_ratio: float | None = None,
    episodes: int = 50,
    pad_state_dim46: bool = False,
    policy_type: Literal["act", "diffusion", "multi_task_dit", "groot"] = "act",
    policy_device: str = "cuda",
    actions_per_chunk: int | None = None,
):
    if render_mode == "rgb_array":
        os.environ.setdefault("MUJOCO_GL", "egl")
    else:
        os.environ.setdefault("MUJOCO_GL", "glfw")
    _set_seed(seed)

    eval_cfg = load_eval_yaml(config)
    env_name = eval_cfg["env_name"]
    task = eval_cfg["prompt"]
    robot_type = eval_cfg.get("robot_type", "dual_arm")

    if replan_ratio is None:
        replan_ratio = default_replan_ratio(robot_type)

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not (checkpoint / "config.json").exists():
        raise FileNotFoundError(
            f"Checkpoint directory must contain config.json: {checkpoint}"
        )

    if output is None:
        output_dir = default_eval_output_dir(
            policy_type, env_name, seed, checkpoint, rand_full=rand_full
        )
    else:
        output_dir = output
    if output_dir.exists() and any(output_dir.iterdir()):
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output path {output_dir} already exists and is not empty. "
                "Remove it, pass --overwrite, or choose another --output path."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    if actions_per_chunk is None:
        actions_per_chunk = resolve_actions_per_chunk(policy_type, checkpoint)
    print(
        f"Eval policy={policy_type} | checkpoint={checkpoint} | "
        f"actions_per_chunk={actions_per_chunk} | replan_ratio={replan_ratio} | "
        f"output={output_dir.resolve()}"
    )

    with tempfile.TemporaryDirectory(prefix="dexjoco_lerobot_robot_cfg_") as tmp_dir:
        robot_cfg_path = Path(tmp_dir) / "robot.yaml"
        write_robot_config_yaml(eval_cfg, robot_cfg_path)

        robot_cfg = DexJoCoRobotConfig(
            id=env_name,
            config_path=robot_cfg_path,
            seed=seed,
            randomize=rand_full,
            randomize_dynamics=randomize_dynamics,
            pad_state_dim46=pad_state_dim46,
            render_mode=render_mode,
        )

        server_address = f"{host}:{port}"
        robot_client_cfg = RobotClientConfig(
            policy_type=policy_type,
            pretrained_name_or_path=str(checkpoint),
            robot=robot_cfg,
            actions_per_chunk=actions_per_chunk,
            task=task,
            server_address=server_address,
            policy_device=policy_device,
            client_device="cpu",
            fps=30,
            aggregate_fn_name="latest_only",
            chunk_size_threshold=replan_ratio,
        )

        client = AsyncObservationRobotClient(robot_client_cfg)
        client.start()

        action_receiver_thread = threading.Thread(
            target=client.receive_actions, daemon=True
        )
        action_receiver_thread.start()
        client.start_barrier.wait()

        try:
            success_rate = eval_n_episodes(
                client,
                n_episodes=episodes,
                task=task,
                video_out_root=output_dir,
            )
            print(f"success_rate={success_rate:.3f} ({int(success_rate * episodes)}/{episodes})")
            print(f"Videos saved under: {output_dir.resolve()}")
            print(f"Camera keys: {video_camera_names(eval_cfg)}")
        finally:
            client.stop()
            action_receiver_thread.join()
