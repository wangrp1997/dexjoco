#!/usr/bin/env python3
from __future__ import annotations

import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from resfit_dexjoco.training._resfit_path import DEXJOco_ROOT  # noqa: E402


@dataclass
class TrainConfig:
    config: Path = DEXJOco_ROOT / "configs/rand_obj/bimanual_assembly.yaml"
    task: str = "bimanual_assembly"
    dataset_root: Path = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")
    checkpoints_root: Path = Path("/mnt/ssd/checkpoints")
    output_dir: Path | None = None
    seed: int = 0
    port: int = 8000
    host: str = "0.0.0.0"
    force_mode: str = "both"

    total_steps: int = 50_000
    batch_size: int = 256
    buffer_size: int = 200_000
    learning_starts: int = 2_000
    train_freq: int = 1
    gradient_steps: int = 1
    gamma: float = 0.99
    offline_fraction: float = 0.5
    use_offline_data: bool = True
    offline_num_episodes: int | None = None
    privileged_sim_state: bool = False

    log_freq: int = 100
    save_freq: int = 5_000
    eval_episodes: int = 5

    wandb_enable: bool = True
    wandb_project: str = "dexjoco"
    wandb_entity: str | None = None
    no_wandb: bool = False
    overwrite: bool = False


def default_output_dir(cfg: TrainConfig) -> Path:
    if cfg.output_dir is not None:
        return cfg.output_dir.expanduser()
    return (
        cfg.checkpoints_root
        / "resfit_dexjoco_ckpt"
        / cfg.task
        / f"forcevla_{cfg.force_mode}"
    )


def _maybe_init_wandb(cfg: TrainConfig, *, output_dir: Path):
    if cfg.no_wandb or not cfg.wandb_enable:
        return None
    import wandb

    run_name = f"resfit_forcevla_{cfg.force_mode}"
    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        config={k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()},
        dir=str(output_dir),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_norm_helpers(state_standardizer, action_scaler, device):
    def standardize_state(state: np.ndarray) -> np.ndarray:
        tensor = state_standardizer.standardize(
            __import__("torch").as_tensor(state, device=device, dtype=__import__("torch").float32)
        )
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    def scale_action(action: np.ndarray) -> np.ndarray:
        tensor = action_scaler.scale(
            __import__("torch").as_tensor(action, device=device, dtype=__import__("torch").float32)
        )
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    def unscale_action(action_n: np.ndarray) -> np.ndarray:
        tensor = action_scaler.unscale(
            __import__("torch").as_tensor(action_n, device=device, dtype=__import__("torch").float32)
        )
        return tensor.detach().cpu().numpy().astype(np.float64, copy=False)

    return standardize_state, scale_action, unscale_action


def main(cfg: TrainConfig) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    _set_seed(cfg.seed)

    import torch

    from resfit_dexjoco.bc.forcevla_client import ForceVLAClient
    from resfit_dexjoco.env.openpi_env import OpenPIEnvConfig, make_openpi_env
    from resfit_dexjoco.env.residual_wrapper import ResidualEnvWrapper
    from resfit_dexjoco.env.rl_obs import rl_obs_spec
    from resfit_dexjoco.training.dataset_stats import load_dataset_stats, pack_norm_stats
    from resfit_dexjoco.training.offline_loader import populate_offline_buffer_gt_as_base
    from resfit_dexjoco.training.proprio_td3 import ProprioResidualTD3
    from resfit_dexjoco.training.replay_buffer import ReplayBuffer, Transition
    from resfit_dexjoco.training.train_env import NormScalars, NormalizedTrainEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_spec = rl_obs_spec(privileged_sim_state=cfg.privileged_sim_state)
    output_dir = default_output_dir(cfg)
    if output_dir.exists() and any(output_dir.iterdir()):
        if cfg.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output path {output_dir} already exists and is not empty. "
                "Remove it, pass --overwrite, or set --output-dir."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir.resolve()}", flush=True)

    wandb_run = _maybe_init_wandb(cfg, output_dir=output_dir)

    state_standardizer, action_scaler = load_dataset_stats(
        cfg.dataset_root,
        state_dim=obs_spec.state_dim,
        device=str(device),
    )
    standardize_state, scale_action, unscale_action = _make_norm_helpers(
        state_standardizer, action_scaler, device
    )

    env_cfg = OpenPIEnvConfig.from_yaml(
        DEXJOco_ROOT / cfg.config,
        seed=cfg.seed,
        force_mode=cfg.force_mode,  # type: ignore[arg-type]
    )
    openpi_env = make_openpi_env(env_cfg)
    bc = ForceVLAClient(host=cfg.host, port=cfg.port, dual_arm=env_cfg.dual_arm)
    bc.start()
    rollout = ResidualEnvWrapper(openpi_env, bc, reward_mode="milestone")
    train_env = NormalizedTrainEnv(
        rollout,
        privileged_sim_state=cfg.privileged_sim_state,
        norm=NormScalars(
            scale_action=scale_action,
            unscale_action=unscale_action,
            standardize_state=standardize_state,
        ),
    )

    agent = ProprioResidualTD3(
        state_dim=obs_spec.state_dim,
        action_dim=obs_spec.action_dim,
        device=device,
    )

    online_rb = ReplayBuffer(cfg.buffer_size, obs_spec.state_dim, obs_spec.action_dim)
    offline_rb = ReplayBuffer(cfg.buffer_size, obs_spec.state_dim, obs_spec.action_dim)

    if cfg.use_offline_data and cfg.offline_fraction > 0.0:
        added = populate_offline_buffer_gt_as_base(
            offline_rb,
            cfg.dataset_root,
            state_dim=obs_spec.state_dim,
            scale_action=scale_action,
            standardize_state=standardize_state,
            num_episodes=cfg.offline_num_episodes,
        )
        print(f"Offline buffer: {added} transitions (GT-as-base, sparse reward)", flush=True)
    else:
        print("Offline buffer disabled", flush=True)

    obs = train_env.reset()
    episode_reward = 0.0
    episode_count = 0

    try:
        for step in range(1, cfg.total_steps + 1):
            state_t = torch.as_tensor(obs["observation.state"], device=device).unsqueeze(0)
            base_t = torch.as_tensor(obs["observation.base_action"], device=device).unsqueeze(0)

            if step < cfg.learning_starts:
                residual_n = (
                    torch.rand((obs_spec.action_dim,), device=device) * 2.0 - 1.0
                ) * 0.2
                residual_n = residual_n.detach().cpu().numpy()
            else:
                with torch.no_grad():
                    residual_n = agent.act(state_t, base_t, eval_mode=False).squeeze(0).cpu().numpy()

            prev_obs = obs
            prev_base_n = train_env.last_base_n.copy()
            next_obs, result = train_env.step_residual_normalized(residual_n)
            combined_n = result.info["combined_normalized"]

            transition = train_env.build_transition(
                prev_obs,
                prev_base_n,
                combined_n,
                result,
                next_obs,
                train_env.last_base_n,
            )
            online_rb.add(transition)

            episode_reward += result.reward
            obs = next_obs

            if result.terminated:
                episode_count += 1
                if step % cfg.log_freq == 0 or result.info.get("succeed"):
                    print(
                        f"step={step} ep_reward={episode_reward:.3f} "
                        f"succeed={result.info.get('succeed')} milestones={result.info.get('milestones_reached')}",
                        flush=True,
                    )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/episode_reward": episode_reward,
                            "train/episode_success": float(bool(result.info.get("succeed"))),
                            "train/global_step": step,
                        },
                        step=step,
                    )
                episode_reward = 0.0
                train_env.end_episode()
                obs = train_env.reset()

            if step >= cfg.learning_starts and step % cfg.train_freq == 0:
                online_batch_size = int(cfg.batch_size * (1.0 - cfg.offline_fraction))
                offline_batch_size = cfg.batch_size - online_batch_size
                batches = []
                if online_batch_size > 0 and len(online_rb) >= online_batch_size:
                    batches.append(online_rb.sample(online_batch_size))
                if (
                    cfg.use_offline_data
                    and cfg.offline_fraction > 0.0
                    and offline_batch_size > 0
                    and len(offline_rb) >= offline_batch_size
                ):
                    batches.append(offline_rb.sample(offline_batch_size))
                if not batches:
                    continue

                if len(batches) == 1:
                    batch = batches[0]
                else:
                    b0, b1 = batches
                    batch = Transition(
                        state=np.concatenate([b0.state, b1.state], axis=0),
                        base_action=np.concatenate([b0.base_action, b1.base_action], axis=0),
                        combined_action=np.concatenate([b0.combined_action, b1.combined_action], axis=0),
                        reward=np.concatenate(
                            [
                                np.asarray(b0.reward).reshape(-1),
                                np.asarray(b1.reward).reshape(-1),
                            ],
                            axis=0,
                        ),
                        next_state=np.concatenate([b0.next_state, b1.next_state], axis=0),
                        next_base_action=np.concatenate(
                            [b0.next_base_action, b1.next_base_action], axis=0
                        ),
                        done=np.concatenate(
                            [
                                np.asarray(b0.done).reshape(-1),
                                np.asarray(b1.done).reshape(-1),
                            ],
                            axis=0,
                        ),
                    )

                for _ in range(cfg.gradient_steps):
                    metrics = agent.update(batch, actor_update=step % 2 == 0)
                if step % cfg.log_freq == 0:
                    print(
                        f"train step={step} online={len(online_rb)} offline={len(offline_rb)} "
                        f"critic={metrics.get('critic_loss', 0):.4f} actor={metrics.get('actor_loss', 0):.4f}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/critic_loss": metrics.get("critic_loss", 0.0),
                                "train/actor_loss": metrics.get("actor_loss", 0.0),
                                "train/online_buffer_size": len(online_rb),
                                "train/offline_buffer_size": len(offline_rb),
                                "train/global_step": step,
                            },
                            step=step,
                        )

            if step % cfg.save_freq == 0:
                ckpt = output_dir / f"checkpoint_step_{step:06d}.pt"
                torch.save(
                    {
                        "step": step,
                        "agent_actor": agent.actor.state_dict(),
                        "agent_critic": agent.critic.state_dict(),
                        "privileged_sim_state": cfg.privileged_sim_state,
                        "state_dim": obs_spec.state_dim,
                        "action_dim": obs_spec.action_dim,
                        "norm_stats": pack_norm_stats(state_standardizer, action_scaler),
                    },
                    ckpt,
                )
                print(f"Saved {ckpt}", flush=True)

    finally:
        train_env.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
