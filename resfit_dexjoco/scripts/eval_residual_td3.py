#!/usr/bin/env python3
"""Evaluate frozen ForceVLA + residual TD3 checkpoint."""

from __future__ import annotations

import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEXJOco_PKG = _REPO_ROOT / "dexjoco"
if str(_DEXJOco_PKG) not in sys.path:
    sys.path.insert(0, str(_DEXJOco_PKG))

from resfit_dexjoco.training._resfit_path import DEXJOco_ROOT  # noqa: E402
from resfit_dexjoco.utils.video_recorder import EpisodeVideoRecorder  # noqa: E402


@dataclass
class EvalConfig:
    checkpoint: Path = Path(
        "/mnt/ssd/checkpoints/resfit_dexjoco_ckpt/bimanual_assembly/forcevla_both/checkpoint_step_050000.pt"
    )
    config: Path = DEXJOco_ROOT / "configs/rand_obj/bimanual_assembly.yaml"
    dataset_root: Path = Path("/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly")
    output: Path | None = None
    seed: int = 0
    port: int = 8000
    host: str = "0.0.0.0"
    force_mode: str = "both"
    episodes: int = 10
    zero_residual: bool = False
    privileged_sim_state: bool | None = None
    rand_full: bool = False
    overwrite: bool = False
    record_video: bool = True
    video_encoder: str = "auto"  # auto | nvenc | cpu


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def _default_output_dir(
    *,
    env_name: str,
    seed: int,
    checkpoint: Path,
    force_mode: str,
    zero_residual: bool,
    rand_full: bool,
) -> Path:
    from dexjoco_lerobot_client.eval_config import resolve_checkpoint_step_label

    suffix = "_rand_full" if rand_full else ""
    force_suffix = f"_{force_mode}" if force_mode else ""
    if zero_residual:
        ckpt_label = "zero_residual"
    else:
        ckpt_label = resolve_checkpoint_step_label(checkpoint)
    return (
        Path("outputs")
        / "resfit_dexjoco"
        / f"{env_name}{suffix}_seed{seed}_{ckpt_label}{force_suffix}"
    )


def main(cfg: EvalConfig) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    _set_seed(cfg.seed)

    import torch

    from resfit_dexjoco.bc.forcevla_client import ForceVLAClient
    from resfit_dexjoco.env.openpi_env import OpenPIEnvConfig, make_openpi_env
    from resfit_dexjoco.env.residual_wrapper import ResidualEnvWrapper
    from resfit_dexjoco.env.rl_obs import rl_obs_spec
    from resfit_dexjoco.training.dataset_stats import load_dataset_stats, load_norm_from_checkpoint
    from resfit_dexjoco.training.proprio_td3 import ProprioResidualTD3
    from resfit_dexjoco.training.train_env import NormScalars, NormalizedTrainEnv

    with open(cfg.config, "r") as f:
        yaml_cfg = yaml.safe_load(f)
    env_name = yaml_cfg["env_name"]
    camera_names = list(yaml_cfg["camera_mapping"].values())

    ckpt = torch.load(cfg.checkpoint.expanduser(), map_location="cpu", weights_only=False)
    privileged = (
        cfg.privileged_sim_state
        if cfg.privileged_sim_state is not None
        else bool(ckpt.get("privileged_sim_state", False))
    )
    obs_spec = rl_obs_spec(privileged_sim_state=privileged)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "norm_stats" in ckpt:
        state_standardizer, action_scaler = load_norm_from_checkpoint(ckpt, device=str(device))
    else:
        print("Checkpoint has no norm_stats; loading from dataset.", flush=True)
        state_standardizer, action_scaler = load_dataset_stats(
            cfg.dataset_root,
            state_dim=obs_spec.state_dim,
            device=str(device),
        )

    def standardize_state(state: np.ndarray) -> np.ndarray:
        t = state_standardizer.standardize(torch.as_tensor(state, device=device, dtype=torch.float32))
        return t.detach().cpu().numpy().astype(np.float32, copy=False)

    def scale_action(action: np.ndarray) -> np.ndarray:
        t = action_scaler.scale(torch.as_tensor(action, device=device, dtype=torch.float32))
        return t.detach().cpu().numpy().astype(np.float32, copy=False)

    def unscale_action(action_n: np.ndarray) -> np.ndarray:
        t = action_scaler.unscale(torch.as_tensor(action_n, device=device, dtype=torch.float32))
        return t.detach().cpu().numpy().astype(np.float64, copy=False)

    agent = ProprioResidualTD3(
        state_dim=obs_spec.state_dim,
        action_dim=obs_spec.action_dim,
        device=device,
    )
    if not cfg.zero_residual:
        agent.actor.load_state_dict(ckpt["agent_actor"])
        agent.actor.eval()

    env_cfg = OpenPIEnvConfig.from_yaml(
        DEXJOco_ROOT / cfg.config,
        seed=cfg.seed,
        rand_full=cfg.rand_full,
        force_mode=cfg.force_mode,  # type: ignore[arg-type]
    )
    openpi_env = make_openpi_env(env_cfg)
    bc = ForceVLAClient(host=cfg.host, port=cfg.port, dual_arm=env_cfg.dual_arm)
    bc.start()
    rollout = ResidualEnvWrapper(openpi_env, bc, reward_mode="milestone")
    eval_env = NormalizedTrainEnv(
        rollout,
        privileged_sim_state=privileged,
        norm=NormScalars(
            scale_action=scale_action,
            unscale_action=unscale_action,
            standardize_state=standardize_state,
        ),
    )

    output_dir = cfg.output or _default_output_dir(
        env_name=env_name,
        seed=cfg.seed,
        checkpoint=cfg.checkpoint,
        force_mode=cfg.force_mode,
        zero_residual=cfg.zero_residual,
        rand_full=cfg.rand_full,
    )
    print(f"Eval output: {output_dir.resolve()}", flush=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        if cfg.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output path {output_dir} already exists and is not empty. "
                "Remove it, pass --overwrite, or set --output."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "zero_residual (BC only)" if cfg.zero_residual else f"residual ckpt step={ckpt.get('step', '?')}"
    print(f"Eval mode: {mode}", flush=True)
    print(f"Checkpoint: {cfg.checkpoint.resolve()}", flush=True)

    num_success = 0
    rewards: list[float] = []
    video_recorder: EpisodeVideoRecorder | None = None

    try:
        for ep in range(cfg.episodes):
            print(f"Episode {ep + 1}/{cfg.episodes}", flush=True)

            if cfg.record_video:
                video_dir = output_dir / f"episode_{ep:02d}_temp"
                video_dir.mkdir(parents=True, exist_ok=True)
                video_recorder = EpisodeVideoRecorder(
                    video_dir,
                    camera_names,
                    encoder=cfg.video_encoder,  # type: ignore[arg-type]
                )

            obs = eval_env.reset()
            if cfg.record_video:
                assert video_recorder is not None
                video_recorder.append(rollout.env.get_raw_images())

            ep_reward = 0.0
            last_info: dict = {}

            while True:
                if cfg.zero_residual:
                    residual_n = np.zeros(obs_spec.action_dim, dtype=np.float32)
                else:
                    state_t = torch.as_tensor(obs["observation.state"], device=device).unsqueeze(0)
                    base_t = torch.as_tensor(obs["observation.base_action"], device=device).unsqueeze(0)
                    with torch.no_grad():
                        residual_n = (
                            agent.act(state_t, base_t, eval_mode=True).squeeze(0).cpu().numpy()
                        )

                obs, result = eval_env.step_residual_normalized(residual_n)
                ep_reward += float(result.reward)
                last_info = result.info

                if cfg.record_video:
                    assert video_recorder is not None
                    video_recorder.append(rollout.env.get_raw_images())

                if result.terminated:
                    break

            eval_env.end_episode(strict_drain=False)
            succeed = bool(last_info.get("succeed"))
            num_success += int(succeed)
            rewards.append(ep_reward)

            if cfg.record_video:
                assert video_recorder is not None
                video_recorder.close()
                video_recorder = None
                result_suffix = "success" if succeed else "failure"
                final_video_dir = output_dir / f"episode_{ep:02d}_{result_suffix}"
                video_dir.rename(final_video_dir)

            print(
                f"  reward={ep_reward:.3f} succeed={succeed} "
                f"milestones={last_info.get('milestones_reached')}",
                flush=True,
            )

        rate = 100.0 * num_success / cfg.episodes
        print(
            f"\nSuccess rate: {num_success}/{cfg.episodes} ({rate:.1f}%) "
            f"mean_reward={float(np.mean(rewards)):.3f}",
            flush=True,
        )
        (output_dir / f"success_rate_{num_success}_{cfg.episodes}.txt").touch()
    finally:
        if video_recorder is not None:
            video_recorder.close()
        eval_env.close()


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
