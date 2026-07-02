import inspect
import functools
import time
import os
import pytz

from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from absl import logging
from typing import Any, Callable, Optional, Tuple
import tqdm
import tyro
import wandb
import numpy as np
import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env

WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")

from brax.training.agents.ppo.networks import make_ppo_networks

import track_mj as tmj
import track_mj.envs.assembly_tracking  # noqa: F401 — register AssemblyTrackingGeneral
from track_mj import update_file_handler
from track_mj.paths import checkpoint_dir, train_log_dir, wandb_dir
from track_mj.envs.assembly_tracking.train.base_env import AssemblyEnv
from track_mj.learning.policy.ppo import train_tracking as ppo
from track_mj.envs.assembly_tracking.utils.wrapper import wrap_fn

from track_mj.eval.tracking.run_eval import (
    AssemblyTrackingEvaluator,
    EVAL_MAX_STEPS,
    EVAL_NUM_ENVS,
)

@dataclass
class Args:
    task: str
    exp_name: str = "debug"
    exp_tags: str = None
    exp_notes: str = None
    seed: int = 42
    convert_onnx: bool = True
    
    # ====== policy ======
    num_timesteps: int = 2_000_000_000
    restore_checkpoint_path: Optional[str] = None
    resume_ckpt_dir: Optional[str] = None

    obs_noise_level: float = 1.0
    history_len: int = 0


def _prepare_exp_name(task: str, exp_name: str) -> str:
    r"""
    timestamp_task_expname
    """
    cst_time = datetime.now(pytz.timezone('Asia/Shanghai'))
    timestamp = cst_time.strftime("%m%d%H%M")
    return f"{timestamp}_{task}_{exp_name}"

def _parse_exp_tags(tags):
    r"""
    Parse tags like `"'[tag1, tag2]'" into a list.
    """
    if isinstance(tags, list):
        return tags
    if isinstance(tags, str):
        cleaned = tags.strip()
        if (cleaned.startswith('[') and cleaned.endswith(']')) or \
           (cleaned.startswith('(') and cleaned.endswith(')')) or \
           (cleaned.startswith('"') and cleaned.endswith('"')) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1]
        # Handle quoted tags
        result = []
        for tag in cleaned.split(','):
            tag = tag.strip()
            if tag.startswith('"') and tag.endswith('"') or \
               tag.startswith("'") and tag.endswith("'"):
                tag = tag[1:-1]
            if tag:  # Ensure no empty tags are added
                result.append(tag)
        return result
    return [str(tags)]

def _validate_exp_name_format(exp_name: str, debug_mode: bool):
    if not debug_mode and len(exp_name.split("_")) != 4:
        raise ValueError(f"exp_name should be in the format <task>_<tag>_<version>, got {exp_name}")


def _apply_policy_args_to_config(args: Args, cfg, debug: bool, smoke: bool):
    cfg.num_timesteps = args.num_timesteps
    if smoke:
        cfg.training_metrics_steps = 10
        cfg.num_evals = 0
        cfg.batch_size = 2
        cfg.num_minibatches = 1
        cfg.num_envs = cfg.batch_size * cfg.num_minibatches
        cfg.episode_length = 50
        cfg.unroll_length = 2
        cfg.num_updates_per_batch = 1
        cfg.action_repeat = 1
        cfg.num_resets_per_eval = 1
        if args.num_timesteps == 0:
            cfg.num_timesteps = 0
        else:
            cfg.num_timesteps = min(args.num_timesteps, 200)
        return
    if debug:
        cfg.training_metrics_steps = 160  # 每 1 次 PPO 更新
        cfg.num_evals = 3  # 2 段 checkpoint + eval
        cfg.batch_size = 8
        cfg.num_minibatches = 2
        cfg.num_envs = cfg.batch_size * cfg.num_minibatches
        cfg.episode_length = 200
        cfg.unroll_length = 10
        cfg.num_updates_per_batch = 1
        cfg.action_repeat = 1
        cfg.num_resets_per_eval = 1
        if args.num_timesteps == 0:
            cfg.num_timesteps = 0
        else:
            cfg.num_timesteps = 640  # 2×segment（160 step/次 × 2 step/段 × 2 段）
        return

def _apply_env_args_to_config(args: Args, cfg):
    cfg.history_len = args.history_len
    cfg.noise_config.level = args.obs_noise_level

    cfg.obs_keys = sorted(list(set(cfg.obs_keys)))
    cfg.privileged_obs_keys = sorted(list(set(cfg.privileged_obs_keys)))

    print("Final obs keys:", cfg.obs_keys)
    print("Final privileged obs keys:", cfg.privileged_obs_keys)


def _enable_debug_mode():
    jax.config.update("jax_traceback_filtering", "off")

def _setup_paths(exp_name: str) -> tuple[Path, Path]:
    logdir = train_log_dir(exp_name)
    logdir.mkdir(parents=True, exist_ok=True)
    update_file_handler(filename=f"{logdir}/info.log")
    ckpt_path = checkpoint_dir(exp_name)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    return logdir, ckpt_path

def _log_checkpoint_path(ckpt_path: Path):
    logging.info(f"Checkpoint path: {ckpt_path}")

def _prepare_training_params(cfg, ckpt_path: Path):
    params = cfg.to_dict()
    params.pop("network_factory", None)
    params["wrap_env_fn"] = wrap_fn
    network_fn = make_ppo_networks
    params["network_factory"] = (
        functools.partial(network_fn, **cfg.network_factory) if hasattr(cfg, "network_factory") else network_fn
    )
    params["save_checkpoint_path"] = ckpt_path
    return params

def _init_wandb(args: Args, exp_name, env_class, task_cfg, ckpt_path, config_fname="config.json"):
    wandb.init(
        name=exp_name,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group="Track",
        config={
            "num_timesteps": args.num_timesteps,
            "task": args.task,
            "group": "Track",
        },
        dir=str(wandb_dir()),
        tags=_parse_exp_tags(args.exp_tags),
        notes=args.exp_notes,
    )
    wandb.config.update(task_cfg.to_dict())
    wandb.save(inspect.getfile(env_class))
    config_path = ckpt_path / config_fname
    config_path.write_text(task_cfg.to_json_best_effort(indent=4))


def _make_progress_fn(total_steps: int, debug_mode: bool, times: list, initial_step: int = 0):
    """Return progress callback with tqdm bar + wandb + ETA logging."""
    pbar = tqdm.tqdm(
        total=total_steps,
        initial=initial_step,
        unit="env_step",
        desc="PPO",
        dynamic_ncols=True,
        mininterval=5.0,
    )

    def progress_fn(num_steps: int, metrics: dict):
        now = time.monotonic()
        times.append(now)
        if num_steps > pbar.n:
            pbar.update(num_steps - pbar.n)
        postfix = {}
        if metrics:
            for key in ("training/sps", "episode/rew", "episode/reward"):
                if key in metrics:
                    postfix[key.split("/")[-1]] = f"{float(metrics[key]):.2f}"
        if postfix:
            pbar.set_postfix(postfix, refresh=False)
        pbar.refresh()

        if metrics and not debug_mode:
            try:
                wandb.log(metrics, step=num_steps)
            except Exception as e:
                logging.warning(f"wandb.log failed: {e}")

        for key, val in (metrics or {}).items():
            if "loss" in key:
                try:
                    if np.isnan(float(val)):
                        logging.warning("NaN metric %s at step %s", key, num_steps)
                except (TypeError, ValueError):
                    pass

        if len(times) < 2 or num_steps == 0:
            return
        sps = float(metrics.get("training/sps", 0)) if metrics else 0.0
        if sps > 0:
            est_seconds_left = (total_steps - num_steps) / sps
        else:
            step_times = np.diff(times)
            median_step_time = np.median(step_times)
            if median_step_time <= 0:
                return
            steps_logged = num_steps / len(step_times)
            est_seconds_left = (total_steps - num_steps) / steps_logged * median_step_time
        pct = 100.0 * num_steps / total_steps
        logging.info(
            f"Progress {num_steps}/{total_steps} ({pct:.2f}%) - EstTimeLeft {est_seconds_left / 3600:.1f}[h]"
        )

    progress_fn.close = pbar.close  # type: ignore[attr-defined]
    return progress_fn

def _report_training_time(times):
    if len(times) > 1:
        logging.info("Done training.")
        logging.info(f"Time to JIT compile: {times[1] - times[0]:.2f}s")
        logging.info(f"Time to train: {times[-1] - times[1]:.2f}s")


def get_trajectory_handler(env, args: Args):
    # load reference trajectory
    trajectory_data = env.prepare_trajectory(env._config.reference_traj_config.name)
    obs_size = env.observation_size
    act_size = env.action_size
    env.th.traj = None

    # output the dataset and observation info of general tracker
    print("=" * 50)
    print(
        f"Tracking {len(trajectory_data.split_points) - 1} trajectories with {trajectory_data.qpos.shape[0]} timesteps, fps={1 / env.dt:.1f}"
    )
    print(f"Observation: {env._config.obs_keys}")
    print(f"Privileged state: {env._config.privileged_obs_keys}")
    print("=" * 50)

    return trajectory_data, obs_size, act_size


def _make_eval_env(training_env, env_class, env_cfg, policy_cfg, num_eval_envs: int):
    eval_env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)
    eval_env.th = training_env.th
    key = jax.random.PRNGKey(123)
    return ppo._maybe_wrap_env(
        eval_env,
        wrap_env=True,
        num_envs=num_eval_envs,
        episode_length=env_cfg.episode_length,
        action_repeat=policy_cfg.action_repeat,
        local_device_count=1,
        key_env=key,
        wrap_env_fn=wrap_fn,
        randomization_fn=None,
    )


def _build_eval_make_policy(eval_env, policy_cfg, trajectory_data):
    """Build inference fn with same network layout as training (for eval JIT warmup)."""
    from brax.training.acme import running_statistics, specs
    from brax.training.agents.ppo import networks as ppo_networks

    key_envs = jax.random.split(jax.random.PRNGKey(0), EVAL_NUM_ENVS)
    state = jax.jit(eval_env.reset)(key_envs, trajectory_data)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], state.obs)
    nf = policy_cfg.network_factory
    network_factory = functools.partial(make_ppo_networks, **dict(nf))
    normalize = lambda x, y: x
    if policy_cfg.normalize_observations:
        normalize = running_statistics.normalize
    ppo_network = network_factory(
        observation_size=obs_shape,
        action_size=eval_env.action_size,
        preprocess_observations_fn=normalize,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    key_policy, key_value = jax.random.split(jax.random.PRNGKey(0))
    policy_params = ppo_network.policy_network.init(key_policy)
    value_params = ppo_network.value_network.init(key_value)
    obs_shape_norm = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jp.dtype("float32")), state.obs
    )
    normalizer_params = running_statistics.init_state(obs_shape_norm)
    # Brax make_policy expects (normalizer, policy, value), same as train_tracking checkpoints.
    eval_init_params = (normalizer_params, policy_params, value_params)
    return make_policy, eval_init_params


def _make_policy_params_fn(
    evaluator: AssemblyTrackingEvaluator,
    progress_fn,
    debug_mode: bool,
):
    def policy_params_fn(current_step: int, make_policy, params):
        del make_policy
        logging.info("Checkpoint saved at step %s — running eval...", current_step)
        try:
            eval_metrics = evaluator.evaluate(params, seed=current_step)
            prefixed = {f"eval/{k}": v for k, v in eval_metrics.items()}
            progress_fn(current_step, prefixed)
            if not debug_mode:
                wandb.log(prefixed, step=current_step)
        except Exception as e:
            logging.warning("Eval failed at step %s: %s", current_step, e)

    return policy_params_fn


def train(args: Args):
    env_class = tmj.registry.get(args.task, "tracking_train_env_class")
    task_cfg = tmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config

    exp_name = _prepare_exp_name(args.task, args.exp_name)
    smoke_mode = "smoke" in exp_name
    debug_mode = "debug" in exp_name and not smoke_mode

    if debug_mode:
        _enable_debug_mode()

    logdir, ckpt_path = _setup_paths(exp_name)
    restored_step = 0
    if args.resume_ckpt_dir:
        ckpt_path = Path(args.resume_ckpt_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        if args.restore_checkpoint_path is None:
            from track_mj.eval.tracking.run_eval import get_latest_ckpt

            args.restore_checkpoint_path = get_latest_ckpt(ckpt_path)
    if args.restore_checkpoint_path:
        step_name = Path(args.restore_checkpoint_path).name
        if step_name.isdigit():
            restored_step = int(step_name)
    _log_checkpoint_path(ckpt_path)

    _apply_policy_args_to_config(args, policy_cfg, debug_mode, smoke_mode)
    policy_cfg.restore_checkpoint_path = args.restore_checkpoint_path
    if restored_step > 0:
        logging.info("Will resume training from env step %s", restored_step)
    _apply_env_args_to_config(args, env_cfg)
    if debug_mode and args.task == "AssemblyTrackingGeneral":
        from track_mj.envs.assembly_tracking import assembly_tracking_constants as asm_consts

        env_cfg.reference_traj_config.name = {asm_consts.TASK_ID: asm_consts.default_trajectory_names(10, "full")}
        print("Debug mode: 10 trajectories, num_evals=3, checkpoint+eval each segment")
    if smoke_mode and args.task == "AssemblyTrackingGeneral":
        from track_mj.envs.assembly_tracking import assembly_tracking_constants as asm_consts

        env_cfg.reference_traj_config.name = {asm_consts.TASK_ID: ["ep000_full"]}
        env_cfg.episode_length = min(env_cfg.episode_length, 50)
        print("Smoke mode: 2 envs, batch 2x1, unroll 2, 1 trajectory (ep000_full)")

    if args.task == "G1TrackingGeneralTerrainDR":
        from track_mj.dr.domain_randomize_tracking import domain_randomize_terrain

        hfield_data = jp.asarray(np.load("storage/data/hfield/terrain.npz")["hfield_data"])
        policy_cfg.randomization_fn = functools.partial(domain_randomize_terrain, all_hfield_data=hfield_data)
        del hfield_data
        assert env_cfg.terrain_type == "rough_terrain"
    elif args.task == "G1TrackingGeneralDR":
        from track_mj.dr.domain_randomize_tracking import domain_randomize

        assert policy_cfg.randomization_fn == domain_randomize
    elif args.task == "G1TrackingGeneral":
        assert policy_cfg.randomization_fn == None
    elif args.task == "AssemblyTrackingGeneral":
        assert policy_cfg.randomization_fn is None
    else:
        pass

    policy_params = _prepare_training_params(policy_cfg, ckpt_path)

    if not debug_mode:
        _init_wandb(args, exp_name, env_class, task_cfg, ckpt_path)

    train_fn = functools.partial(ppo.train, **policy_params)
    times = [time.monotonic()]
    progress_fn = _make_progress_fn(policy_cfg.num_timesteps, debug_mode, times, restored_step)

    env: AssemblyEnv = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)

    trajectory_data, obs_size, act_size = get_trajectory_handler(env, args)
    env.th.traj = None

    eval_envs = EVAL_NUM_ENVS
    eval_steps = EVAL_MAX_STEPS
    eval_env = _make_eval_env(env, env_class, env_cfg, policy_cfg, eval_envs)
    make_eval_policy, eval_init_params = _build_eval_make_policy(eval_env, policy_cfg, trajectory_data)
    evaluator = AssemblyTrackingEvaluator(
        eval_env,
        trajectory_data,
        make_eval_policy,
        num_envs=eval_envs,
        max_steps=eval_steps,
    )
    evaluator.warmup(eval_init_params)
    policy_params_fn = _make_policy_params_fn(evaluator, progress_fn, debug_mode)

    try:
        make_inference_fn, params, _ = train_fn(
            environment=env,
            trajectory_data=trajectory_data,
            progress_fn=progress_fn,
            policy_params_fn=policy_params_fn,
        )
    finally:
        progress_fn.close()

    _report_training_time(times)
    inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    # eval_env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)
    # _run_evaluation(args.task, task_cfg, eval_env, inference_fn, debug_mode)
    logging.info(f"Run {exp_name} Train done.")

    if args.convert_onnx:
        env.prepare_trajectory(env._config.reference_traj_config.name)

        try:
            from track_mj.eval.tracking.brax2onnx import convert_jax2onnx, get_latest_ckpt

            ckpt_dir = get_latest_ckpt(ckpt_path)
            policy_obs_key = policy_cfg.network_factory.policy_obs_key
            convert_jax2onnx(
                ckpt_dir=ckpt_dir,
                output_path=f"{ckpt_dir}/policy.onnx",
                inference_fn=inference_fn,
                hidden_layer_sizes=policy_cfg.network_factory.policy_hidden_layer_sizes,
                obs_size=obs_size,
                action_size=act_size,
                policy_obs_key=policy_obs_key,
                jax_params=params,
                activation="swish",
            )
        except ImportError:
            logging.warning("TensorFlow is not installed. Please install TensorFlow to use ONNX conversion.")


if __name__ == "__main__":
    train(tyro.cli(Args))
