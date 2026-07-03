"""Panda bimanual assembly reference tracking env (OpenTrack-style)."""

from __future__ import annotations

from functools import partial
from dataclasses import replace
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from tqdm import tqdm

import track_mj as tmj
from track_mj.envs.assembly_tracking import assembly_tracking_constants as consts
from track_mj.envs.assembly_tracking.train import base_env as assembly_base
from track_mj.paths import mocap_dir
from track_mj.utils.dataset.traj_class import Trajectory, TrajectoryData
from track_mj.utils.dataset.traj_handler import TrajectoryHandler, TrajCarry
from mujoco_playground._src import mjx_env

EPISODE_LENGTH = 800
CTRL_DT = 1.0 / 30.0
_SHARED_TRAJ_METADATA = {"source": "dexjoco_zarr_replay", "task": consts.TASK_ID}


def _with_shared_traj_metadata(traj: Trajectory) -> Trajectory:
    return Trajectory(info=replace(traj.info, metadata=_SHARED_TRAJ_METADATA), data=traj.data)


def assembly_tracking_general_task_config() -> config_dict.ConfigDict:
    env_config = config_dict.create(
        terrain_type="flat_terrain",
        ctrl_dt=CTRL_DT,
        sim_dt=0.002,
        episode_length=EPISODE_LENGTH,
        action_repeat=1,
        action_scale=0.25,
        recalculate_vel_in_reward=False,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        reference_traj_config=config_dict.create(
            name={consts.TASK_ID: consts.default_trajectory_names(100, "full")},
            random_start=True,
            fixed_start_frame=0,
        ),
        termination_config=config_dict.create(
            joint_dif_threshold=3.0,
            object_pos_threshold=0.15,
        ),
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.02,
                joint_vel=1.0,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                joint_pos_tracking=1.0,
                joint_vel_tracking=0.3,
                object_pos_tracking=0.5,
                penalty_action_rate=-0.2,
                penalty_torque=-1e-5,
                dof_pos_limit=-5.0,
                termination=-100.0,
            ),
            auxiliary=config_dict.create(
                joint_pos_sigma=5.0,
                joint_vel_sigma=1.0,
                object_pos_sigma=0.05,
            ),
        ),
        obs_scales_config=config_dict.create(joint_vel=0.05, dif_joint_vel=0.05),
        obs_keys=[
            "dif_joint_pos",
            "dif_joint_vel",
            "joint_pos",
            "joint_vel",
            "last_motor_targets",
        ],
        privileged_obs_keys=[
            "joint_pos",
            "joint_vel",
            "dif_joint_pos",
            "dif_joint_vel",
            "dif_object_pos",
            "last_motor_targets",
        ],
        history_keys=["joint_pos", "joint_vel"],
    )

    policy_config = config_dict.create(
        num_timesteps=200_000_000,
        max_devices_per_host=1,
        wrap_env=True,
        num_envs=512,
        episode_length=EPISODE_LENGTH,
        action_repeat=1,
        wrap_env_fn=None,
        randomization_fn=None,
        learning_rate=3e-4,
        entropy_cost=0.01,
        discounting=0.97,
        unroll_length=20,
        batch_size=128,
        num_minibatches=4,
        num_updates_per_batch=4,
        num_resets_per_eval=0,
        normalize_observations=False,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        normalize_advantage=True,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 512, 256, 128),
            value_hidden_layer_sizes=(512, 512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        seed=0,
        num_evals=201,  # 200 段 checkpoint（每段 ~1M env step，约 24h/次 @12 step/s）
        log_training_metrics=True,
        training_metrics_steps=10_240,  # 每 1 次 PPO 更新打 log / 刷新进度
        progress_fn=lambda *args: None,
        save_checkpoint_path=None,
        restore_checkpoint_path=None,
        restore_params=None,
        restore_value_fn=True,
    )

    return config_dict.create(env_config=env_config, policy_config=policy_config)


tmj.registry.register("AssemblyTrackingGeneral", "tracking_config")(assembly_tracking_general_task_config())


def hybrid_control_step(
    rng: jax.Array,
    model: mjx.Model,
    data: mjx.Data,
    qpos_des: jax.Array,
    kps: jax.Array,
    kds: jax.Array,
    torque_limit: jax.Array,
    motor_mask: jax.Array,
    n_substeps: int = 1,
) -> tuple[jax.Array, mjx.Data, jax.Array]:
    """PD torque on Panda motors; position targets on Allegro actuators."""

    def single_step(carry, _):
        rng, data, _ = carry
        pos_err = qpos_des - data.qpos[consts.CTRL_QPOS_SLICE]
        vel_err = -data.qvel[consts.CTRL_QVEL_SLICE]
        torque = kps * pos_err + kds * vel_err
        torque = jp.clip(torque, -torque_limit, torque_limit)
        ctrl = jp.where(motor_mask, torque, qpos_des)
        data = data.replace(ctrl=ctrl)
        data = mjx.step(model, data)
        return (rng, data, torque), None

    initial_torque = jp.zeros_like(torque_limit)
    (final_rng, final_data, final_torque), _ = jax.lax.scan(
        single_step, (rng, data, initial_torque), (), n_substeps
    )
    return final_rng, final_data, final_torque


@tmj.registry.register("AssemblyTrackingGeneral", "tracking_train_env_class")
class AssemblyTrackingGeneralEnv(assembly_base.AssemblyEnv):
    @property
    def action_size(self) -> int:
        return consts.NUM_CTRL_JOINTS

    def __init__(
        self,
        terrain_type: str = "flat_terrain",
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.task_to_xml(terrain_type).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
      self.num_joints = consts.NUM_CTRL_JOINTS
      self.episode_length = self._config.episode_length
      self.action_joint_ids = jp.arange(self.num_joints)
      self._motor_mask = jp.array(consts.MOTOR_ACTUATOR_MASK)
      self._kps = jp.array(consts.KPs)
      self._kds = jp.array(consts.KDs)
      self._default_qpos = jp.array(consts.DEFAULT_QPOS)
      self.torque_limit = jp.array(consts.TORQUE_LIMIT)

      self._track_body_ids = jp.array(
          [self.mj_model.body(name).id for name in consts.TRACK_BODY_NAMES],
          dtype=jp.int32,
      )

      lowers, uppers = [], []
      for j_name in consts.ACTION_JOINT_NAMES:
          jnt = self.mj_model.joint(j_name)
          if jnt.limited:
              lowers.append(jnt.range[0])
              uppers.append(jnt.range[1])
          else:
              lowers.append(-jp.pi)
              uppers.append(jp.pi)
      self._lowers = jp.array(lowers)
      self._uppers = jp.array(uppers)
      c = (self._lowers + self._uppers) / 2
      r = self._uppers - self._lowers
      self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
      self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

      self.th: TrajectoryHandler | None = None
      if self._config.reference_traj_config.random_start:
          self._th_params = {"random_start": True}
      else:
          self._th_params = {
              "random_start": False,
              "fixed_start_conf": [0, self._config.reference_traj_config.fixed_start_frame],
          }
      self._data = mujoco.MjData(self._mj_model)

    def reset(self, rng: jax.Array, trajectory_data: TrajectoryData | None = None) -> mjx_env.State:
      if trajectory_data is None:
          trajectory_data = self.th.traj.data
      rng, noise_rng = jax.random.split(rng, 2)

      carry = self.th.reset_state_with_trajectory(trajectory_data, TrajCarry(rng, self.th.init_state()))
      init_traj_data = self.th.get_current_traj_data_with_trajectory(trajectory_data, carry)

      data = mjx_env.init(
          self.mjx_model,
          qpos=init_traj_data.qpos,
          qvel=init_traj_data.qvel,
          ctrl=init_traj_data.qpos[consts.CTRL_QPOS_SLICE],
      )

      carry = self.th.update_state_with_trajectory(trajectory_data, carry)
      traj_data = self.th.get_current_traj_data_with_trajectory(trajectory_data, carry)

      info = {
          "rng": carry.key,
          "step": 0,
          "last_motor_targets": data.qpos[consts.CTRL_QPOS_SLICE],
          "last_action": jp.zeros(self.action_size),
          "last_dof_pos": data.qpos[consts.CTRL_QPOS_SLICE],
          "last_joint_vel": jp.zeros(self.num_joints),
          "traj_no": carry.traj_state.traj_no,
          "traj_info": carry,
      }

      metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
      obs, history = self._get_obs(data, traj_data, info, noise_rng)
      if self._config.history_len > 0:
          _, init_history = self._get_obs(data, init_traj_data, info, noise_rng)
          info["previous_obs"] = jp.stack([init_history] * self._config.history_len, axis=0)
          obs["state"] = jp.concatenate([obs["state"], info["previous_obs"].flatten()], axis=0)
          obs["privileged_state"] = jp.concatenate(
              [obs["privileged_state"], info["previous_obs"].flatten()], axis=0
          )
          info["previous_obs"] = jp.concatenate([info["previous_obs"][1:], history[None, :]], axis=0)

      return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(
      self, state: mjx_env.State, action: jax.Array, trajectory_data: TrajectoryData | None = None
  ) -> mjx_env.State:
      if trajectory_data is None:
          trajectory_data = self.th.traj.data

      traj_data = self.th.get_current_traj_data_with_trajectory(trajectory_data, state.info["traj_info"])
      ref_q = traj_data.qpos[consts.CTRL_QPOS_SLICE]
      motor_targets = ref_q + action * self._config.action_scale
      motor_targets = jp.clip(motor_targets, self._soft_lowers, self._soft_uppers)

      state.info["rng"], data, torque = hybrid_control_step(
          state.info["rng"],
          self.mjx_model,
          state.data,
          motor_targets,
          kps=self._kps,
          kds=self._kds,
          torque_limit=self.torque_limit,
          motor_mask=self._motor_mask,
          n_substeps=self.n_substeps,
      )

      sim_invalid = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
      rewards = self._get_reward(data, traj_data, action, motor_targets, torque, state.info)
      rewards = {k: jp.nan_to_num(v * self._config.reward_config.scales[k], nan=0.0) for k, v in rewards.items()}
      reward = jp.clip(sum(rewards.values()) * self.dt, a_max=10000.0)
      reward = jp.where(sim_invalid, -1.0, reward)
      for k, v in rewards.items():
          state.metrics[f"reward/{k}"] = v

      state.info["step"] += 1
      state.info["last_motor_targets"] = motor_targets
      state.info["last_action"] = action
      state.info["last_dof_pos"] = data.qpos[consts.CTRL_QPOS_SLICE]
      state.info["last_joint_vel"] = data.qvel[consts.CTRL_QVEL_SLICE]

      termination = self._get_termination(data, traj_data) | sim_invalid
      state.info["traj_info"] = self.th.update_state_with_trajectory(trajectory_data, state.info["traj_info"])
      truncated = state.info["step"] >= self._config.episode_length
      done = (termination | truncated).astype(reward.dtype)
      state.info["step"] = jp.where(done, 0, state.info["step"])

      state.info["rng"], noise_rng = jax.random.split(state.info["rng"])
      obs, history = self._get_obs(data, traj_data, state.info, noise_rng)
      if self._config.history_len > 0:
          obs["state"] = jp.concatenate([obs["state"], state.info["previous_obs"].flatten()], axis=0)
          obs["privileged_state"] = jp.concatenate(
              [obs["privileged_state"], state.info["previous_obs"].flatten()], axis=0
          )
          state.info["previous_obs"] = jp.concatenate(
              [state.info["previous_obs"][1:], history[None, :]], axis=0
          )

      state = state.replace(data=data, obs=obs, reward=reward, done=done)
      state = jax.lax.cond(
          done, partial(self._reset_and_update_state, trajectory_data=trajectory_data), lambda x: x, state
      )
      return state

    def _reset_and_update_state(
        self, current_state: mjx_env.State, trajectory_data: TrajectoryData
    ) -> mjx_env.State:
        """Merge reset into current state (keeps wrapper info keys for lax.cond)."""
        reset_rng, current_state.info["rng"] = jax.random.split(current_state.info["rng"])
        reset_state = self.reset(reset_rng, trajectory_data)
        current_state = current_state.replace(data=reset_state.data, obs=reset_state.obs)
        for key in reset_state.info.keys():
            current_state.info[key] = reset_state.info[key]
        return current_state

    def _get_termination(self, data: mjx.Data, traj_data: TrajectoryData) -> jax.Array:
      dif_joint = jp.abs(traj_data.qpos[consts.CTRL_QPOS_SLICE] - data.qpos[consts.CTRL_QPOS_SLICE])
      joint_term = jp.max(dif_joint) > self._config.termination_config.joint_dif_threshold
      dif_obj = jp.linalg.norm(
          traj_data.xpos[self._track_body_ids[:2]] - data.xpos[self._track_body_ids[:2]], axis=-1
      )
      object_term = jp.any(dif_obj > self._config.termination_config.object_pos_threshold)
      return joint_term | object_term | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    def _get_obs(
      self,
      data: mjx.Data,
      traj_data: TrajectoryData,
      info: dict[str, Any],
      noise_rng: jax.Array,
  ) -> tuple[mjx_env.Observation, jax.Array]:
      joint_pos = data.qpos[consts.CTRL_QPOS_SLICE]
      joint_vel = data.qvel[consts.CTRL_QVEL_SLICE]
      dif_joint_pos = traj_data.qpos[consts.CTRL_QPOS_SLICE] - joint_pos
      dif_joint_vel = traj_data.qvel[consts.CTRL_QVEL_SLICE] - joint_vel
      dif_object_pos = (traj_data.xpos[self._track_body_ids[:2]] - data.xpos[self._track_body_ids[:2]]).flatten()

      noisy_joint_pos = joint_pos + (
          (2 * jax.random.uniform(noise_rng, shape=joint_pos.shape) - 1)
          * self._config.noise_config.level
          * self._config.noise_config.scales.joint_pos
      )

      state_dict = {
          "dif_joint_pos": dif_joint_pos,
          "dif_joint_vel": dif_joint_vel * self._config.obs_scales_config.dif_joint_vel,
          "joint_pos": noisy_joint_pos,
          "joint_vel": joint_vel * self._config.obs_scales_config.joint_vel,
          "last_motor_targets": info["last_motor_targets"],
          "dif_object_pos": dif_object_pos,
      }

      state = jp.concatenate([state_dict[k] for k in self._config.obs_keys], axis=-1)
      privileged = jp.concatenate([state_dict[k] for k in self._config.privileged_obs_keys], axis=-1)
      history = jp.concatenate([state_dict[k] for k in self._config.history_keys], axis=-1)
      state = jp.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
      privileged = jp.nan_to_num(privileged, nan=0.0, posinf=1e6, neginf=-1e6)
      return {"state": state, "privileged_state": privileged}, history

    def _get_reward(
      self,
      data: mjx.Data,
      traj_data: TrajectoryData,
      action: jax.Array,
      motor_targets: jax.Array,
      torque: jax.Array,
      info: dict[str, Any],
  ) -> dict[str, jax.Array]:
      dif_joint_pos = traj_data.qpos[consts.CTRL_QPOS_SLICE] - data.qpos[consts.CTRL_QPOS_SLICE]
      dif_joint_vel = traj_data.qvel[consts.CTRL_QVEL_SLICE] - data.qvel[consts.CTRL_QVEL_SLICE]
      dif_object = traj_data.xpos[self._track_body_ids[:2]] - data.xpos[self._track_body_ids[:2]]
      object_dist = jp.sum(dif_object**2)

      aux = self._config.reward_config.auxiliary
      return {
          "joint_pos_tracking": jp.exp(-jp.sum(jp.abs(dif_joint_pos)) / aux.joint_pos_sigma),
          "joint_vel_tracking": jp.exp(-jp.sum(jp.abs(dif_joint_vel)) / aux.joint_vel_sigma),
          "object_pos_tracking": jp.exp(-object_dist / aux.object_pos_sigma),
          "penalty_action_rate": jp.sum(jp.square(motor_targets - info["last_motor_targets"])),
          "penalty_torque": jp.nan_to_num(jp.sum(jp.square(torque)), nan=0.0, posinf=1e6, neginf=1e6),
          "dof_pos_limit": jp.sum(jp.maximum(0.0, self._soft_lowers - data.qpos[consts.CTRL_QPOS_SLICE]))
          + jp.sum(jp.maximum(0.0, data.qpos[consts.CTRL_QPOS_SLICE] - self._soft_uppers)),
          "termination": self._get_termination(data, traj_data).astype(jp.float32),
      }

    def load_trajectory(self, traj: Trajectory | None = None, warn: bool = True) -> TrajectoryHandler:
      th_params = self._th_params if self._th_params is not None else {}
      return TrajectoryHandler(model=self._mj_model, warn=warn, traj=traj, control_dt=self.dt, **th_params)

    def prepare_trajectory(self, dataset_dict: Dict[str, List[str]]) -> TrajectoryData:
      import jax.numpy as jnp

      all_trajectories: list[Trajectory] = []
      for dataset_name, traj_names in dataset_dict.items():
          print(f"Loading dataset: {dataset_name} with {len(traj_names)} trajectories.")
          dataset_path = mocap_dir(dataset_name, consts.ROBOT_SUBDIR)
          for t_name in tqdm(traj_names):
              traj_path = dataset_path / f"{t_name}.npz"
              if not traj_path.exists():
                  raise FileNotFoundError(f"Motion file {traj_path} not found.")
              traj = Trajectory.load(str(traj_path), backend=jnp)
              all_trajectories.append(_with_shared_traj_metadata(traj))
              print(f"Loaded trajectory {t_name}")

      if len(all_trajectories) == 1:
          trajectory = all_trajectories[0]
      else:
          traj_datas = [t.data for t in all_trajectories]
          traj_infos = [t.info for t in all_trajectories]
          traj_data, traj_info = TrajectoryData.concatenate(traj_datas, traj_infos, backend=jnp)
          trajectory = Trajectory(traj_info, traj_data)

      self.th = self.load_trajectory(trajectory, warn=False)
      return trajectory.data

    @property
    def observation_size(self) -> dict[str, int]:
        n = self.num_joints
        sizes = {
            "dif_joint_pos": n,
            "dif_joint_vel": n,
            "joint_pos": n,
            "joint_vel": n,
            "last_motor_targets": n,
            "dif_object_pos": 6,
        }
        return {
            "state": sum(sizes[k] for k in self._config.obs_keys),
            "privileged_state": sum(sizes[k] for k in self._config.privileged_obs_keys),
        }
