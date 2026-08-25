from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dexjoco.tasks.bimanual_assembly.config import TaskConfig
from interaction_retarget.sim.contact import AssemblyContactDetector

from .causal import CausalAssemblyController


class BimanualPhysicsRLEnv(gym.Env):
    """Fast privileged-state RL wrapper around DexJoCo's native task."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int = 0,
        randomize: bool = False,
        randomize_dynamics: bool = False,
        translation_scale: float = 0.01,
        rotation_scale: float = 0.05,
        shaping_gamma: float = 0.995,
        root_bank: str | Path | None = None,
        root_max_offset: int | None = None,
        episode_steps: int | None = None,
        root_noise: float = 0.0,
        residual_cost: float = 1e-3,
        online_bias_correction: bool = False,
        causal_templates: str | Path | None = None,
        causal_warm_start: bool = False,
    ) -> None:
        super().__init__()
        if episode_steps is not None and episode_steps < 1:
            raise ValueError("episode_steps must be positive")
        if root_noise < 0 or residual_cost < 0:
            raise ValueError("root_noise and residual_cost must be non-negative")
        if root_bank is not None and causal_templates is not None:
            raise ValueError("root_bank and causal_templates are mutually exclusive")

        self._env = TaskConfig().get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            image_obs=False,
            randomize=randomize,
            randomize_dynamics=randomize_dynamics,
            seed=seed,
        )
        self.raw = self._env.unwrapped
        self._contact_detector = AssemblyContactDetector(self.raw)

        # The native 30 Hz sleep and reset-time camera priming are for teleoperation.
        self.raw.hz = 0
        self.raw._prime_rgb_array_renderer = lambda: None

        self.translation_scale = float(translation_scale)
        self.rotation_scale = float(rotation_scale)
        self.shaping_gamma = float(shaping_gamma)
        self.root_noise = float(root_noise)
        self.residual_cost = float(residual_cost)
        self.online_bias_correction = bool(online_bias_correction)
        self._causal_templates = Path(causal_templates) if causal_templates else None
        self._causal_warm_start = bool(causal_warm_start)
        self._causal_controller: CausalAssemblyController | None = None
        self._last_base_action = np.zeros(46, dtype=np.float32)
        self._causal_phase = 0.0
        self._root_rng = np.random.default_rng(seed)
        self._roots = self._load_roots(root_bank, root_max_offset) if root_bank else None
        self._episode_limit = episode_steps if episode_steps is not None else (300 if self._roots else None)
        self._episode_steps = 0
        self._grasp_loss_steps = 0
        self._root_info: dict[str, int] = {}
        self._root_index = 0
        self._reference_pos_bias = np.zeros((2, 3), dtype=np.float64)
        self._estimated_reference_bias = np.zeros((2, 3), dtype=np.float64)
        self._last_reference_pos = np.zeros((2, 3), dtype=np.float64)
        self._last_position_delta = np.zeros((2, 3), dtype=np.float64)
        self._has_reference_command = False
        action_size = 3 if self._causal_templates is not None else 12
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(action_size,), dtype=np.float32
        )

        obs_size = (
            self.raw._model.nq
            + self.raw._model.nv
            + self.raw._model.nu
            + 2 * 3
            + 2 * 4
            + 9
            + 15
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(obs_size,), dtype=np.float32
        )

        peg_bottom_z = (
            self.raw._model.geom_pos[self.raw._peg_geom_id, 2]
            - self.raw._model.geom_size[self.raw._peg_geom_id, 1]
        )
        socket_bottom_top_z = (
            self.raw._model.geom_pos[self.raw._socket_bottom_geom_id, 2]
            + self.raw._model.geom_size[self.raw._socket_bottom_geom_id, 2]
        )
        self._target_tip_z = (
            socket_bottom_top_z
            - peg_bottom_z
            + self.raw._model.site_pos[self.raw._peg_tip_site_id, 2]
        )
        self._initial_object_z = np.zeros(2, dtype=np.float64)
        self._previous_potential = 0.0

    def _load_roots(self, path: str | Path, max_offset: int | None):
        with np.load(Path(path), allow_pickle=False) as bank:
            version = int(bank["version"].item())
            if version != 2:
                raise ValueError(f"Unsupported root bank version: {version}")
            state_spec = int(bank["state_spec"].item())
            if state_spec != int(mujoco.mjtState.mjSTATE_INTEGRATION):
                raise ValueError(f"Unsupported MuJoCo state spec: {state_spec}")
            mask = np.ones(len(bank["state"]), dtype=bool)
            if max_offset is not None:
                mask &= bank["offset"] <= max_offset
            if not mask.any():
                raise ValueError(f"No roots at or below offset {max_offset}")
            keys = (
                "state",
                "table_pos",
                "leg_sizes",
                "delta_h",
                "offset",
                "source_episode",
                "source_frame",
                "reference_action",
                "reference_length",
            )
            roots = {key: np.asarray(bank[key][mask]).copy() for key in keys}

        expected = mujoco.mj_stateSize(
            self.raw._model, mujoco.mjtState.mjSTATE_INTEGRATION
        )
        if roots["state"].shape[1] != expected:
            raise ValueError(
                f"Root state size {roots['state'].shape[1]} does not match model {expected}"
            )
        expected_legs = (len(self.raw._table_leg_geom_ids), 3)
        if roots["leg_sizes"].shape[1:] != expected_legs:
            raise ValueError(
                f"Root leg geometry {roots['leg_sizes'].shape[1:]} does not match {expected_legs}"
            )
        if roots["reference_action"].ndim != 3 or roots["reference_action"].shape[2] != 46:
            raise ValueError("Reference actions must have shape (roots, steps, 46)")
        if np.any(roots["reference_length"] < 1) or np.any(
            roots["reference_length"] > roots["reference_action"].shape[1]
        ):
            raise ValueError("Invalid reference action lengths")
        return roots

    def _restore_root(self) -> dict[str, int]:
        index = int(self._root_rng.integers(len(self._roots["state"])))
        self._root_index = index
        self.raw._model.body_pos[self.raw._table_body_id] = self._roots["table_pos"][
            index
        ]
        self.raw._model.geom_size[self.raw._table_leg_geom_ids] = self._roots[
            "leg_sizes"
        ][index]
        self.raw.delta_h = np.float64(self._roots["delta_h"][index])
        mujoco.mj_setState(
            self.raw._model,
            self.raw._data,
            self._roots["state"][index],
            mujoco.mjtState.mjSTATE_INTEGRATION,
        )
        self._reference_pos_bias.fill(0.0)
        if self.root_noise > 0:
            self._reference_pos_bias[:] = self._root_rng.uniform(
                -self.root_noise, self.root_noise, size=(2, 3)
            )
        mujoco.mj_forward(self.raw._model, self.raw._data)
        self.raw.env_step = 0
        return {
            "root_offset_steps": int(self._roots["offset"][index]),
            "root_source_episode": int(self._roots["source_episode"][index]),
            "root_source_frame": int(self._roots["source_frame"][index]),
        }

    @staticmethod
    def _integrate_quaternion(quat_wxyz: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
        current = Rotation.from_quat(np.roll(quat_wxyz, -1))
        updated = Rotation.from_rotvec(rotvec) * current
        return np.roll(updated.as_quat(), 1)

    def _reference(self) -> tuple[np.ndarray | None, float]:
        if self._roots is None:
            return None, 0.0
        length = int(self._roots["reference_length"][self._root_index])
        step = min(self._episode_steps, length - 1)
        phase = min(self._episode_steps / max(length - 1, 1), 1.0)
        return self._roots["reference_action"][self._root_index, step], phase

    def _absolute_action(self, action: np.ndarray) -> np.ndarray:
        residual = np.asarray(action, dtype=np.float64)
        if self._causal_controller is not None:
            if residual.shape != (3,):
                raise ValueError(f"Expected action shape (3,), got {residual.shape}.")
            gains = 1.0 + 0.5 * np.clip(residual, -1.0, 1.0)
            reference = self._causal_controller.action(self._episode_steps, gains)
            phase = self._causal_controller.phase(self._episode_steps)[0]
            phases = ("left_skill", "right_skill", "align", "orient", "insert")
            self._causal_phase = phases.index(phase) / (len(phases) - 1)
            self._last_base_action[:] = reference
            return reference
        if residual.shape != (12,):
            raise ValueError(f"Expected action shape (12,), got {residual.shape}.")
        # ponytail: successful replay supplies grasp motion; PPO learns bounded pose residuals.
        residual = np.clip(residual, -1.0, 1.0)
        if self._roots is not None:
            residual *= 0.5

        common_pos, common_rot = residual[0:3], residual[3:6]
        relative_pos, relative_rot = residual[6:9], residual[9:12]
        right_pos_delta = self.translation_scale * (common_pos + 0.5 * relative_pos)
        left_pos_delta = self.translation_scale * (common_pos - 0.5 * relative_pos)
        right_rot_delta = self.rotation_scale * (common_rot + 0.5 * relative_rot)
        left_rot_delta = self.rotation_scale * (common_rot - 0.5 * relative_rot)

        data = self.raw._data
        right_id, left_id = self.raw._mocap_right_id, self.raw._mocap_left_id
        reference, _ = self._reference()
        if reference is not None:
            if self.online_bias_correction and self._has_reference_command:
                observed = data.mocap_pos[[right_id, left_id]]
                self._estimated_reference_bias += (
                    observed - self._last_reference_pos - self._last_position_delta
                )
            right_base_pos = (
                reference[0:3]
                + self._reference_pos_bias[0]
                - self._estimated_reference_bias[0]
            )
            right_base_quat = reference[3:7]
            left_base_pos = (
                reference[7:10]
                + self._reference_pos_bias[1]
                - self._estimated_reference_bias[1]
            )
            left_base_quat = reference[10:14]
            hand_targets = reference[14:]
        else:
            right_base_pos, right_base_quat = (
                data.mocap_pos[right_id],
                data.mocap_quat[right_id],
            )
            left_base_pos, left_base_quat = (
                data.mocap_pos[left_id],
                data.mocap_quat[left_id],
            )
            hand_targets = data.ctrl[self.raw._allegro_ctrl_ids]

        workspace_low = np.asarray([-0.75, -0.85, 0.72])
        workspace_high = np.asarray([0.25, 0.85, 1.75])
        right_pos = np.clip(
            right_base_pos + right_pos_delta, workspace_low, workspace_high
        )
        left_pos = np.clip(
            left_base_pos + left_pos_delta, workspace_low, workspace_high
        )
        right_quat = self._integrate_quaternion(right_base_quat, right_rot_delta)
        left_quat = self._integrate_quaternion(left_base_quat, left_rot_delta)
        if reference is not None:
            self._last_reference_pos[:] = reference[[0, 1, 2, 7, 8, 9]].reshape(2, 3)
            self._last_position_delta[:] = right_pos - right_base_pos, left_pos - left_base_pos
            self._has_reference_command = True
        return np.concatenate(
            [right_pos, right_quat, left_pos, left_quat, hand_targets]
        ).astype(np.float32)

    def _task_features(self) -> tuple[np.ndarray, np.ndarray, float]:
        data = self.raw._data
        socket_rotation = data.xmat[self.raw._socket_body_id].reshape(3, 3)
        peg_rotation = data.xmat[self.raw._peg_body_id].reshape(3, 3)
        tip_local = socket_rotation.T @ (
            data.site_xpos[self.raw._peg_tip_site_id]
            - data.xpos[self.raw._socket_body_id]
        )
        tip_error = tip_local - np.asarray([0.0, 0.0, self._target_tip_z])
        peg_axis_local = socket_rotation.T @ peg_rotation[:, 2]
        contact_progress = min(
            getattr(self.raw, "_success_counter", 0) / 30.0, 1.0
        )
        return tip_error, peg_axis_local, float(contact_progress)

    def _observation(self) -> np.ndarray:
        data = self.raw._data
        mocap_ids = [self.raw._mocap_right_id, self.raw._mocap_left_id]
        tip_error, peg_axis_local, contact_progress = self._task_features()
        hand_contact = self._contact_detector.compute(self.raw)
        reference, phase = self._reference()
        if self._causal_controller is not None:
            reference_context = np.concatenate(
                [self._last_base_action[:14], np.asarray([self._causal_phase])]
            )
        else:
            reference_context = (
                np.concatenate([reference[:14], np.asarray([phase])])
                if reference is not None
                else np.zeros(15)
            )
        return np.concatenate(
            [
                data.qpos,
                data.qvel,
                data.ctrl,
                data.mocap_pos[mocap_ids].ravel(),
                data.mocap_quat[mocap_ids].ravel(),
                tip_error,
                peg_axis_local,
                np.asarray([contact_progress]),
                np.asarray(
                    [hand_contact.peg_contact, hand_contact.tray_contact]
                ),
                reference_context,
            ]
        ).astype(np.float32)

    def _task_metrics(self) -> dict[str, float]:
        hand_contact = self._contact_detector.compute(self.raw)
        data = self.raw._data
        tip_error, peg_axis_local, contact_progress = self._task_features()
        tip_distance = np.linalg.norm(
            data.site_xpos[self.raw._peg_tip_site_id]
            - data.site_xpos[self.raw._socket_site_id]
        )
        return {
            "tip_distance_m": float(tip_distance),
            "lateral_error_m": float(np.linalg.norm(tip_error[:2])),
            "axial_error_m": float(tip_error[2]),
            "angular_error_rad": float(
                np.arccos(np.clip(peg_axis_local[2], -1.0, 1.0))
            ),
            "contact_progress": contact_progress,
            "peg_hand_contact": float(hand_contact.peg_contact),
            "tray_hand_contact": float(hand_contact.tray_contact),
            "dual_grasp": float(hand_contact.peg_contact and hand_contact.tray_contact),
            "peg_drop_m": float(
                max(
                    self._initial_object_z[0]
                    - data.xpos[self.raw._peg_body_id, 2],
                    0.0,
                )
            ),
            "socket_drop_m": float(
                max(
                    self._initial_object_z[1]
                    - data.xpos[self.raw._socket_body_id, 2],
                    0.0,
                )
            ),
        }

    @staticmethod
    def _potential(metrics: dict[str, float]) -> float:
        lateral = np.exp(-metrics["lateral_error_m"] / 0.01)
        axial = np.exp(-abs(metrics["axial_error_m"]) / 0.025)
        angular = np.exp(-metrics["angular_error_rad"] / 0.15)
        funnel = lateral * axial * angular
        return float(
            3.0 * lateral
            + 2.0 * axial
            + 2.0 * angular
            + 4.0 * metrics["dual_grasp"]
            + 5.0 * funnel
            + 5.0 * metrics["contact_progress"]
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._root_rng = np.random.default_rng(seed)
        _, info = self._env.reset(seed=seed, options=options)
        self._root_info = self._restore_root() if self._roots is not None else {}
        self._estimated_reference_bias.fill(0.0)
        self._last_reference_pos.fill(0.0)
        self._last_position_delta.fill(0.0)
        self._has_reference_command = False
        self.raw._success_started = False
        self.raw._success_counter = 0
        self._episode_steps = 0
        self._last_base_action.fill(0.0)
        self._causal_phase = 0.0
        if self._causal_templates is not None:
            self._causal_controller = CausalAssemblyController(
                self.raw, self._causal_templates
            )
            self._causal_controller.reset()
            if self._causal_warm_start:
                while self._episode_steps < self._causal_controller._skill_end:
                    base = self._causal_controller.action(self._episode_steps)
                    self._env.step(base)
                    self._last_base_action[:] = base
                    self._episode_steps += 1
        self._grasp_loss_steps = 0
        self._initial_object_z[:] = (
            self.raw._data.xpos[self.raw._peg_body_id, 2],
            self.raw._data.xpos[self.raw._socket_body_id, 2],
        )
        metrics = self._task_metrics()
        self._previous_potential = self._potential(metrics)
        info = dict(info)
        info.update(self._root_info)
        info.update(metrics)
        info["is_success"] = bool(info.get("succeed", False))
        return self._observation(), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        _, _, terminated, truncated, info = self._env.step(
            self._absolute_action(action)
        )
        self._episode_steps += 1
        metrics = self._task_metrics()
        if metrics["dual_grasp"]:
            self._grasp_loss_steps = 0
        else:
            self._grasp_loss_steps += 1
        potential = self._potential(metrics)
        success = bool(info.get("succeed", False))
        reward = (
            self.shaping_gamma * potential
            - self._previous_potential
            + 100.0 * success
            - self.residual_cost * float(np.mean(np.square(action)))
        )
        grasp_lost = self._roots is not None and self._grasp_loss_steps >= 30
        dropped = grasp_lost and max(metrics["peg_drop_m"], metrics["socket_drop_m"]) > 0.10
        if (
            self._episode_limit is not None
            and self._episode_steps >= self._episode_limit
            and not terminated
        ):
            truncated = True
        self._previous_potential = potential
        info = dict(info)
        info.update(self._root_info)
        info.update(metrics)
        info["is_success"] = success
        info["dropped"] = dropped
        info["grasp_lost"] = grasp_lost
        return self._observation(), float(reward), terminated, truncated, info

    def close(self) -> None:
        self._env.close()
