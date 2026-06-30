"""Bimanual insert controllers: direct wrist12 or zarr oracle replay."""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from hybrid_insert.assembly_contacts import AssemblyContactLabeler
from hybrid_insert.geometry import peg_insert_end_pos, tip_socket_distance

from interaction_retarget.constants import PEG_BODY
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.replay import raw_flat_to_dict

from .adapter import build_obs_pose9, clamp_wrist_target, read_sim_poses7
from .config import PoseInsertAdapterConfig
from .inference import PoseInsertPolicyRunner
from .inference_bimanual import BimanualAction44Runner, BimanualPoseInsertRunner
from .pre_insert import is_pre_insert_ready
from .wrist_actions import arm23_to_wrist6, dual_wrist12_to_action44, zarr_flat_to_action44, zarr_flat_to_dual_wrist12

_SOCKET_SITE = "industreal_tray_insert_round_peg_8mm_socket_site"


class _Phase(Enum):
    POLICY = auto()
    SETTLE = auto()
    POSEINSERT = auto()
    RELEASE = auto()


class DirectDualWristController:
    """PoseDP outputs dual wrist12; fingers locked at handoff."""

    def __init__(
        self,
        policy: BimanualPoseInsertRunner,
        config: PoseInsertAdapterConfig | None = None,
    ) -> None:
        self.policy = policy
        self.config = config or PoseInsertAdapterConfig(freeze_left_arm_at_handoff=False)
        self._labeler: AssemblyContactLabeler | None = None
        self._peg_body_id: int | None = None
        self._socket_site_id: int | None = None
        self._peg_rest_z: float | None = None

        self._phase = _Phase.POLICY
        self._handoff_streak = 0
        self._insert_streak = 0
        self._policy_steps = 0
        self._settle_steps = 0
        self._poseinsert_steps = 0
        self._release_steps = 0
        self._peg_lost_streak = 0
        self._replan_counter = 0
        self._handoff_happened = False

        self._hold_r_hand: np.ndarray | None = None
        self._hold_l_hand: np.ndarray | None = None
        self._open_r_hand: np.ndarray | None = None
        self._right_wrist = np.zeros(6, dtype=np.float64)
        self._left_wrist = np.zeros(6, dtype=np.float64)
        self._waypoints: list[np.ndarray] = []
        self._waypoint_idx = 0

    @property
    def active(self) -> bool:
        return self._phase != _Phase.POLICY

    @property
    def handoff_happened(self) -> bool:
        return self._handoff_happened

    @property
    def phase_name(self) -> str:
        return self._phase.name

    @property
    def insert_done(self) -> bool:
        if not self._handoff_happened:
            return False
        if self._phase == _Phase.POLICY:
            return True
        return (
            self._phase == _Phase.POSEINSERT
            and self._poseinsert_steps > self.config.max_poseinsert_steps
        )

    @property
    def needs_policy_left(self) -> bool:
        return False

    def reset(self, raw_env, *, peg_rest_z: float | None = None, tray_rest_z: float | None = None) -> None:
        self._bind_env(raw_env)
        assert self._labeler is not None
        if peg_rest_z is not None and tray_rest_z is not None:
            self._labeler._peg_rest_z = float(peg_rest_z)  # noqa: SLF001
            self._labeler._tray_rest_z = float(tray_rest_z)  # noqa: SLF001
            self._peg_rest_z = float(peg_rest_z)
        else:
            self._labeler.reset_reference(raw_env)
            self._peg_rest_z = float(raw_env._data.xpos[self._peg_body_id, 2])
        self._phase = _Phase.POLICY
        self._handoff_streak = 0
        self._insert_streak = 0
        self._policy_steps = 0
        self._settle_steps = 0
        self._poseinsert_steps = 0
        self._release_steps = 0
        self._peg_lost_streak = 0
        self._replan_counter = 0
        self._handoff_happened = False
        self._hold_r_hand = None
        self._hold_l_hand = None
        self._open_r_hand = None
        self._waypoints = []
        self._waypoint_idx = 0

    def update_handoff(self, raw_env, policy_action44: np.ndarray) -> None:
        if self.active:
            return
        self._bind_env(raw_env)
        assert self._labeler is not None
        outcome = self._labeler.compute(raw_env)
        peg_dz = float(raw_env._data.xpos[self._peg_body_id, 2]) - float(self._peg_rest_z or 0.0)
        ready = outcome.tray_ok and outcome.peg_ok and peg_dz >= self.config.lift_ready_m
        self._handoff_streak = self._handoff_streak + 1 if ready else 0
        if self._handoff_streak >= self.config.handoff_confirm_frames:
            self._activate(policy_action44, raw_env)

    def begin_pose_insert(self, raw_env, policy_action44: np.ndarray) -> bool:
        if self.active:
            return True
        self._bind_env(raw_env)
        assert self._labeler is not None
        if not is_pre_insert_ready(
            raw_env, self._labeler, lift_ready_m=self.config.lift_ready_m, peg_rest_z=self._peg_rest_z
        ):
            return False
        self._activate(np.asarray(policy_action44, dtype=np.float64).reshape(-1), raw_env)
        return True

    def merge_dual_arm(self, raw_env, policy_action44: np.ndarray) -> np.ndarray:
        action = np.asarray(policy_action44, dtype=np.float64).reshape(-1).copy()
        if action.shape[0] != 44:
            raise ValueError(f"Expected 44d action, got {action.shape[0]}")
        self._bind_env(raw_env)
        outcome = self._labeler.compute(raw_env) if self._labeler else None
        if not self.active:
            self._policy_steps += 1
            return action.astype(np.float32)

        assert self._hold_r_hand is not None and self._hold_l_hand is not None
        if outcome is not None:
            if outcome.peg_ok:
                self._peg_lost_streak = 0
            elif self._phase in (_Phase.SETTLE, _Phase.POSEINSERT):
                self._peg_lost_streak += 1
        if (
            self._phase in (_Phase.SETTLE, _Phase.POSEINSERT)
            and outcome is not None
            and self._peg_lost_streak >= self.config.peg_lost_abort_frames
        ):
            print("direct_wrist12: peg lost -> POLICY", flush=True)
            self._deactivate()
            return action.astype(np.float32)

        if self._phase == _Phase.SETTLE:
            self._settle_steps += 1
            if self._settle_steps >= self.config.handoff_settle_frames:
                self._phase = _Phase.POSEINSERT
                self._poseinsert_steps = 0
                self._replan_counter = 0
                self._waypoints = []
                self._waypoint_idx = 0
                print("direct_wrist12: settle done -> POSEINSERT", flush=True)
        elif self._phase == _Phase.POSEINSERT:
            self._poseinsert_steps += 1
            if self._poseinsert_steps > self.config.max_poseinsert_steps:
                if self._poseinsert_steps == self.config.max_poseinsert_steps + 1:
                    print("direct_wrist12: POSEINSERT timeout", flush=True)
            else:
                self._poseinsert_step(raw_env)
            if self._should_release_grasp(raw_env):
                self._insert_streak += 1
            else:
                self._insert_streak = 0
            if self._insert_streak >= self.config.release_confirm_frames:
                self._phase = _Phase.RELEASE
                self._release_steps = 0
                print("direct_wrist12: POSEINSERT -> RELEASE", flush=True)
        elif self._phase == _Phase.RELEASE:
            self._release_steps += 1

        merged = dual_wrist12_to_action44(
            np.concatenate([self._right_wrist, self._left_wrist]),
            right_hand=self._current_right_hand(),
            left_hand=self._hold_l_hand,
        )
        return merged.astype(np.float32)

    def episode_summary(self) -> str:
        return f"handoff={self._handoff_happened} phase={self.phase_name} direct_wrist12=True"

    def _activate(self, policy_action44: np.ndarray, raw_env) -> None:
        action = np.asarray(policy_action44, dtype=np.float64).reshape(-1)
        self._hold_r_hand = action[6:22].copy()
        self._hold_l_hand = action[28:44].copy()
        self._open_r_hand = self._compute_open_hand_pose(self._hold_r_hand)
        self._right_wrist = action[0:6].copy()
        self._left_wrist = action[22:28].copy()
        self._phase = _Phase.SETTLE
        self._handoff_happened = True
        self._settle_steps = 0
        self._insert_streak = 0
        self._peg_lost_streak = 0
        peg_dz = float(raw_env._data.xpos[self._peg_body_id, 2]) - float(self._peg_rest_z or 0.0)
        print(f"direct_wrist12: pre_insert -> SETTLE peg_dz={peg_dz:.3f}", flush=True)

    def _deactivate(self) -> None:
        self._phase = _Phase.POLICY
        self._handoff_streak = 0
        self._insert_streak = 0
        self._settle_steps = 0
        self._poseinsert_steps = 0
        self._release_steps = 0
        self._peg_lost_streak = 0
        self._waypoints = []
        self._waypoint_idx = 0

    def _poseinsert_step(self, raw_env) -> None:
        cfg = self.config
        need_replan = (
            self._waypoint_idx >= len(self._waypoints)
            or self._replan_counter % max(1, cfg.replan_interval) == 0
        )
        if need_replan:
            source_pose7, target_pose7 = read_sim_poses7(raw_env)
            workspace = self.policy.pose_workspace if self.policy.normalize else None
            obs_pose9 = build_obs_pose9(
                source_pose7,
                target_pose7,
                workspace=workspace if cfg.normalize_translation else None,
            )
            horizon = self.policy.predict_wrist12_horizon(obs_pose9)
            self._waypoints = [horizon[i] for i in range(horizon.shape[0])]
            self._waypoint_idx = 0
            self._replan_counter = 0

        tgt = np.asarray(self._waypoints[self._waypoint_idx], dtype=np.float64).reshape(12)
        self._waypoint_idx += 1
        self._replan_counter += 1

        r_xyz, r_rot = clamp_wrist_target(
            self._right_wrist[0:3],
            self._right_wrist[3:6],
            tgt[0:3],
            tgt[3:6],
            max_step_m=cfg.max_wrist_step_m,
            max_rot_step_rad=cfg.max_wrist_rot_step_rad,
        )
        l_xyz, l_rot = clamp_wrist_target(
            self._left_wrist[0:3],
            self._left_wrist[3:6],
            tgt[6:9],
            tgt[9:12],
            max_step_m=cfg.max_wrist_step_m,
            max_rot_step_rad=cfg.max_wrist_rot_step_rad,
        )
        self._right_wrist = np.concatenate([r_xyz, r_rot])
        self._left_wrist = np.concatenate([l_xyz, l_rot])

    def _bind_env(self, raw_env) -> None:
        if self._labeler is not None:
            return
        self._labeler = AssemblyContactLabeler(raw_env, lift_threshold_m=self.config.lift_threshold_m)
        model = raw_env._model
        self._socket_site_id = int(model.site(_SOCKET_SITE).id)
        self._peg_body_id = int(model.body(PEG_BODY).id)

    def _peg_insert_pos(self, raw_env) -> np.ndarray:
        data = raw_env._data
        return peg_insert_end_pos(data.xpos[self._peg_body_id], data.xmat[self._peg_body_id])

    def _socket_pos(self, raw_env) -> np.ndarray:
        return np.asarray(raw_env._data.site_xpos[self._socket_site_id], dtype=np.float64)

    def _should_release_grasp(self, raw_env) -> bool:
        return tip_socket_distance(self._peg_insert_pos(raw_env), self._socket_pos(raw_env)) <= (
            self.config.release_insert_socket_dist_m
        )

    def _current_right_hand(self) -> np.ndarray:
        assert self._hold_r_hand is not None
        if self._phase != _Phase.RELEASE or self._open_r_hand is None:
            return self._hold_r_hand
        alpha = min(1.0, self._release_steps / max(1, self.config.release_steps))
        return (1.0 - alpha) * self._hold_r_hand + alpha * self._open_r_hand

    @staticmethod
    def _compute_open_hand_pose(closed_hand: np.ndarray) -> np.ndarray:
        closed = np.asarray(closed_hand, dtype=np.float64)
        open_pose = np.minimum(closed, 0.05)
        return np.minimum(open_pose, closed * 0.15)


class ZarrInsertOracleController:
    """Open-loop zarr insert segment (both arms); for demo_replay oracle baseline."""

    def __init__(
        self,
        zarr_path: Path | str,
        start_frame: int,
        config: PoseInsertAdapterConfig | None = None,
        end_frame: int | None = None,
    ) -> None:
        self.config = config or PoseInsertAdapterConfig()
        actions, _, _ = load_zarr_episode(Path(zarr_path))
        start = int(start_frame)
        end = int(end_frame) if end_frame is not None else len(actions) - 1
        self._actions = actions[start + 1 : end + 1]
        self._step_idx = 0
        self._labeler: AssemblyContactLabeler | None = None
        self._peg_body_id: int | None = None
        self._socket_site_id: int | None = None
        self._peg_rest_z: float | None = None
        self._phase = _Phase.POSEINSERT
        self._handoff_happened = True
        self._hold_r_hand: np.ndarray | None = None
        self._hold_l_hand: np.ndarray | None = None
        self._poseinsert_steps = 0

    @property
    def active(self) -> bool:
        return self._handoff_happened and self._step_idx < len(self._actions)

    @property
    def handoff_happened(self) -> bool:
        return self._handoff_happened

    @property
    def phase_name(self) -> str:
        return "ORACLE_ZARR" if self.active else "DONE"

    @property
    def insert_done(self) -> bool:
        return self._step_idx >= len(self._actions) or self._poseinsert_steps > self.config.max_poseinsert_steps

    @property
    def needs_policy_left(self) -> bool:
        return False

    def reset(self, raw_env, *, peg_rest_z: float | None = None, tray_rest_z: float | None = None) -> None:
        self._bind_env(raw_env)
        assert self._labeler is not None
        if peg_rest_z is not None and tray_rest_z is not None:
            self._labeler._peg_rest_z = float(peg_rest_z)  # noqa: SLF001
            self._labeler._tray_rest_z = float(tray_rest_z)  # noqa: SLF001
            self._peg_rest_z = float(peg_rest_z)
        else:
            self._labeler.reset_reference(raw_env)
            self._peg_rest_z = float(raw_env._data.xpos[self._peg_body_id, 2])

    def update_handoff(self, raw_env, policy_action44: np.ndarray) -> None:
        return

    def begin_pose_insert(self, raw_env, policy_action44: np.ndarray) -> bool:
        self._step_idx = 0
        self._poseinsert_steps = 0
        print(f"zarr_oracle: insert open-loop frames={len(self._actions)}", flush=True)
        return True

    def merge_dual_arm(self, raw_env, policy_action44: np.ndarray) -> np.ndarray:
        if not self.active:
            return np.asarray(policy_action44, dtype=np.float32)
        self._poseinsert_steps += 1
        flat = self._actions[self._step_idx]
        self._step_idx += 1
        return zarr_flat_to_action44(flat).astype(np.float32)

    def episode_summary(self) -> str:
        return f"oracle_zarr steps={self._step_idx}/{len(self._actions)}"

    def _bind_env(self, raw_env) -> None:
        if self._labeler is not None:
            return
        self._labeler = AssemblyContactLabeler(raw_env, lift_threshold_m=self.config.lift_threshold_m)
        model = raw_env._model
        self._peg_body_id = int(model.body(PEG_BODY).id)
        self._socket_site_id = int(model.site(_SOCKET_SITE).id)


# Backward alias
BimanualRelativePoseController = DirectDualWristController


class DirectAction44Controller:
    """PoseDP outputs full dual-arm44 (wrist+hand); closed-loop replan each step."""

    def __init__(
        self,
        policy: BimanualAction44Runner,
        config: PoseInsertAdapterConfig | None = None,
    ) -> None:
        self.policy = policy
        self.config = config or PoseInsertAdapterConfig(freeze_left_arm_at_handoff=False)
        self._labeler: AssemblyContactLabeler | None = None
        self._peg_body_id: int | None = None
        self._socket_site_id: int | None = None
        self._peg_rest_z: float | None = None

        self._phase = _Phase.POLICY
        self._handoff_streak = 0
        self._insert_streak = 0
        self._policy_steps = 0
        self._settle_steps = 0
        self._poseinsert_steps = 0
        self._release_steps = 0
        self._peg_lost_streak = 0
        self._replan_counter = 0
        self._handoff_happened = False

        self._hold_action44 = np.zeros(44, dtype=np.float64)
        self._target_action44 = np.zeros(44, dtype=np.float64)
        self._open_r_hand: np.ndarray | None = None

    @property
    def active(self) -> bool:
        return self._phase != _Phase.POLICY

    @property
    def handoff_happened(self) -> bool:
        return self._handoff_happened

    @property
    def phase_name(self) -> str:
        return self._phase.name

    @property
    def insert_done(self) -> bool:
        if not self._handoff_happened:
            return False
        if self._phase == _Phase.POLICY:
            return True
        return (
            self._phase == _Phase.POSEINSERT
            and self._poseinsert_steps > self.config.max_poseinsert_steps
        )

    @property
    def needs_policy_left(self) -> bool:
        return False

    def reset(self, raw_env, *, peg_rest_z: float | None = None, tray_rest_z: float | None = None) -> None:
        self._bind_env(raw_env)
        assert self._labeler is not None
        if peg_rest_z is not None and tray_rest_z is not None:
            self._labeler._peg_rest_z = float(peg_rest_z)  # noqa: SLF001
            self._labeler._tray_rest_z = float(tray_rest_z)  # noqa: SLF001
            self._peg_rest_z = float(peg_rest_z)
        else:
            self._labeler.reset_reference(raw_env)
            self._peg_rest_z = float(raw_env._data.xpos[self._peg_body_id, 2])
        self._phase = _Phase.POLICY
        self._handoff_streak = 0
        self._insert_streak = 0
        self._policy_steps = 0
        self._settle_steps = 0
        self._poseinsert_steps = 0
        self._release_steps = 0
        self._peg_lost_streak = 0
        self._replan_counter = 0
        self._handoff_happened = False
        self._open_r_hand = None

    def update_handoff(self, raw_env, policy_action44: np.ndarray) -> None:
        if self.active:
            return
        self._bind_env(raw_env)
        assert self._labeler is not None
        outcome = self._labeler.compute(raw_env)
        peg_dz = float(raw_env._data.xpos[self._peg_body_id, 2]) - float(self._peg_rest_z or 0.0)
        ready = outcome.tray_ok and outcome.peg_ok and peg_dz >= self.config.lift_ready_m
        self._handoff_streak = self._handoff_streak + 1 if ready else 0
        if self._handoff_streak >= self.config.handoff_confirm_frames:
            self._activate(policy_action44, raw_env)

    def begin_pose_insert(self, raw_env, policy_action44: np.ndarray) -> bool:
        if self.active:
            return True
        self._bind_env(raw_env)
        assert self._labeler is not None
        if not is_pre_insert_ready(
            raw_env, self._labeler, lift_ready_m=self.config.lift_ready_m, peg_rest_z=self._peg_rest_z
        ):
            return False
        self._activate(np.asarray(policy_action44, dtype=np.float64).reshape(-1), raw_env)
        return True

    def merge_dual_arm(self, raw_env, policy_action44: np.ndarray) -> np.ndarray:
        action = np.asarray(policy_action44, dtype=np.float64).reshape(-1).copy()
        if action.shape[0] != 44:
            raise ValueError(f"Expected 44d action, got {action.shape[0]}")
        self._bind_env(raw_env)
        outcome = self._labeler.compute(raw_env) if self._labeler else None
        if not self.active:
            self._policy_steps += 1
            return action.astype(np.float32)

        if outcome is not None:
            if outcome.peg_ok:
                self._peg_lost_streak = 0
            elif self._phase in (_Phase.SETTLE, _Phase.POSEINSERT):
                self._peg_lost_streak += 1
        if (
            self._phase in (_Phase.SETTLE, _Phase.POSEINSERT)
            and outcome is not None
            and self._peg_lost_streak >= self.config.peg_lost_abort_frames
        ):
            print("direct_action44: peg lost -> POLICY", flush=True)
            self._deactivate()
            return action.astype(np.float32)

        if self._phase == _Phase.SETTLE:
            self._settle_steps += 1
            if self._settle_steps >= self.config.handoff_settle_frames:
                self._phase = _Phase.POSEINSERT
                self._poseinsert_steps = 0
                self._replan_counter = 0
                print("direct_action44: settle done -> POSEINSERT", flush=True)
        elif self._phase == _Phase.POSEINSERT:
            self._poseinsert_steps += 1
            if self._poseinsert_steps <= self.config.max_poseinsert_steps:
                self._poseinsert_step(raw_env)
            elif self._poseinsert_steps == self.config.max_poseinsert_steps + 1:
                print("direct_action44: POSEINSERT timeout", flush=True)
            if self._should_release_grasp(raw_env):
                self._insert_streak += 1
            else:
                self._insert_streak = 0
            if self._insert_streak >= self.config.release_confirm_frames:
                self._phase = _Phase.RELEASE
                self._release_steps = 0
                print("direct_action44: POSEINSERT -> RELEASE", flush=True)
        elif self._phase == _Phase.RELEASE:
            self._release_steps += 1

        out = self._target_action44.copy()
        if self._phase == _Phase.RELEASE and self._open_r_hand is not None:
            alpha = min(1.0, self._release_steps / max(1, self.config.release_steps))
            out[6:22] = (1.0 - alpha) * out[6:22] + alpha * self._open_r_hand
        return out.astype(np.float32)

    def episode_summary(self) -> str:
        return f"handoff={self._handoff_happened} phase={self.phase_name} direct_action44=True"

    def _activate(self, policy_action44: np.ndarray, raw_env) -> None:
        action = np.asarray(policy_action44, dtype=np.float64).reshape(-1)
        self._hold_action44 = action.copy()
        self._target_action44 = action.copy()
        self._open_r_hand = DirectDualWristController._compute_open_hand_pose(action[6:22])
        self._phase = _Phase.SETTLE
        self._handoff_happened = True
        self._settle_steps = 0
        self._insert_streak = 0
        self._peg_lost_streak = 0
        peg_dz = float(raw_env._data.xpos[self._peg_body_id, 2]) - float(self._peg_rest_z or 0.0)
        print(f"direct_action44: pre_insert -> SETTLE peg_dz={peg_dz:.3f}", flush=True)

    def _deactivate(self) -> None:
        self._phase = _Phase.POLICY
        self._handoff_streak = 0
        self._insert_streak = 0
        self._settle_steps = 0
        self._poseinsert_steps = 0
        self._release_steps = 0
        self._peg_lost_streak = 0

    def _poseinsert_step(self, raw_env) -> None:
        source_pose7, target_pose7 = read_sim_poses7(raw_env)
        workspace = self.policy.pose_workspace if self.policy.normalize else None
        obs_pose9 = build_obs_pose9(
            source_pose7,
            target_pose7,
            workspace=workspace if self.config.normalize_translation else None,
        )
        horizon = self.policy.predict_action44_horizon(obs_pose9)
        pred = np.asarray(horizon[0], dtype=np.float64).reshape(44)
        blend = float(np.clip(self.config.action_blend, 0.05, 1.0))
        self._target_action44 = (1.0 - blend) * self._target_action44 + blend * pred
        self._replan_counter += 1

    def _bind_env(self, raw_env) -> None:
        if self._labeler is not None:
            return
        self._labeler = AssemblyContactLabeler(raw_env, lift_threshold_m=self.config.lift_threshold_m)
        model = raw_env._model
        self._socket_site_id = int(model.site(_SOCKET_SITE).id)
        self._peg_body_id = int(model.body(PEG_BODY).id)

    def _peg_insert_pos(self, raw_env) -> np.ndarray:
        data = raw_env._data
        return peg_insert_end_pos(data.xpos[self._peg_body_id], data.xmat[self._peg_body_id])

    def _socket_pos(self, raw_env) -> np.ndarray:
        return np.asarray(raw_env._data.site_xpos[self._socket_site_id], dtype=np.float64)

    def _should_release_grasp(self, raw_env) -> bool:
        return tip_socket_distance(self._peg_insert_pos(raw_env), self._socket_pos(raw_env)) <= (
            self.config.release_insert_socket_dist_m
        )
