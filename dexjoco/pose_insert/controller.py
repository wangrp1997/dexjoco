"""Closed-loop PoseInsert controller for bimanual_assembly sim."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np
from scipy.spatial.transform import Rotation as R

from hybrid_insert.assembly_contacts import AssemblyContactLabeler
from hybrid_insert.geometry import (
    hole_opening_axis,
    in_approach_cylinder,
    peg_insert_end_pos,
    tip_socket_distance,
)

from interaction_retarget.constants import PEG_BODY

from .adapter import (
    build_obs_pose9,
    calibrate_peg_to_wrist,
    clamp_wrist_target,
    read_sim_poses7,
    relative_pose9_to_world_source,
    source_pose7_to_wrist_pose7,
    wrist_pose7_to_rotvec_action,
)
from .config import PoseInsertAdapterConfig
from .inference import PoseInsertPolicyRunner
from .pre_insert import is_pre_insert_ready

_SOCKET_SITE = "industreal_tray_insert_round_peg_8mm_socket_site"
_BOTTOM_GEOM = "industreal_tray_insert_round_peg_8mm_bottom_contact"


class _Phase(Enum):
    POLICY = auto()
    SETTLE = auto()
    POSEINSERT = auto()
    RELEASE = auto()


class PoseInsertController:
    """Hand off after approach cylinder; execute PoseDP relative pose on right arm."""

    def __init__(
        self,
        policy: PoseInsertPolicyRunner,
        config: PoseInsertAdapterConfig | None = None,
    ) -> None:
        self.policy = policy
        self.config = config or PoseInsertAdapterConfig()
        self._labeler: AssemblyContactLabeler | None = None
        self._socket_site_id: int | None = None
        self._peg_body_id: int | None = None
        self._bottom_geom_id: int | None = None
        self._peg_rest_z: float | None = None

        self._phase = _Phase.POLICY
        self._handoff_streak = 0
        self._grasp_lock_streak = 0
        self._insert_streak = 0
        self._policy_steps = 0
        self._settle_steps = 0
        self._poseinsert_steps = 0
        self._release_steps = 0
        self._peg_lost_streak = 0
        self._replan_counter = 0

        self._handoff_happened = False
        self._grasp_locked_hand: np.ndarray | None = None
        self._hold_r_hand: np.ndarray | None = None
        self._hold_l_arm: np.ndarray | None = None
        self._open_r_hand: np.ndarray | None = None

        self._right_wrist_xyz: np.ndarray | None = None
        self._right_wrist_rotvec: np.ndarray | None = None
        self._peg_to_wrist4: np.ndarray | None = None
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
        """True after handoff when peg lost or POSEINSERT timed out."""
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
        if not self.active:
            return False
        return self._hold_l_arm is None

    def reset(
        self,
        raw_env,
        *,
        peg_rest_z: float | None = None,
        tray_rest_z: float | None = None,
    ) -> None:
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
        self._grasp_lock_streak = 0
        self._insert_streak = 0
        self._policy_steps = 0
        self._settle_steps = 0
        self._poseinsert_steps = 0
        self._release_steps = 0
        self._peg_lost_streak = 0
        self._replan_counter = 0
        self._handoff_happened = False
        self._grasp_locked_hand = None
        self._hold_r_hand = None
        self._hold_l_arm = None
        self._open_r_hand = None
        self._right_wrist_xyz = None
        self._right_wrist_rotvec = None
        self._peg_to_wrist4 = None
        self._waypoints = []
        self._waypoint_idx = 0

    def update_handoff(self, raw_env, policy_action44: np.ndarray) -> None:
        """Wait until pre-insert (lift done), then start PoseInsert."""
        if self.active:
            return
        self._bind_env(raw_env)
        assert self._labeler is not None
        outcome = self._labeler.compute(raw_env)
        peg_dz = self._peg_dz(raw_env)
        ready = (
            outcome.tray_ok
            and outcome.peg_ok
            and peg_dz >= self.config.lift_ready_m
        )
        if ready:
            self._handoff_streak += 1
        else:
            self._handoff_streak = 0

        if (
            self.config.handoff_debug_interval > 0
            and outcome.peg_ok
            and self._policy_steps % self.config.handoff_debug_interval == 0
        ):
            print(
                "pose_insert: waiting pre_insert "
                f"tray_ok={outcome.tray_ok} peg_ok={outcome.peg_ok} peg_dz={peg_dz:.3f} "
                f"streak={self._handoff_streak}",
                flush=True,
            )

        if self._handoff_streak >= self.config.handoff_confirm_frames:
            self._activate(policy_action44, raw_env)

    def begin_pose_insert(self, raw_env, policy_action44: np.ndarray) -> bool:
        """Enter PoseInsert from pre-insert state (after demo replay or VLA lift)."""
        if self.active:
            return True
        self._bind_env(raw_env)
        assert self._labeler is not None
        if not is_pre_insert_ready(
            raw_env,
            self._labeler,
            lift_ready_m=self.config.lift_ready_m,
            peg_rest_z=self._peg_rest_z,
        ):
            return False
        action = np.asarray(policy_action44, dtype=np.float64).reshape(-1)
        # Demo/VLA handoff skips POLICY phase — lock current finger ctrl targets.
        self._grasp_locked_hand = action[6:22].copy()
        self._activate(action, raw_env)
        return True

    def merge_right_arm(self, raw_env, policy_action44: np.ndarray) -> np.ndarray:
        action = np.asarray(policy_action44, dtype=np.float64).reshape(-1).copy()
        if action.shape[0] != 44:
            raise ValueError(f"Expected 44d action, got {action.shape[0]}")

        self._bind_env(raw_env)
        outcome = self._labeler.compute(raw_env) if self._labeler else None

        if not self.active:
            self._policy_steps += 1
            self._update_grasp_lock(action, outcome)
            return action.astype(np.float32)

        assert self._hold_r_hand is not None
        assert self._right_wrist_xyz is not None
        assert self._right_wrist_rotvec is not None

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
            print("pose_insert: peg lost -> POLICY", flush=True)
            self._deactivate()
            return action.astype(np.float32)

        if self._phase == _Phase.SETTLE:
            self._settle_steps += 1
            action[0:3] = self._right_wrist_xyz
            action[3:6] = self._right_wrist_rotvec
            action[6:22] = self._current_right_hand()
            self._apply_left_arm_hold(action)
            if self._settle_steps >= self.config.handoff_settle_frames:
                self._phase = _Phase.POSEINSERT
                self._poseinsert_steps = 0
                self._replan_counter = 0
                self._waypoints = []
                self._waypoint_idx = 0
                print("pose_insert: settle done -> POSEINSERT", flush=True)
            return action.astype(np.float32)

        if self._phase == _Phase.POSEINSERT:
            self._poseinsert_steps += 1
            if self._poseinsert_steps > self.config.max_poseinsert_steps:
                if self._poseinsert_steps == self.config.max_poseinsert_steps + 1:
                    print("pose_insert: POSEINSERT timeout", flush=True)
            else:
                self._poseinsert_step(raw_env)
            if self._should_release_grasp(raw_env):
                self._insert_streak += 1
            else:
                self._insert_streak = 0
            if self._insert_streak >= self.config.release_confirm_frames:
                self._phase = _Phase.RELEASE
                self._release_steps = 0
                self._grasp_locked_hand = None
                self._hold_l_arm = None
                print("pose_insert: POSEINSERT -> RELEASE", flush=True)
        elif self._phase == _Phase.RELEASE:
            self._release_steps += 1

        action[0:3] = self._right_wrist_xyz
        action[3:6] = self._right_wrist_rotvec
        action[6:22] = self._current_right_hand()
        self._apply_left_arm_hold(action)
        return action.astype(np.float32)

    def episode_summary(self) -> str:
        return (
            f"handoff={self._handoff_happened} phase={self.phase_name} "
            f"grasp_locked={self._grasp_locked_hand is not None}"
        )

    def _activate(self, policy_action44: np.ndarray, raw_env) -> None:
        action = np.asarray(policy_action44, dtype=np.float64).reshape(-1).copy()
        locked = self._grasp_locked_hand
        self._hold_r_hand = (locked if locked is not None else action[6:22]).copy()
        self._hold_l_arm = (
            action[22:44].copy() if self.config.freeze_left_arm_at_handoff else None
        )
        self._open_r_hand = self._compute_open_hand_pose(self._hold_r_hand)
        self._right_wrist_xyz = action[0:3].copy()
        self._right_wrist_rotvec = action[3:6].copy()

        source_pose7, _ = read_sim_poses7(raw_env)
        quat_xyzw = R.from_rotvec(self._right_wrist_rotvec).as_quat()
        wrist_pose7 = np.concatenate([self._right_wrist_xyz, quat_xyzw], dtype=np.float64)
        self._peg_to_wrist4 = calibrate_peg_to_wrist(source_pose7, wrist_pose7)

        self._phase = _Phase.SETTLE
        self._handoff_happened = True
        self._settle_steps = 0
        self._insert_streak = 0
        self._peg_lost_streak = 0
        self._waypoints = []
        self._waypoint_idx = 0
        print(
            "pose_insert: pre_insert -> SETTLE "
            f"peg_dz={self._peg_dz(raw_env):.3f} left_frozen={self._hold_l_arm is not None}",
            flush=True,
        )

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
        assert self._peg_to_wrist4 is not None
        assert self._right_wrist_xyz is not None
        assert self._right_wrist_rotvec is not None

        cfg = self.config
        need_replan = (
            self._waypoint_idx >= len(self._waypoints)
            or self._replan_counter % max(1, cfg.replan_interval) == 0
        )
        if need_replan:
            source_pose7, target_pose7 = read_sim_poses7(raw_env)
            workspace = self.policy.workspace if self.policy.normalize else None
            obs_pose9 = build_obs_pose9(
                source_pose7,
                target_pose7,
                workspace=workspace if cfg.normalize_translation else None,
            )
            horizon = self.policy.predict_pose9_horizon(obs_pose9)
            self._waypoints = [horizon[i] for i in range(horizon.shape[0])]
            self._waypoint_idx = 0
            self._replan_counter = 0

        pose9 = self._waypoints[self._waypoint_idx]
        self._waypoint_idx += 1
        self._replan_counter += 1

        _, target_pose7 = read_sim_poses7(raw_env)
        workspace = self.policy.workspace if self.policy.normalize else None
        source_pose7 = relative_pose9_to_world_source(
            pose9,
            target_pose7,
            workspace=workspace if cfg.normalize_translation else None,
        )
        wrist_pose7 = source_pose7_to_wrist_pose7(source_pose7, self._peg_to_wrist4)
        tgt_xyz, tgt_rotvec = wrist_pose7_to_rotvec_action(wrist_pose7)
        new_xyz, new_rotvec = clamp_wrist_target(
            self._right_wrist_xyz,
            self._right_wrist_rotvec,
            tgt_xyz,
            tgt_rotvec,
            max_step_m=cfg.max_wrist_step_m,
            max_rot_step_rad=cfg.max_wrist_rot_step_rad,
        )
        self._right_wrist_xyz = new_xyz
        self._right_wrist_rotvec = new_rotvec

    def _update_grasp_lock(self, action: np.ndarray, outcome) -> None:
        if outcome is None:
            return
        if outcome.peg_ok and outcome.tray_ok:
            self._grasp_lock_streak += 1
            if self._grasp_lock_streak >= self.config.grasp_lock_frames:
                if self._grasp_locked_hand is None:
                    self._grasp_locked_hand = action[6:22].copy()
                    print("pose_insert: right grasp locked", flush=True)
                else:
                    self._grasp_locked_hand = np.maximum(self._grasp_locked_hand, action[6:22])
        else:
            self._grasp_lock_streak = 0
            if not outcome.peg_ok:
                self._grasp_locked_hand = None
        if self._grasp_locked_hand is not None and outcome.peg_ok:
            action[6:22] = self._grasp_locked_hand

    def _apply_left_arm_hold(self, action: np.ndarray) -> None:
        if self._hold_l_arm is not None:
            action[22:44] = self._hold_l_arm

    def _bind_env(self, raw_env) -> None:
        if self._labeler is not None:
            return
        self._labeler = AssemblyContactLabeler(
            raw_env,
            lift_threshold_m=self.config.lift_threshold_m,
        )
        model = raw_env._model
        self._socket_site_id = int(model.site(_SOCKET_SITE).id)
        self._peg_body_id = int(model.body(PEG_BODY).id)
        self._bottom_geom_id = int(model.geom(_BOTTOM_GEOM).id)

    def _hole_axis(self, raw_env) -> np.ndarray:
        data = raw_env._data
        socket_pos = np.asarray(data.site_xpos[self._socket_site_id], dtype=np.float64)
        socket_xmat = np.asarray(data.site_xmat[self._socket_site_id], dtype=np.float64)
        bottom_pos = np.asarray(data.geom_xpos[self._bottom_geom_id], dtype=np.float64)
        return hole_opening_axis(socket_pos, socket_xmat, bottom_pos)

    def _peg_dz(self, raw_env) -> float:
        if self._peg_rest_z is None:
            return 0.0
        return float(raw_env._data.xpos[self._peg_body_id, 2]) - self._peg_rest_z

    def _peg_insert_pos(self, raw_env) -> np.ndarray:
        data = raw_env._data
        return peg_insert_end_pos(
            data.xpos[self._peg_body_id],
            data.xmat[self._peg_body_id],
        )

    def _socket_pos(self, raw_env) -> np.ndarray:
        return np.asarray(raw_env._data.site_xpos[self._socket_site_id], dtype=np.float64)

    def _in_approach_cylinder(self, raw_env) -> bool:
        cfg = self.config
        return in_approach_cylinder(
            self._peg_insert_pos(raw_env),
            self._socket_pos(raw_env),
            self._hole_axis(raw_env),
            xy_tol_m=cfg.approach_xy_m,
            z_min_m=cfg.approach_z_min_m,
            z_max_m=cfg.approach_z_max_m,
        )

    def _insert_socket_dist(self, raw_env) -> float:
        return tip_socket_distance(self._peg_insert_pos(raw_env), self._socket_pos(raw_env))

    def _should_release_grasp(self, raw_env) -> bool:
        return self._insert_socket_dist(raw_env) <= self.config.release_insert_socket_dist_m

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
