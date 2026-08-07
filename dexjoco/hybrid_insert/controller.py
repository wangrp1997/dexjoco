"""Privileged geometry controller for right-arm peg alignment and insertion."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np
from scipy.spatial.transform import Rotation as R

from .assembly_contacts import AssemblyContactLabeler

from .config import HybridInsertConfig
from .geometry import (
    axis_parallel_error_rad,
    body_z_axis,
    height_along_axis,
    hole_opening_axis,
    in_approach_cylinder,
    insert_along_hole_delta,
    lateral_align_delta,
    lateral_error,
    line_align_target_axis,
    pbvs_tip_feature_error,
    pbvs_tip_velocity,
    peg_insert_end_pos,
    rotation_world_from_to,
    tip_socket_distance,
    wrist_rotvec_align_peg_axis,
)


class _Phase(Enum):
    POLICY = auto()
    ALIGN = auto()
    INSERT = auto()
    RELEASE = auto()


class HybridInsertController:
    """Dual-arm relative PBVS: both arms reduce tip↔socket feature error."""

    _SOCKET_SITE = "industreal_tray_insert_round_peg_8mm_socket_site"
    _PEG_BODY = "industreal_round_peg_8mm"
    _TRAY_BODY = "industreal_tray_insert_round_peg_8mm"
    _BOTTOM_GEOM = "industreal_tray_insert_round_peg_8mm_bottom_contact"

    def __init__(self, config: HybridInsertConfig | None = None) -> None:
        self.config = config or HybridInsertConfig()
        self._labeler: AssemblyContactLabeler | None = None
        self._socket_site_id: int | None = None
        self._peg_body_id: int | None = None
        self._tray_body_id: int | None = None
        self._bottom_geom_id: int | None = None
        self._peg_rest_z: float | None = None
        self._phase = _Phase.POLICY
        self._handoff_streak = 0
        self._grasp_lock_streak = 0
        self._insert_streak = 0
        self._insert_align_streak = 0
        self._align_steps = 0
        self._insert_steps = 0
        self._release_steps = 0
        self._policy_steps = 0
        self._settle_steps = 0
        self._peg_lost_streak = 0
        self._handoff_happened = False
        self._grasp_locked_hand: np.ndarray | None = None
        self._hold_r_hand: np.ndarray | None = None
        self._hold_l_arm: np.ndarray | None = None  # legacy full freeze
        self._hold_l_hand: np.ndarray | None = None
        self._open_r_hand: np.ndarray | None = None
        self._right_wrist_xyz: np.ndarray | None = None
        self._right_wrist_rotvec: np.ndarray | None = None
        self._left_wrist_xyz: np.ndarray | None = None
        self._left_wrist_rotvec: np.ndarray | None = None
        self._anchor_wrist_rotvec: np.ndarray | None = None
        self._handoff_insert_socket_dist: float | None = None
        self._stall_frames = 0
        self._retreat_left = 0
        self._best_along = float("inf")
        self._best_tip = float("inf")
        self._tip_jam_frames = 0
        self.last_diag: dict = {}

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
    def needs_policy_left(self) -> bool:
        if not self.active:
            return False
        return self._hold_l_arm is None

    def reset(self, raw_env) -> None:
        self._bind_env(raw_env)
        assert self._labeler is not None
        self._labeler.reset_reference(raw_env)
        data = raw_env._data
        self._peg_rest_z = float(data.xpos[self._peg_body_id, 2])
        self._phase = _Phase.POLICY
        self._handoff_streak = 0
        self._grasp_lock_streak = 0
        self._insert_streak = 0
        self._insert_align_streak = 0
        self._align_steps = 0
        self._insert_steps = 0
        self._release_steps = 0
        self._policy_steps = 0
        self._settle_steps = 0
        self._peg_lost_streak = 0
        self._handoff_happened = False
        self._grasp_locked_hand = None
        self._hold_r_hand = None
        self._hold_l_arm = None
        self._hold_l_hand = None
        self._open_r_hand = None
        self._right_wrist_xyz = None
        self._right_wrist_rotvec = None
        self._left_wrist_xyz = None
        self._left_wrist_rotvec = None
        self._anchor_wrist_rotvec = None
        self._handoff_insert_socket_dist = None
        self._stall_frames = 0
        self._retreat_left = 0
        self._best_along = float("inf")
        self._best_tip = float("inf")
        self._tip_jam_frames = 0
        self.last_diag = {}

    def update_handoff(self, raw_env, policy_action44: np.ndarray) -> None:
        """Check approach-cylinder readiness; activate right-arm hybrid when stable."""
        if self.active:
            return
        self._bind_env(raw_env)
        assert self._labeler is not None
        outcome = self._labeler.compute(raw_env)
        peg_dz = self._peg_dz(raw_env)
        in_cylinder = self._in_approach_cylinder(raw_env)

        ready = (
            outcome.tray_ok
            and outcome.peg_ok
            and peg_dz >= self.config.lift_ready_m
            and in_cylinder
        )
        if ready:
            self._handoff_streak += 1
        else:
            self._handoff_streak = 0

        self._maybe_log_handoff_progress(raw_env, outcome, peg_dz, in_cylinder)

        if self._handoff_streak >= self.config.handoff_confirm_frames:
            self._activate(policy_action44, raw_env)

    def merge_right_arm(self, raw_env, policy_action44: np.ndarray) -> np.ndarray:
        """Return full 44d action: policy left arm + hybrid right arm."""
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
            elif self._phase in (_Phase.ALIGN, _Phase.INSERT):
                self._peg_lost_streak += 1
            if self.last_diag is not None:
                self.last_diag["peg_ok"] = bool(outcome.peg_ok)

        if (
            self._phase in (_Phase.ALIGN, _Phase.INSERT)
            and outcome is not None
            and self._peg_lost_streak >= self.config.peg_lost_abort_frames
        ):
            if self._geometry_seated(raw_env):
                # Peg often loses hand-contact / "lifted" once it seats — RELEASE + open.
                print(
                    "hybrid_insert: peg_ok lost but fully seated -> RELEASE",
                    flush=True,
                )
                self._begin_release(raw_env)
            elif self._geometry_entering(raw_env):
                # Rim entry: keep dual PBVS + tray z-up; do not abort mid-insert.
                if self._peg_lost_streak == self.config.peg_lost_abort_frames:
                    print(
                        "hybrid_insert: peg_ok flicker while entering — keep inserting",
                        flush=True,
                    )
                self._peg_lost_streak = 0
            else:
                print("hybrid_insert: peg lost -> POLICY", flush=True)
                self._deactivate()
                return action.astype(np.float32)

        if self._in_handoff_settle():
            self._settle_steps += 1
            self._right_wrist_xyz = action[0:3].copy()
            self._right_wrist_rotvec = action[3:6].copy()
            if self._left_wrist_xyz is not None:
                self._left_wrist_xyz = action[22:25].copy()
                self._left_wrist_rotvec = action[25:28].copy()
            action[6:22] = self._current_right_hand()
            self._apply_left_arm_cmd(action)
            if self._settle_steps == self.config.handoff_settle_frames:
                print("hybrid_insert: settle done -> slow ALIGN", flush=True)
            return action.astype(np.float32)

        if self._phase == _Phase.ALIGN:
            self._align_step(raw_env)
            if self._is_xy_axis_aligned(raw_env):
                self._insert_align_streak += 1
            else:
                self._insert_align_streak = 0
            if self._insert_align_streak >= self.config.insert_align_confirm_frames:
                lat_err, axis_err = self._alignment_errors(raw_env)
                dist = self._insert_socket_dist(raw_env)
                self._freeze_wrist_orientation()
                self._phase = _Phase.INSERT
                self._insert_steps = 0
                self._stall_frames = 0
                self._retreat_left = 0
                self._tip_jam_frames = 0
                self._best_along = height_along_axis(
                    self._peg_insert_pos(raw_env),
                    self._socket_pos(raw_env),
                    self._hole_axis(raw_env),
                )
                self._best_tip = self._insert_socket_dist(raw_env)
                print(
                    "hybrid_insert: ALIGN -> INSERT "
                    f"lat={lat_err*1000:.1f}mm axis={np.degrees(axis_err):.1f}deg "
                    f"insert-socket={dist*1000:.1f}mm",
                    flush=True,
                )
        elif self._phase == _Phase.INSERT:
            self._insert_step(raw_env)
            if self._should_release_grasp(raw_env):
                self._insert_streak += 1
            else:
                self._insert_streak = 0
            if self._insert_streak >= self.config.release_confirm_frames:
                self._begin_release(raw_env)
        elif self._phase == _Phase.RELEASE:
            self._release_step(raw_env)

        action[0:3] = self._right_wrist_xyz
        action[3:6] = self._right_wrist_rotvec
        action[6:22] = self._current_right_hand()
        self._apply_left_arm_cmd(action)
        return action.astype(np.float32)

    def episode_summary(self) -> str:
        return (
            f"handoff={self._handoff_happened} phase={self.phase_name} "
            f"grasp_locked={self._grasp_locked_hand is not None}"
        )

    def _update_grasp_lock(self, action: np.ndarray, outcome) -> None:
        """Keep right Allegro closed while peg is held, before/during policy phase."""
        if outcome is None:
            return
        if outcome.peg_ok and outcome.tray_ok:
            self._grasp_lock_streak += 1
            if self._grasp_lock_streak >= self.config.grasp_lock_frames:
                if self._grasp_locked_hand is None:
                    self._grasp_locked_hand = action[6:22].copy()
                    print("hybrid_insert: right grasp locked", flush=True)
                else:
                    # Keep the tighter (larger-magnitude) closure seen so far.
                    self._grasp_locked_hand = np.maximum(
                        self._grasp_locked_hand, action[6:22]
                    )
        else:
            if self._grasp_locked_hand is not None and not outcome.peg_ok:
                print("hybrid_insert: grasp lock cleared (peg lost)", flush=True)
            self._grasp_lock_streak = 0
            if not outcome.peg_ok:
                self._grasp_locked_hand = None

        if self._grasp_locked_hand is not None and outcome.peg_ok:
            action[6:22] = self._grasp_locked_hand

    def _maybe_log_handoff_progress(
        self, raw_env, outcome, peg_dz: float, in_cylinder: bool
    ) -> None:
        if outcome is None or not outcome.peg_ok:
            return
        interval = self.config.handoff_debug_interval
        if interval <= 0 or self._policy_steps % interval != 0:
            return
        insert = self._peg_insert_pos(raw_env)
        socket = self._socket_pos(raw_env)
        hole_axis = self._hole_axis(raw_env)
        lat, _ = lateral_error(insert, socket, hole_axis)
        dist = self._insert_socket_dist(raw_env)
        cfg = self.config
        lat_ok = lat <= cfg.approach_xy_m
        print(
            "hybrid_insert: waiting handoff "
            f"tray_ok={outcome.tray_ok} peg_dz={peg_dz:.3f} "
            f"lat={lat*1000:.1f}mm({'ok' if lat_ok else 'no'}) "
            f"insert-socket={dist*1000:.1f}mm "
            f"in_zone={in_cylinder} streak={self._handoff_streak}",
            flush=True,
        )

    def _activate(self, policy_action44: np.ndarray, raw_env) -> None:
        action = np.asarray(policy_action44, dtype=np.float64).reshape(-1).copy()
        locked = self._grasp_locked_hand
        self._hold_r_hand = (locked if locked is not None else action[6:22]).copy()
        # Dual-arm PBVS by default: command left wrist, freeze left fingers only.
        self._hold_l_hand = action[28:44].copy()
        self._left_wrist_xyz = action[22:25].copy()
        self._left_wrist_rotvec = action[25:28].copy()
        self._hold_l_arm = (
            action[22:44].copy() if self.config.freeze_left_arm_at_handoff else None
        )
        self._open_r_hand = self._compute_open_hand_pose(self._hold_r_hand)
        self._right_wrist_xyz = action[0:3].copy()
        self._right_wrist_rotvec = action[3:6].copy()
        self._anchor_wrist_rotvec = None
        self._phase = _Phase.ALIGN
        self._handoff_happened = True
        self._align_steps = 0
        self._insert_steps = 0
        self._insert_streak = 0
        self._insert_align_streak = 0
        self._release_steps = 0
        self._settle_steps = 0
        self._peg_lost_streak = 0
        self._stall_frames = 0
        self._retreat_left = 0
        tip0 = self._peg_insert_pos(raw_env)
        sock0 = self._socket_pos(raw_env)
        hole0 = self._hole_axis(raw_env)
        self._best_along = height_along_axis(tip0, sock0, hole0)
        self._best_tip = tip_socket_distance(tip0, sock0)
        self._tip_jam_frames = 0
        self.last_diag = {}
        peg_axis = self._peg_axis(raw_env)
        hole_axis = self._hole_axis(raw_env)
        axis_err = axis_parallel_error_rad(peg_axis, hole_axis)
        peg_dot_hole = float(np.dot(peg_axis, hole_axis))
        self._handoff_insert_socket_dist = self._insert_socket_dist(raw_env)
        print(
            "hybrid_insert: handoff axis "
            f"line_err={np.degrees(axis_err):.1f}deg peg·hole={peg_dot_hole:+.2f} "
            f"insert-socket={self._handoff_insert_socket_dist*1000:.1f}mm",
            flush=True,
        )
        if self._hold_l_arm is not None:
            print(
                "hybrid_insert: handoff -> ALIGN (right hybrid, left frozen, settling...)",
                flush=True,
            )
        else:
            print(
                "hybrid_insert: handoff -> ALIGN (dual-arm PBVS, left tray servo, settling...)",
                flush=True,
            )

    def _deactivate(self) -> None:
        self._phase = _Phase.POLICY
        self._handoff_streak = 0
        self._insert_streak = 0
        self._insert_align_streak = 0
        self._align_steps = 0
        self._insert_steps = 0
        self._release_steps = 0
        self._settle_steps = 0
        self._peg_lost_streak = 0

    def _apply_left_arm_cmd(self, action: np.ndarray) -> None:
        """Write left wrist/hand into 44d action (dual PBVS or legacy freeze)."""
        if self._hold_l_arm is not None:
            action[22:44] = self._hold_l_arm
            return
        if self._left_wrist_xyz is None or self._left_wrist_rotvec is None:
            return
        action[22:25] = self._left_wrist_xyz
        action[25:28] = self._left_wrist_rotvec
        if self._hold_l_hand is not None:
            action[28:44] = self._hold_l_hand

    def _apply_left_arm_hold(self, action: np.ndarray) -> None:
        # Back-compat alias.
        self._apply_left_arm_cmd(action)

    def _apply_socket_delta_to_left_wrist(self, socket_delta: np.ndarray) -> None:
        """Move left wrist so the grasped tray/socket translates with socket_delta."""
        if self._left_wrist_xyz is None or self._hold_l_arm is not None:
            return
        cfg = self.config
        d = cfg.left_wrist_tip_scale * np.asarray(socket_delta, dtype=np.float64)
        d = self._clamp_norm(d, cfg.max_left_wrist_step_m)
        self._left_wrist_xyz = self._left_wrist_xyz + d

    def _freeze_wrist_orientation(self) -> None:
        assert self._right_wrist_rotvec is not None
        self._anchor_wrist_rotvec = self._right_wrist_rotvec.copy()

    def _in_handoff_settle(self) -> bool:
        return (
            self._phase == _Phase.ALIGN
            and self._settle_steps < self.config.handoff_settle_frames
        )

    @staticmethod
    def _clamp_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm <= max_norm or norm < 1e-9:
            return vec
        return vec * (max_norm / norm)

    def _bind_env(self, raw_env) -> None:
        if self._labeler is not None:
            return
        self._labeler = AssemblyContactLabeler(
            raw_env,
            lift_threshold_m=self.config.lift_threshold_m,
        )
        model = raw_env._model
        self._socket_site_id = int(model.site(self._SOCKET_SITE).id)
        self._peg_body_id = int(model.body(self._PEG_BODY).id)
        self._tray_body_id = int(model.body(self._TRAY_BODY).id)
        self._bottom_geom_id = int(model.geom(self._BOTTOM_GEOM).id)

    def _hole_axis(self, raw_env) -> np.ndarray:
        data = raw_env._data
        socket_pos = np.asarray(data.site_xpos[self._socket_site_id], dtype=np.float64)
        socket_xmat = np.asarray(data.site_xmat[self._socket_site_id], dtype=np.float64)
        bottom_pos = np.asarray(data.geom_xpos[self._bottom_geom_id], dtype=np.float64)
        return hole_opening_axis(socket_pos, socket_xmat, bottom_pos)

    def _peg_axis(self, raw_env) -> np.ndarray:
        data = raw_env._data
        return body_z_axis(data.xmat[self._peg_body_id])

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
        data = raw_env._data
        return np.asarray(data.site_xpos[self._socket_site_id], dtype=np.float64)

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
        return tip_socket_distance(
            self._peg_insert_pos(raw_env), self._socket_pos(raw_env)
        )

    def _alignment_errors(self, raw_env) -> tuple[float, float]:
        insert = self._peg_insert_pos(raw_env)
        socket = self._socket_pos(raw_env)
        hole_axis = self._hole_axis(raw_env)
        lat_err, _ = lateral_error(insert, socket, hole_axis)
        axis_err = axis_parallel_error_rad(self._peg_axis(raw_env), hole_axis)
        return lat_err, axis_err

    def _is_xy_axis_aligned(self, raw_env) -> bool:
        cfg = self.config
        lat_err, axis_err = self._alignment_errors(raw_env)
        return lat_err <= cfg.pos_tol_m and axis_err <= cfg.angle_tol_rad

    def _begin_release(self, raw_env) -> None:
        """Stop pushing: freeze wrists, slight retract, open right hand."""
        if self._phase == _Phase.RELEASE:
            return
        cfg = self.config
        self._phase = _Phase.RELEASE
        self._release_steps = 0
        self._grasp_locked_hand = None
        self._retreat_left = 0
        # Freeze left tray completely (no more z-up / lateral).
        if self._left_wrist_xyz is not None and self._left_wrist_rotvec is not None:
            self._hold_l_arm = np.concatenate(
                [
                    np.asarray(self._left_wrist_xyz, dtype=np.float64),
                    np.asarray(self._left_wrist_rotvec, dtype=np.float64),
                    np.asarray(
                        self._hold_l_hand
                        if self._hold_l_hand is not None
                        else np.zeros(16, dtype=np.float64),
                        dtype=np.float64,
                    ),
                ]
            )
        # Retract right wrist slightly out of the hole so opening fingers does not eject.
        if self._right_wrist_xyz is not None:
            hole = self._hole_axis(raw_env)
            axis = hole / (np.linalg.norm(hole) + 1e-8)
            self._right_wrist_xyz = (
                self._right_wrist_xyz + axis * float(cfg.release_retract_m)
            )
        print(
            "hybrid_insert: INSERT -> RELEASE "
            "(retract + open right hand, left frozen)",
            flush=True,
        )

    def _geometry_seated(self, raw_env) -> bool:
        """RELEASE when tip ~成功集 depth (~35–40mm), never early 55mm fake seat."""
        tip = self._peg_insert_pos(raw_env)
        socket = self._socket_pos(raw_env)
        hole = self._hole_axis(raw_env)
        dist = tip_socket_distance(tip, socket)
        along = height_along_axis(tip, socket, hole)
        lat, _ = lateral_error(tip, socket, hole)
        cfg = self.config
        if lat > cfg.seated_lat_m:
            return False
        if dist <= cfg.release_insert_socket_dist_m:
            return True
        if along <= cfg.seated_along_m:
            return True
        if dist <= cfg.soft_seat_tip_m:
            return True
        # Jammed but already fairly deep: release rather than wait forever (still no ram).
        if (
            dist <= 0.050
            and getattr(self, "_tip_jam_frames", 0) >= cfg.tip_jam_frames * 2
        ):
            return True
        return False

    def _geometry_entering(self, raw_env) -> bool:
        """Tip past rim / partially in — keep pushing + tray z-up, do not RELEASE yet."""
        tip = self._peg_insert_pos(raw_env)
        socket = self._socket_pos(raw_env)
        hole = self._hole_axis(raw_env)
        dist = tip_socket_distance(tip, socket)
        along = height_along_axis(tip, socket, hole)
        lat, _ = lateral_error(tip, socket, hole)
        return dist <= 0.095 and lat <= 0.015 and along <= 0.100

    def _should_release_grasp(self, raw_env) -> bool:
        return self._geometry_seated(raw_env)

    def _apply_left_wrist_rot_about_socket(
        self, raw_env, rotvec_world: np.ndarray, *, max_step_rad: float
    ) -> None:
        """Apply world rotation to left wrist, pivoting about the socket site."""
        if self._left_wrist_xyz is None or self._left_wrist_rotvec is None:
            return
        if self._hold_l_arm is not None:
            return
        cfg = self.config
        corr = np.asarray(rotvec_world, dtype=np.float64)
        ang = float(np.linalg.norm(corr))
        if ang < 1e-8:
            return
        step = min(float(max_step_rad), ang)
        r_step = R.from_rotvec(corr * (step / ang))
        socket = self._socket_pos(raw_env)
        wrist = np.asarray(self._left_wrist_xyz, dtype=np.float64)
        r_old = R.from_rotvec(self._left_wrist_rotvec)
        r_new = r_step * r_old
        wrist_in_sock = wrist - socket
        wrist_new = socket + r_step.apply(wrist_in_sock)
        wrist_delta = self._clamp_norm(wrist_new - wrist, cfg.max_left_wrist_step_m * 1.5)
        self._left_wrist_xyz = wrist + wrist_delta
        self._left_wrist_rotvec = r_new.as_rotvec()

    def _apply_hole_axis_align_to_left(
        self,
        raw_env,
        peg_axis: np.ndarray,
        hole_axis: np.ndarray,
        *,
        gain: float | None = None,
    ) -> None:
        """Relative PBVS: rotate hole toward peg about socket (left/tray)."""
        cfg = self.config
        target = line_align_target_axis(hole_axis, peg_axis)
        err = axis_parallel_error_rad(hole_axis, peg_axis)
        if err <= cfg.angle_tol_rad:
            return
        r_corr = rotation_world_from_to(hole_axis, target)
        corr = r_corr.as_rotvec()
        ang = float(np.linalg.norm(corr))
        if ang < 1e-8:
            return
        g = float(cfg.pbvs_lambda_rot if gain is None else gain)
        max_step = min(cfg.max_left_wrist_rot_step_rad, g * ang)
        self._apply_left_wrist_rot_about_socket(raw_env, corr, max_step_rad=max_step)

    def _apply_tray_z_up(self, raw_env) -> None:
        """Optional absolute upright: hole_axis → world +Z (NOT relative PBVS)."""
        if not getattr(self.config, "tray_z_up_enable", False):
            return
        cfg = self.config
        hole = self._hole_axis(raw_env)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        target = line_align_target_axis(hole, world_up)
        err = axis_parallel_error_rad(hole, world_up)
        if err <= cfg.tray_z_up_tol_rad:
            return
        r_corr = rotation_world_from_to(hole, target)
        corr = r_corr.as_rotvec()
        ang = float(np.linalg.norm(corr))
        if ang < 1e-8:
            return
        max_step = min(cfg.max_left_wrist_rot_step_rad, cfg.tray_z_up_gain * ang)
        self._apply_left_wrist_rot_about_socket(raw_env, corr, max_step_rad=max_step)

    def _apply_relative_axis_align(
        self,
        raw_env,
        peg_axis: np.ndarray,
        hole_axis: np.ndarray,
        *,
        tip: np.ndarray,
        gain: float | None = None,
    ) -> None:
        """Split relative axis error: right twists peg, left twists hole toward each other."""
        cfg = self.config
        left_share = float(np.clip(cfg.left_share_rot, 0.0, 0.85))
        g = float(cfg.pbvs_lambda_rot if gain is None else gain)
        if left_share < 0.999:
            self._apply_peg_axis_align_to_wrist(
                peg_axis,
                hole_axis,
                tip=tip,
                gain=g * (1.0 - left_share),
            )
        if left_share > 1e-6:
            self._apply_hole_axis_align_to_left(
                raw_env,
                peg_axis,
                hole_axis,
                gain=g * left_share,
            )

    def _apply_tip_delta_to_wrist(self, tip_delta: np.ndarray) -> None:
        assert self._right_wrist_xyz is not None
        cfg = self.config
        wrist_delta = cfg.wrist_tip_scale * np.asarray(tip_delta, dtype=np.float64)
        wrist_delta = self._clamp_norm(wrist_delta, cfg.max_wrist_step_m)
        self._right_wrist_xyz = self._right_wrist_xyz + wrist_delta

    def _apply_rot_delta_to_wrist(self, rot_delta: np.ndarray) -> None:
        assert self._right_wrist_rotvec is not None
        cfg = self.config
        rot_delta = self._clamp_norm(
            np.asarray(rot_delta, dtype=np.float64), cfg.max_wrist_rot_step_rad
        )
        r_rot = R.from_rotvec(self._right_wrist_rotvec)
        new_rotvec = (r_rot * R.from_rotvec(rot_delta)).as_rotvec()

        if self._anchor_wrist_rotvec is not None and self._phase == _Phase.INSERT:
            r_anchor = R.from_rotvec(self._anchor_wrist_rotvec)
            r_new = R.from_rotvec(new_rotvec)
            delta = r_anchor.inv() * r_new
            delta_rotvec = delta.as_rotvec()
            angle = float(np.linalg.norm(delta_rotvec))
            max_angle = cfg.max_wrist_rot_from_anchor_rad
            if angle > max_angle:
                delta_rotvec = delta_rotvec * (max_angle / angle)
                r_new = r_anchor * R.from_rotvec(delta_rotvec)
                new_rotvec = r_new.as_rotvec()

        self._right_wrist_rotvec = new_rotvec

    def _align_step(self, raw_env) -> None:
        cfg = self.config
        self._align_steps += 1
        if self._align_steps > cfg.max_align_steps:
            if self._align_steps == cfg.max_align_steps + 1:
                feat = self._pbvs_features(raw_env, target_along_m=cfg.pbvs_standoff_m)
                print(
                    "hybrid_insert: ALIGN timeout "
                    f"lat={float(feat['lat_err'])*1000:.1f}mm "
                    f"insert-socket={float(feat['tip_socket_dist'])*1000:.1f}mm "
                    f"axis={np.degrees(float(feat['axis_err'])):.1f}deg",
                    flush=True,
                )
            return
        self._pbvs_servo(raw_env, mode="align")
        interval = cfg.align_debug_interval
        if interval > 0 and self._align_steps % interval == 0:
            d = self.last_diag
            print(
                "hybrid_insert: ALIGN "
                f"step={self._align_steps} lat={d.get('lat_mm', float('nan')):.1f}mm "
                f"along={d.get('along_mm', float('nan')):.1f}mm "
                f"axis={d.get('axis_deg', float('nan')):.1f}deg "
                f"streak={self._insert_align_streak} mode={d.get('mode')}",
                flush=True,
            )

    def _insert_step(self, raw_env) -> None:
        cfg = self.config
        self._insert_steps += 1
        if self._insert_steps > cfg.max_insert_steps:
            return
        if self._should_release_grasp(raw_env):
            return
        self._pbvs_servo(raw_env, mode="insert")
        if (
            cfg.insert_debug_interval > 0
            and self._insert_steps % cfg.insert_debug_interval == 0
        ):
            d = self.last_diag
            print(
                f"hybrid_insert: INSERT step={self._insert_steps} "
                f"tip={d.get('tip_mm', float('nan')):.1f}mm "
                f"lat={d.get('lat_mm', float('nan')):.1f}mm "
                f"along={d.get('along_mm', float('nan')):.1f}mm "
                f"axis={d.get('axis_deg', float('nan')):.1f}deg "
                f"stall={self._stall_frames} retreat={self._retreat_left}",
                flush=True,
            )

    def _pbvs_features(self, raw_env, *, target_along_m: float) -> dict:
        tip = self._peg_insert_pos(raw_env)
        socket = self._socket_pos(raw_env)
        hole_axis = self._hole_axis(raw_env)
        peg_axis = self._peg_axis(raw_env)
        return pbvs_tip_feature_error(
            tip,
            socket,
            hole_axis,
            peg_axis,
            target_along_m=target_along_m,
        )

    def _pbvs_servo(self, raw_env, *, mode: str) -> None:
        """Dual-arm relative PBVS: both arms reduce tip↔socket feature error e.

        Lateral: tip -(1-α)λ e_lat, socket +α λ e_lat.
        Along: primarily right tip along hole.
        Axis: split relative align peg↔hole (not absolute world +Z).
        """
        cfg = self.config
        tip = self._peg_insert_pos(raw_env)
        socket = self._socket_pos(raw_env)
        hole_axis = self._hole_axis(raw_env)
        peg_axis = self._peg_axis(raw_env)

        if mode == "align":
            target_along = cfg.pbvs_standoff_m
        else:
            target_along = cfg.pbvs_insert_target_along_m

        feat = pbvs_tip_feature_error(
            tip, socket, hole_axis, peg_axis, target_along_m=target_along
        )
        lat_err = float(feat["lat_err"])
        lat_vec = np.asarray(feat["lat_vec"], dtype=np.float64)
        along = float(feat["along"])
        axis_err = float(feat["axis_err"])
        tip_dist = float(feat["tip_socket_dist"])
        deep = tip_dist <= cfg.stop_lateral_tip_m or along <= cfg.seated_along_m

        enter_tip = float(getattr(cfg, "pbvs_enter_tip_m", 0.100))
        entering = tip_dist <= enter_tip and lat_err <= 0.015
        jam_hold = False
        jam_zone = tip_dist <= max(float(cfg.soft_seat_tip_m) + 0.017, 0.055)
        if mode == "insert":
            improved = tip_dist + cfg.tip_jam_improve_m < self._best_tip
            if improved:
                self._best_tip = tip_dist
                self._tip_jam_frames = 0
            elif jam_zone:
                self._tip_jam_frames += 1
                prev_axis = None
                if self.last_diag and "axis_deg" in self.last_diag:
                    prev_axis = float(self.last_diag["axis_deg"])
                if (
                    prev_axis is not None
                    and float(np.degrees(axis_err)) > prev_axis + 0.8
                    and tip_dist > self._best_tip - 0.002
                ):
                    self._tip_jam_frames = max(
                        self._tip_jam_frames, cfg.tip_jam_frames
                    )
            else:
                self._tip_jam_frames = 0
            jam_hold = jam_zone and self._tip_jam_frames >= cfg.tip_jam_frames
            if jam_hold and self._tip_jam_frames == cfg.tip_jam_frames:
                print(
                    f"hybrid_insert: tip jammed — stop Z push "
                    f"tip={tip_dist*1000:.1f}mm best={self._best_tip*1000:.1f}mm "
                    f"axis={float(np.degrees(axis_err)):.1f}deg",
                    flush=True,
                )

        if mode == "insert" and not deep and not entering:
            prev_tip = tip_dist
            if self.last_diag and "tip_mm" in self.last_diag:
                prev_tip = float(self.last_diag["tip_mm"]) / 1000.0
            if along + 5e-4 < self._best_along:
                self._best_along = along
                self._stall_frames = 0
            else:
                self._stall_frames += 1
            near_hole = tip_dist < 0.12
            if near_hole and self._retreat_left <= 0 and tip_dist > prev_tip + 0.015:
                self._retreat_left = cfg.pbvs_retreat_frames
                self._stall_frames = 0
                print(
                    f"hybrid_insert: tip bounce {prev_tip*1000:.1f}->{tip_dist*1000:.1f}mm -> retreat",
                    flush=True,
                )
            elif (
                near_hole
                and self._retreat_left <= 0
                and self._stall_frames >= cfg.pbvs_stall_frames
                and lat_err <= max(cfg.pos_tol_m * 3.0, 0.012)
            ):
                self._retreat_left = cfg.pbvs_retreat_frames
                self._stall_frames = 0
                print(
                    f"hybrid_insert: PBVS stall -> retreat "
                    f"along={along*1000:.1f}mm tip={tip_dist*1000:.1f}mm",
                    flush=True,
                )
        elif mode == "insert" and (deep or entering) and self._retreat_left <= 0:
            self._stall_frames = 0
            if along + 5e-4 < self._best_along:
                self._best_along = along

        mode_tag = mode
        left_share = float(np.clip(cfg.left_share_xy, 0.0, 0.85))
        soft = tip_dist <= cfg.soft_seat_tip_m and lat_err <= cfg.seated_lat_m
        do_lateral = (
            (not deep) and (not entering) and (not soft) and lat_err > cfg.pos_tol_m * 0.5
        )

        # Relative axis: peg↔hole toward each other (classic dual-arm relative feature).
        # Deep contact (tip≤60mm): freeze twist — twisting pries the peg out (ep91).
        rel_axis = False
        tray_up = False
        rel_axis_ok = tip_dist > float(
            getattr(cfg, "pbvs_rel_axis_min_tip_m", 0.060)
        )
        if (
            (not soft)
            and rel_axis_ok
            and axis_err > cfg.angle_tol_rad
            and lat_err <= max(cfg.axis_align_max_lat_m * 2.0, 0.012)
        ):
            self._apply_relative_axis_align(
                raw_env,
                peg_axis,
                hole_axis,
                tip=tip,
                gain=cfg.pbvs_lambda_rot * (0.7 if mode == "insert" else 1.0),
            )
            rel_axis = True
            mode_tag = "rel_axis"
        # Absolute world upright is opt-in only (conflicts with relative PBVS).
        if (
            getattr(cfg, "tray_z_up_enable", False)
            and mode == "insert"
            and tip_dist <= cfg.tray_z_up_enable_tip_m
            and tip_dist > cfg.tray_z_up_disable_tip_m
            and not soft
        ):
            self._apply_tray_z_up(raw_env)
            tray_up = True
            mode_tag = "tray_z_up"

        allow_retreat = (
            self._retreat_left > 0
            and not deep
            and (not entering or jam_hold or tip_dist > cfg.soft_seat_tip_m + 0.01)
        )

        # Deep jam: stop Z forever-creep; only micro-nudge hole with left arm.
        deep_jam_nudge = (
            mode == "insert"
            and jam_hold
            and tip_dist <= 0.060
            and tip_dist > cfg.soft_seat_tip_m
            and lat_err <= cfg.seated_lat_m * 1.5
        )

        if soft:
            mode_tag = "soft_seat"
            v_tip = np.zeros(3, dtype=np.float64)
            self._retreat_left = 0
        elif allow_retreat:
            mode_tag = "retreat"
            self._retreat_left -= 1
            axis = hole_axis / (np.linalg.norm(hole_axis) + 1e-8)
            v_right = axis * cfg.pbvs_retreat_step_m
            if do_lateral:
                v_right = v_right - (1.0 - left_share) * cfg.pbvs_lambda_xy * lat_vec
                self._apply_socket_delta_to_left_wrist(
                    left_share * cfg.pbvs_lambda_xy * lat_vec
                )
            v_right = self._clamp_norm(v_right, cfg.max_wrist_step_m)
            self._apply_tip_delta_to_wrist(v_right)
            v_tip = v_right
        elif deep_jam_nudge:
            # No Z, no twist: left moves hole laterally to free the jam.
            mode_tag = "jam_nudge"
            v_tip = np.zeros(3, dtype=np.float64)
            if lat_err > cfg.pos_tol_m * 0.25:
                self._apply_socket_delta_to_left_wrist(
                    cfg.pbvs_lambda_xy * 0.35 * lat_vec
                )
        else:
            if tip_dist > 0.12:
                axis_ok_insert = True
            elif entering or deep:
                axis_ok_insert = axis_err <= cfg.angle_tol_rad * 1.6
            else:
                axis_ok_insert = axis_err <= cfg.angle_tol_rad
            if mode == "align":
                allow_z = (not do_lateral) and axis_err <= cfg.angle_tol_rad
            else:
                allow_z = (
                    ((not do_lateral) or deep or entering)
                    and (not jam_hold)
                    and axis_ok_insert
                )
            # Creep only in shallow jam band (60–65mm); deeper uses jam_nudge.
            creep_z = (
                mode == "insert"
                and jam_hold
                and tip_dist > 0.060
                and tip_dist <= 0.065
                and tip_dist > cfg.soft_seat_tip_m
                and lat_err <= cfg.seated_lat_m
                and axis_err <= cfg.angle_tol_rad * 1.6
            )
            v_tip = np.zeros(3, dtype=np.float64)
            if do_lateral:
                v_right_lat = -(1.0 - left_share) * cfg.pbvs_lambda_xy * lat_vec
                v_left_sock = left_share * cfg.pbvs_lambda_xy * lat_vec
                v_tip = v_tip + v_right_lat
                self._apply_socket_delta_to_left_wrist(v_left_sock)
            if allow_z or creep_z:
                axis = hole_axis / (np.linalg.norm(hole_axis) + 1e-8)
                e_along = float(feat["e_along"])
                z_gain = cfg.pbvs_lambda_z * (0.7 if mode == "insert" else 1.0)
                if entering:
                    z_gain *= 0.55
                if deep:
                    z_gain *= 0.35
                if creep_z:
                    z_gain *= 0.25
                    mode_tag = "creep"
                v_z = -z_gain * e_along * axis
                if mode == "insert" and (entering or deep or creep_z or tip_dist <= enter_tip):
                    z_cap = float(cfg.max_insert_z_step_m)
                    if tip_dist <= 0.070 or creep_z:
                        z_cap = min(z_cap, 0.0005)
                    zn = float(np.linalg.norm(v_z))
                    if zn > z_cap:
                        v_z = v_z * (z_cap / zn)
                v_tip = v_tip + v_z
            if jam_hold and not creep_z:
                mode_tag = "jam_hold"
                axis_u = hole_axis / (np.linalg.norm(hole_axis) + 1e-8)
                v_tip = v_tip - float(np.dot(v_tip, axis_u)) * axis_u
                # No deep rel_axis while jammed (would pry peg out).
            # Spiral only far from hole — near-rim spiral destabilized ep91.
            spiral_min = float(getattr(cfg, "pbvs_spiral_min_tip_m", 0.120))
            if (
                mode == "insert"
                and do_lateral
                and tip_dist > spiral_min
                and lat_err > cfg.pos_tol_m
                and self._stall_frames > 15
                and along < 0.13
                and not jam_hold
            ):
                mode_tag = "spiral"
                axis_u = hole_axis / (np.linalg.norm(hole_axis) + 1e-8)
                helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                if abs(float(np.dot(helper, axis_u))) > 0.9:
                    helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                ex = np.cross(axis_u, helper)
                ex = ex / (np.linalg.norm(ex) + 1e-8)
                ey = np.cross(axis_u, ex)
                ang = 0.35 * float(self._stall_frames)
                rad = 0.0012
                spiral = rad * (np.cos(ang) * ex + np.sin(ang) * ey)
                v_tip = v_tip + (1.0 - left_share) * spiral
                self._apply_socket_delta_to_left_wrist(left_share * spiral)
            keep = ("spiral", "tray_z_up", "jam_hold", "rel_axis", "creep", "jam_nudge")
            if deep and mode_tag not in keep:
                mode_tag = "seat"
            elif rel_axis and mode_tag not in ("spiral", "retreat", "jam_hold", "creep", "jam_nudge"):
                mode_tag = "rel_axis"
            elif tray_up and mode_tag not in ("spiral", "retreat", "jam_hold"):
                mode_tag = "tray_z_up"
            v_tip = self._clamp_norm(v_tip, cfg.max_wrist_step_m)
            self._apply_tip_delta_to_wrist(v_tip)

        hole_up_err = axis_parallel_error_rad(
            hole_axis, np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )
        self.last_diag = {
            "mode": mode_tag,
            "lat_mm": lat_err * 1000.0,
            "along_mm": along * 1000.0,
            "axis_deg": float(np.degrees(axis_err)),
            "tip_mm": tip_dist * 1000.0,
            "e_along_mm": float(feat["e_along"]) * 1000.0,
            "cmd_norm_mm": float(np.linalg.norm(v_tip)) * 1000.0,
            "stall": int(self._stall_frames),
            "retreat_left": int(self._retreat_left),
            "deep": bool(deep),
            "entering": bool(entering),
            "soft_seat": bool(soft),
            "tray_up": bool(tray_up),
            "rel_axis": bool(rel_axis),
            "jam_hold": bool(jam_hold),
            "tip_jam": int(self._tip_jam_frames),
            "hole_up_deg": float(np.degrees(hole_up_err)),
            "do_lateral": bool(do_lateral),
            "left_share": float(left_share),
            "allow_z": bool(
                (not soft) and (not jam_hold) and (not do_lateral or deep or entering)
            ),
            "peg_ok": None,
            "seated": bool(self._geometry_seated(raw_env)),
        }

    def _apply_peg_axis_align_to_wrist(
        self,
        peg_axis: np.ndarray,
        hole_axis: np.ndarray,
        *,
        tip: np.ndarray | None = None,
        gain: float | None = None,
    ) -> None:
        """Twist peg axis toward hole; if tip given, rotate about tip (wrist xyz compensates)."""
        assert self._right_wrist_rotvec is not None
        assert self._right_wrist_xyz is not None
        cfg = self.config
        old_rotvec = self._right_wrist_rotvec.copy()
        new_rotvec = wrist_rotvec_align_peg_axis(
            peg_axis,
            hole_axis,
            old_rotvec,
            angle_tol_rad=cfg.angle_tol_rad,
            gain=float(cfg.align_rot_gain if gain is None else gain),
            max_step_rad=cfg.max_wrist_rot_step_rad,
        )
        if new_rotvec is None:
            return
        if tip is not None:
            # Keep insert end fixed while palm orientation changes.
            tip_w = np.asarray(tip, dtype=np.float64)
            wrist = np.asarray(self._right_wrist_xyz, dtype=np.float64)
            r_old = R.from_rotvec(old_rotvec)
            r_new = R.from_rotvec(new_rotvec)
            tip_in_palm = r_old.inv().apply(tip_w - wrist)
            wrist_new = tip_w - r_new.apply(tip_in_palm)
            # Bound compensatory translation so opspace does not explode.
            wrist_delta = self._clamp_norm(wrist_new - wrist, cfg.max_wrist_step_m * 2.0)
            self._right_wrist_xyz = wrist + wrist_delta
        self._right_wrist_rotvec = new_rotvec

    def _release_step(self, raw_env=None) -> None:
        """Open hand; keep wrists frozen (optional tiny retract already applied at begin)."""
        self._release_steps += 1

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
        open_pose = np.minimum(open_pose, closed * 0.15)
        return open_pose
