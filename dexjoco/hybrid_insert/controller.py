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
    hole_opening_axis,
    in_approach_cylinder,
    insert_along_hole_delta,
    lateral_align_delta,
    lateral_error,
    peg_insert_end_pos,
    tip_socket_distance,
    wrist_rotvec_align_peg_axis,
)


class _Phase(Enum):
    POLICY = auto()
    ALIGN = auto()
    INSERT = auto()
    RELEASE = auto()


class HybridInsertController:
    """Hand off right arm after approach cylinder; left arm frozen at handoff by default."""

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
        self._hold_l_arm: np.ndarray | None = None
        self._open_r_hand: np.ndarray | None = None
        self._right_wrist_xyz: np.ndarray | None = None
        self._right_wrist_rotvec: np.ndarray | None = None
        self._anchor_wrist_rotvec: np.ndarray | None = None
        self._handoff_insert_socket_dist: float | None = None

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
        self._open_r_hand = None
        self._right_wrist_xyz = None
        self._right_wrist_rotvec = None
        self._anchor_wrist_rotvec = None
        self._handoff_insert_socket_dist = None

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

        if (
            self._phase in (_Phase.ALIGN, _Phase.INSERT)
            and outcome is not None
            and self._peg_lost_streak >= self.config.peg_lost_abort_frames
        ):
            print("hybrid_insert: peg lost -> POLICY", flush=True)
            self._deactivate()
            return action.astype(np.float32)

        if self._in_handoff_settle():
            self._settle_steps += 1
            self._right_wrist_xyz = action[0:3].copy()
            self._right_wrist_rotvec = action[3:6].copy()
            action[6:22] = self._current_right_hand()
            self._apply_left_arm_hold(action)
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
                print(
                    "hybrid_insert: ALIGN -> INSERT "
                    f"lat={lat_err*1000:.1f}mm axis={np.degrees(axis_err):.1f}deg "
                    f"insert-socket={dist*1000:.1f}mm",
                    flush=True,
                )
        elif self._phase == _Phase.INSERT:
            self._insert_step(raw_env)
            if (
                self.config.insert_debug_interval > 0
                and self._insert_steps % self.config.insert_debug_interval == 0
            ):
                dist = self._insert_socket_dist(raw_env)
                print(
                    f"hybrid_insert: INSERT step={self._insert_steps} "
                    f"insert-socket={dist*1000:.1f}mm",
                    flush=True,
                )
            if self._should_release_grasp(raw_env):
                self._insert_streak += 1
            else:
                self._insert_streak = 0
            if self._insert_streak >= self.config.release_confirm_frames:
                self._phase = _Phase.RELEASE
                self._release_steps = 0
                self._grasp_locked_hand = None
                self._hold_l_arm = None
                print(
                    "hybrid_insert: INSERT -> RELEASE "
                    "(insert end near socket, open right hand, left -> policy)",
                    flush=True,
                )
        elif self._phase == _Phase.RELEASE:
            self._release_step()

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
            print("hybrid_insert: handoff -> ALIGN (right arm only, settling...)", flush=True)

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

    def _apply_left_arm_hold(self, action: np.ndarray) -> None:
        if self._hold_l_arm is not None:
            action[22:44] = self._hold_l_arm

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

    def _should_release_grasp(self, raw_env) -> bool:
        return (
            self._insert_socket_dist(raw_env) <= self.config.release_insert_socket_dist_m
        )

    def _apply_tip_delta_to_wrist(self, tip_delta: np.ndarray) -> None:
        assert self._right_wrist_xyz is not None
        cfg = self.config
        wrist_delta = cfg.wrist_tip_scale * np.asarray(tip_delta, dtype=np.float64)
        wrist_delta = self._clamp_norm(wrist_delta, cfg.max_wrist_step_m)
        self._right_wrist_xyz = self._right_wrist_xyz + wrist_delta

    def _apply_peg_axis_align_to_wrist(self, peg_axis: np.ndarray, hole_axis: np.ndarray) -> None:
        assert self._right_wrist_rotvec is not None
        cfg = self.config
        new_rotvec = wrist_rotvec_align_peg_axis(
            peg_axis,
            hole_axis,
            self._right_wrist_rotvec,
            angle_tol_rad=cfg.angle_tol_rad,
            gain=cfg.align_rot_gain,
            max_step_rad=cfg.max_wrist_rot_step_rad,
        )
        if new_rotvec is not None:
            self._right_wrist_rotvec = new_rotvec

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
                insert = self._peg_insert_pos(raw_env)
                socket = self._socket_pos(raw_env)
                hole_axis = self._hole_axis(raw_env)
                lat_err, _ = lateral_error(insert, socket, hole_axis)
                axis_err = axis_parallel_error_rad(self._peg_axis(raw_env), hole_axis)
                dist = self._insert_socket_dist(raw_env)
                print(
                    "hybrid_insert: ALIGN timeout "
                    f"lat={lat_err*1000:.1f}mm insert-socket={dist*1000:.1f}mm "
                    f"axis={np.degrees(axis_err):.1f}deg",
                    flush=True,
                )
            return

        insert = self._peg_insert_pos(raw_env)
        socket = self._socket_pos(raw_env)
        hole_axis = self._hole_axis(raw_env)
        peg_axis = self._peg_axis(raw_env)

        lat_err, _ = lateral_error(insert, socket, hole_axis)
        axis_err = axis_parallel_error_rad(peg_axis, hole_axis)

        interval = cfg.align_debug_interval
        if interval > 0 and self._align_steps % interval == 0:
            dist = self._insert_socket_dist(raw_env)
            print(
                "hybrid_insert: ALIGN "
                f"step={self._align_steps} lat={lat_err*1000:.1f}mm "
                f"insert-socket={dist*1000:.1f}mm axis={np.degrees(axis_err):.1f}deg "
                f"streak={self._insert_align_streak}",
                flush=True,
            )

        if lat_err > cfg.pos_tol_m:
            delta_pos = lateral_align_delta(
                insert, socket, hole_axis, gain=cfg.align_pos_gain
            )
            self._apply_tip_delta_to_wrist(delta_pos)
        if axis_err > cfg.angle_tol_rad:
            self._apply_peg_axis_align_to_wrist(peg_axis, hole_axis)

    def _insert_step(self, raw_env) -> None:
        cfg = self.config
        self._insert_steps += 1
        if self._insert_steps > cfg.max_insert_steps:
            return
        if self._should_release_grasp(raw_env):
            return

        insert = self._peg_insert_pos(raw_env)
        socket = self._socket_pos(raw_env)
        hole_axis = self._hole_axis(raw_env)
        peg_axis = self._peg_axis(raw_env)
        lat_err, _ = lateral_error(insert, socket, hole_axis)
        axis_err = axis_parallel_error_rad(peg_axis, hole_axis)

        if lat_err > cfg.pos_tol_m:
            delta_pos = lateral_align_delta(
                insert, socket, hole_axis, gain=cfg.align_pos_gain
            )
            self._apply_tip_delta_to_wrist(delta_pos)
        if axis_err > cfg.angle_tol_rad:
            self._apply_peg_axis_align_to_wrist(peg_axis, hole_axis)

        along_delta = insert_along_hole_delta(
            hole_axis, step_m=cfg.insert_along_step_m
        )
        self._apply_tip_delta_to_wrist(
            cfg.insert_wrist_tip_scale * along_delta
        )

    def _release_step(self) -> None:
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
