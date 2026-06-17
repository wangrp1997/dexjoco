"""Configuration for privileged geometry-based insert control."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HybridInsertConfig:
    """Right-arm-only insert controller for DexJoCo bimanual assembly."""

    # Lock right fingers once peg is grasped (before hybrid handoff).
    grasp_lock_frames: int = 5

    # Handoff: peg in approach cylinder above socket (plan C).
    handoff_confirm_frames: int = 8
    lift_ready_m: float = 0.06
    approach_xy_m: float = 0.06
    approach_z_min_m: float = 0.01
    approach_z_max_m: float | None = None

    # Freeze left arm at handoff so policy tray lift does not push hole into the peg.
    freeze_left_arm_at_handoff: bool = True

    # ALIGN / INSERT (right wrist only): small steps to avoid opspace violent tracking.
    handoff_settle_frames: int = 30
    wrist_tip_scale: float = 0.2
    align_pos_gain: float = 0.22
    align_rot_gain: float = 0.25
    max_wrist_step_m: float = 0.003
    max_wrist_rot_step_rad: float = 0.012
    max_wrist_rot_from_anchor_rad: float = 0.55
    approach_max_step_m: float = 0.006
    insert_wrist_tip_scale: float = 0.35
    insert_along_step_m: float = 0.004
    pos_tol_m: float = 0.005
    angle_tol_rad: float = 0.12
    axis_align_max_lat_m: float = 0.03
    # Require stable lat+axis alignment before INSERT.
    insert_align_confirm_frames: int = 15
    max_align_steps: int = 1200
    align_debug_interval: int = 120
    max_insert_steps: int = 600
    insert_debug_interval: int = 30

    # Grasp detection for handoff / abort.
    lift_threshold_m: float = 0.05
    peg_lost_abort_frames: int = 15

    # Release when insert end (cylinder bottom) is close to socket site.
    release_insert_socket_dist_m: float = 0.04
    release_confirm_frames: int = 3
    release_steps: int = 60

    handoff_debug_interval: int = 60
