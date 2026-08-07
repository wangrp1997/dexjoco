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

    # Dual-arm relative PBVS: both arms reduce tip↔socket feature error e.
    # Classic coop: v_R - v_L ≈ -λ e (relative), not absolute world tray upright.
    freeze_left_arm_at_handoff: bool = False
    # Fraction of lateral / axis error corrected by moving the hole (left/tray).
    left_share_xy: float = 0.45
    left_share_rot: float = 0.40
    left_wrist_tip_scale: float = 0.9
    max_left_wrist_step_m: float = 0.0035

    # ALIGN / INSERT: privileged tip PBVS.
    handoff_settle_frames: int = 30
    wrist_tip_scale: float = 0.95
    align_pos_gain: float = 0.55
    align_rot_gain: float = 0.20
    max_wrist_step_m: float = 0.004
    max_wrist_rot_step_rad: float = 0.012
    max_wrist_rot_from_anchor_rad: float = 0.55
    approach_max_step_m: float = 0.006
    insert_wrist_tip_scale: float = 0.9
    insert_along_step_m: float = 0.003
    pos_tol_m: float = 0.004
    angle_tol_rad: float = 0.14
    axis_align_max_lat_m: float = 0.008
    # Require stable lat+axis alignment before INSERT.
    insert_align_confirm_frames: int = 10
    max_align_steps: int = 1200
    align_debug_interval: int = 120
    max_insert_steps: int = 600
    insert_debug_interval: int = 30

    # PBVS gains: tip feature error -> tip twist (λ), then map to wrist.
    pbvs_lambda_xy: float = 0.7
    pbvs_lambda_z: float = 0.45
    pbvs_lambda_rot: float = 0.22
    pbvs_standoff_m: float = 0.055
    pbvs_insert_target_along_m: float = 0.02  # don't aim past rim into -5mm (causes ram)
    pbvs_stall_frames: int = 35
    pbvs_retreat_frames: int = 20
    pbvs_retreat_step_m: float = 0.0025
    # If tip does not get deeper, stop Z push immediately (no hard ram).
    tip_jam_frames: int = 10
    tip_jam_improve_m: float = 0.0004
    # Cap along step once near hole (mm-scale gentle slide).
    max_insert_z_step_m: float = 0.0015
    # Soft seat: RELEASE only when tip ~成功集 depth (~35mm), not 55mm fake seat.
    stop_lateral_tip_m: float = 0.050
    soft_seat_tip_m: float = 0.038
    seated_along_m: float = 0.038
    # Tip distance treated as "entering" the hole (for Z cap / lateral gate).
    pbvs_enter_tip_m: float = 0.100
    # Spiral search only far from hole (near-rim spiral blew axis on ep91).
    pbvs_spiral_min_tip_m: float = 0.120
    # Relative axis twist only above this tip depth (deep twist pries peg out).
    pbvs_rel_axis_min_tip_m: float = 0.060
    # Absolute tray upright (hole→world +Z). Off by default — conflicts with relative PBVS.
    tray_z_up_enable: bool = False
    tray_z_up_enable_tip_m: float = 0.100
    tray_z_up_disable_tip_m: float = 0.045
    tray_z_up_tol_rad: float = 0.10
    tray_z_up_gain: float = 0.28
    max_left_wrist_rot_step_rad: float = 0.015

    # Grasp detection for handoff / abort.
    lift_threshold_m: float = 0.05
    peg_lost_abort_frames: int = 15
    seated_lat_m: float = 0.010

    # Release when tip near socket; then retract + open (no more push).
    release_insert_socket_dist_m: float = 0.038
    release_confirm_frames: int = 2
    release_steps: int = 50
    release_retract_m: float = 0.010

    handoff_debug_interval: int = 60
