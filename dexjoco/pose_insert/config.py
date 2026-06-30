"""Runtime config for PoseInsert sim adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PoseInsertAdapterConfig:
    """Right-arm PoseInsert rollout after hybrid-style handoff."""

    grasp_lock_frames: int = 5
    handoff_confirm_frames: int = 8
    lift_ready_m: float = 0.06
    approach_xy_m: float = 0.06
    approach_z_min_m: float = 0.01
    approach_z_max_m: float | None = None
    freeze_left_arm_at_handoff: bool = True
    left_insert_coop_gain: float = 0.35
    handoff_settle_frames: int = 30
    handoff_debug_interval: int = 60

    num_action: int = 20
    replan_interval: int = 1
    normalize_translation: bool = True
    max_wrist_step_m: float = 0.004
    max_wrist_rot_step_rad: float = 0.015
    action_blend: float = 0.45

    lift_threshold_m: float = 0.05
    peg_lost_abort_frames: int = 15
    max_poseinsert_steps: int = 800

    release_insert_socket_dist_m: float = 0.04
    release_confirm_frames: int = 3
    release_steps: int = 60
