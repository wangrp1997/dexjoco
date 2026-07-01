"""Constants for Panda bimanual assembly tracking (MJX)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from track_mj.paths import mocap_dir

# DexJoCo scene XML (patched for MJX in base_env).
DEXJOJO_XMLS = Path("/home/wangrenpeng/dexjoco/dexjoco/dexjoco/sim/envs/xmls")
FLAT_TERRAIN_XML = DEXJOJO_XMLS / "arena_arm_hand_bimanual_assembly.xml"

NUM_CTRL_JOINTS = 46
CTRL_QPOS_SLICE = slice(0, NUM_CTRL_JOINTS)
CTRL_QVEL_SLICE = slice(0, NUM_CTRL_JOINTS)

TASK_ID = "bimanual_assembly"
ROBOT_SUBDIR = "PandaBimanual"
MOCAP_ROOT = mocap_dir(TASK_ID, ROBOT_SUBDIR)

PEG_BODY = "industreal_round_peg_8mm"
TRAY_BODY = "industreal_tray_insert_round_peg_8mm"
TABLE_BODY = "table"

TRACK_BODY_NAMES = (
    PEG_BODY,
    TRAY_BODY,
    "allegro_palm_right",
    "allegro_palm_left",
    "link7_right",
    "link7_left",
)

# Actuator order matches qpos[0:46]: 7 motor + 16 position per arm.
MOTOR_ACTUATOR_MASK = np.array(
    [True] * 7 + [False] * 16 + [True] * 7 + [False] * 16,
    dtype=bool,
)

ACTION_JOINT_NAMES = [
    "joint1_right",
    "joint2_right",
    "joint3_right",
    "joint4_right",
    "joint5_right",
    "joint6_right",
    "joint7_right",
    "ffj0_right",
    "ffj1_right",
    "ffj2_right",
    "ffj3_right",
    "mfj0_right",
    "mfj1_right",
    "mfj2_right",
    "mfj3_right",
    "rfj0_right",
    "rfj1_right",
    "rfj2_right",
    "rfj3_right",
    "thj0_right",
    "thj1_right",
    "thj2_right",
    "thj3_right",
    "joint1_left",
    "joint2_left",
    "joint3_left",
    "joint4_left",
    "joint5_left",
    "joint6_left",
    "joint7_left",
    "rfj0_left",
    "rfj1_left",
    "rfj2_left",
    "rfj3_left",
    "mfj0_left",
    "mfj1_left",
    "mfj2_left",
    "mfj3_left",
    "ffj0_left",
    "ffj1_left",
    "ffj2_left",
    "ffj3_left",
    "thj0_left",
    "thj1_left",
    "thj2_left",
    "thj3_left",
]

# Panda motor PD + Allegro position (ctrl = qpos target).
KPs = np.array(
    [80.0] * 7 + [1.0] * 16 + [80.0] * 7 + [1.0] * 16,
    dtype=np.float64,
)
KDs = np.array(
    [4.0] * 7 + [0.1] * 16 + [4.0] * 7 + [0.1] * 16,
    dtype=np.float64,
)
TORQUE_LIMIT = np.array(
    [87, 87, 87, 87, 12, 12, 12]
    + [5.0] * 16
    + [87, 87, 87, 87, 12, 12, 12]
    + [5.0] * 16,
    dtype=np.float64,
)

DEFAULT_QPOS = np.zeros(NUM_CTRL_JOINTS, dtype=np.float64)
_panda_home = (0.0, -0.785, 0.0, -2.35, 0.0, 1.57, np.pi / 4)
DEFAULT_QPOS[0:7] = _panda_home
DEFAULT_QPOS[23:30] = _panda_home
DEFAULT_QPOS[22] = 0.263  # thj3_right home from dexjoco
DEFAULT_QPOS[45] = 0.263  # thj3_left


def task_to_xml(task_name: str) -> Path:
    if task_name != "flat_terrain":
        raise ValueError(f"Unsupported task_name {task_name}.")
    return FLAT_TERRAIN_XML


def default_trajectory_names(num_episodes: int = 100, segment: str = "full") -> list[str]:
    return [f"ep{i:03d}_{segment}" for i in range(num_episodes)]
