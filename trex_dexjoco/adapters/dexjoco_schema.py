"""DexJoCo embodiment constants (must match utils/lerobot_common.py)."""

from utils.lerobot_common import (
    ACTION_CHUNK,
    ACTION_DIM,
    F6_DIM,
    F6_PER_FINGER,
    KEY_EGO_SRC,
    KEY_WRIST_L,
    KEY_WRIST_R,
    N_FINGERS,
    N_FINGERS_PER_HAND,
    STATS_KEY,
)

# Source LeRobot keys
SRC_ACTION = "action"                 # [44]
SRC_STATE = "observation.state"       # [46] quat proprio
SRC_EGO = KEY_EGO_SRC
SRC_WRIST_L = KEY_WRIST_L
SRC_WRIST_R = KEY_WRIST_R

DEFAULT_LEROBOT_ROOT = (
    "/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly"
)
DEFAULT_FORCE_PARQUET = (
    "/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly/"
    "force_labels/forces.parquet"
)
DEFAULT_INSTRUCTION = "Assemble the peg into the tray."

# Finger order in force labels: ff, mf, rf, th (4 × xyz)
FINGER_NAMES = ("ff", "mf", "rf", "th")
