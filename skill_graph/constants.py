"""Task body names (bimanual_assembly only)."""

from __future__ import annotations

from typing import Literal

TRAY_BODY = "industreal_tray_insert_round_peg_8mm"
PEG_BODY = "industreal_round_peg_8mm"
LEFT_HAND_ROOT = "allegro_palm_left"
RIGHT_HAND_ROOT = "allegro_palm_right"

Side = Literal["left", "right"]
ObjectName = Literal["tray", "peg"]

MIN_GRASP_CONTACTS = 3
CONTACT_DIST_M = 0.002
APPROACH_LOOKBACK = 80
MAX_APPROACH_WAYPOINTS = 80

# pi0.5 bimanual_assembly eval parity (30 Hz, ~50 s policy steps)
EVAL_MAX_ENV_STEPS = 1500
EVAL_MAX_VIDEO_FRAMES = 1500

# Peg is a cylinder: try grasp poses rotated around object Z at selection time.
PEG_YAW_CANDIDATES_RAD = (0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469)
