"""Micro-demo pilot v0 constants (code-enforced; YAML cannot loosen)."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Write path is intentionally disabled in this revision.
WRITE_IMPLEMENTATION_ENABLED = False

PILOT_TAG = "micro_demo_pilot_v0"
PILOT_DIR_NAME = "pilot_micro_demo_v0"

# Sole allowlisted output root (resolved). Future writes must land only here.
ALLOWED_OUT_ROOT = (PROJECT_ROOT / "data" / PILOT_DIR_NAME).resolve()

# Hard caps for v0 — not overridable by YAML to a looser value.
MAX_FAMILIES = 1
MAX_EPISODES_PER_FAMILY = 2
MAX_TRAJECTORIES_PER_EPISODE = 1
MAX_TOTAL_TRAJECTORIES = 1
MAX_HORIZON_STEPS = 80
MIN_HORIZON_STEPS = 5  # hold+lift+transport(+neg) planner needs >= 5 env steps


# Future states.npz limits (documented; write path not enabled yet).
MAX_STATES_NPZ_BYTES = 32 * 1024 * 1024
FORBIDDEN_NPZ_OBJECT_DTYPE = True
