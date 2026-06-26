"""Task-specific names and paths for bimanual_assembly interaction retargeting."""

from __future__ import annotations

from pathlib import Path

_XML_DIR = Path(__file__).resolve().parents[1] / "dexjoco" / "dexjoco" / "sim" / "envs" / "xmls"
_MESH_DIR = _XML_DIR / "industreal" / "mesh" / "industreal_pegs"

TRAY_BODY = "industreal_tray_insert_round_peg_8mm"
PEG_BODY = "industreal_round_peg_8mm"
LEFT_PALM = "allegro_palm_left"
RIGHT_PALM = "allegro_palm_right"

LEFT_HAND_ROOT = LEFT_PALM
RIGHT_HAND_ROOT = RIGHT_PALM

# Interaction mesh hand vertices (TopoRetarget N_h=21): palm + 4 fingers × 5 link bodies.
# Order: wrist/palm, thumb chain, index(ff), middle(mf), ring(rf). Allegro has no pinky;
# each finger uses base→proximal→medial→distal→tip instead of MediaPipe's 4 joints/finger.
_FINGER_CHAIN = ("base", "proximal", "medial", "distal", "tip")
_FINGERS = (
    ("th", "thumb"),
    ("ff", "index"),
    ("mf", "middle"),
    ("rf", "ring"),
)


def _allegro_hand_bodies(*, side: str) -> tuple[str, ...]:
    suffix = f"_{side}"
    names: list[str] = [f"allegro_palm{suffix}"]
    for prefix, _ in _FINGERS:
        for link in _FINGER_CHAIN:
            body = f"{prefix}_{link}{suffix}" if link != "tip" else f"{prefix}_tip{suffix}"
            names.append(body)
    return tuple(names)


LEFT_HAND_BODIES = _allegro_hand_bodies(side="left")
RIGHT_HAND_BODIES = _allegro_hand_bodies(side="right")

# Fingertips only — used by spider-style contact detection, not interaction mesh.
LEFT_TIP_BODIES = ("th_tip_left", "ff_tip_left", "mf_tip_left", "rf_tip_left")
RIGHT_TIP_BODIES = ("th_tip_right", "ff_tip_right", "mf_tip_right", "rf_tip_right")

NUM_HAND_KEYPOINTS = 21
NUM_OBJECT_SAMPLES = 50  # TopoRetarget appendix N_o=50
NUM_INTERACTION_VERTICES = NUM_HAND_KEYPOINTS + NUM_OBJECT_SAMPLES

# Palm(0) + 4 fingers × 5 links; chain edges within each finger only.
HAND_SKELETON_EDGES: tuple[tuple[int, int], ...] = tuple(
    (i, i + 1) for base in (1, 6, 11, 16) for i in range(base, base + 4)
)

TRAY_MESH_PATH = _MESH_DIR / "industreal_tray_insert_round_peg_8mm.obj"
PEG_MESH_PATH = _MESH_DIR / "industreal_round_peg_8mm.obj"
# Must match scale in industreal_*_*.xml mesh assets.
INDUSTREAL_MESH_SCALE = 4.5

CONTACT_SAMPLE_SIGMA_M = 0.025

# Grasp-stable detection (@ ~30 Hz).
CONTACT_WINDOW = 10
GRIPPER_VEL_EPS = 0.02
LIFT_HEIGHT_M = 0.02  # above rest z at replay start (spider uses 0.05 for success label)
ON_TABLE_MARGIN_M = 0.015  # object still on table when grasp snapshot is taken
MIN_GRASP_CONTACT_COUNT = 3  # ignore brief brush contacts (tray often reaches 6-8)

TASK_ID = "bimanual_assembly"

# Default sidecar export root: /.../interaction_sidecar/<task_id>/
DEFAULT_ARTIFACTS_ROOT = Path("/mnt/hdd/dexjoco/interaction_sidecar")


def default_sidecar_dir(task_id: str = TASK_ID) -> Path:
    return DEFAULT_ARTIFACTS_ROOT / task_id
