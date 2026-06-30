"""Default artifact paths for skill_graph."""

from __future__ import annotations

from pathlib import Path

TASK_ID = "bimanual_assembly"
DEFAULT_ARTIFACTS_ROOT = Path("/mnt/hdd/dexjoco/skill_graph")
DEFAULT_SIDECAR_ROOT = Path("/mnt/hdd/dexjoco/interaction_sidecar")
DEFAULT_VIDEO_ROOT = Path("/mnt/hdd/dexjoco/outputs/skill_graph/videos")


def template_bank_dir(task_id: str = TASK_ID) -> Path:
    return DEFAULT_ARTIFACTS_ROOT / task_id / "grasp_templates"


def sidecar_manifest(task_id: str = TASK_ID) -> Path:
    return DEFAULT_SIDECAR_ROOT / task_id / "manifest.json"


def default_video_path(tag: str, seed: int = 0) -> Path:
    return DEFAULT_VIDEO_ROOT / f"{tag}_seed{seed}.mp4"
