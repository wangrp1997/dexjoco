"""Default paths for generated data."""

from __future__ import annotations

from pathlib import Path

TASK_ID = "bimanual_assembly"
DEFAULT_GENDATA_ROOT = Path("/mnt/ssd/datasets/dexjoco_gendata")
DEFAULT_MANIFEST = Path("/mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly/manifest.json")


def task_dir(task: str = TASK_ID, root: Path = DEFAULT_GENDATA_ROOT) -> Path:
    return Path(root) / task


def video_dir(task: str = TASK_ID, root: Path = DEFAULT_GENDATA_ROOT) -> Path:
    return task_dir(task, root) / "videos"
