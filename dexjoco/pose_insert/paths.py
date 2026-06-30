"""Default artifact paths for PoseInsert sim datasets."""

from __future__ import annotations

from pathlib import Path

DEFAULT_POSEINSERT_DATA_ROOT = Path("/mnt/hdd/dexjoco/poseinsert_sim")
DEFAULT_POSEINSERT_OUTPUT_ROOT = Path("/mnt/hdd/dexjoco/outputs/poseinsert_sim")


def default_poseinsert_data_dir(task_id: str = "bimanual_assembly") -> Path:
    return DEFAULT_POSEINSERT_DATA_ROOT / task_id


def default_eval_video_path(episode_index: int, seed: int = 0) -> Path:
    return DEFAULT_POSEINSERT_OUTPUT_ROOT / "videos" / f"ep{int(episode_index)}_seed{int(seed)}.mp4"
