"""Artifact paths for dex_track_assembly."""

from __future__ import annotations

from pathlib import Path

DEFAULT_ARTIFACTS_ROOT = Path("/mnt/hdd/dexjoco/dex_track_assembly")
DEFAULT_OUTPUT_ROOT = Path("/mnt/hdd/dexjoco/outputs/dex_track_assembly")
DEFAULT_MANIFEST = Path("/mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly/manifest.json")


def mocap_dir(task: str = "bimanual_assembly", robot: str = "PandaBimanual") -> Path:
    return DEFAULT_ARTIFACTS_ROOT / task / "mocap" / robot


def checkpoint_dir(exp_name: str) -> Path:
    return DEFAULT_ARTIFACTS_ROOT / "checkpoints" / exp_name


def train_log_dir(exp_name: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / "logs" / "track" / exp_name


def wandb_dir() -> Path:
    return DEFAULT_OUTPUT_ROOT / "wandb"
