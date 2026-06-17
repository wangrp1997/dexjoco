"""LeRobot dataset adapters and contact-outcome labels."""

from __future__ import annotations

from .outcome_labels import build_outcome_index_map, default_outcome_label_dir, load_outcome_labels
from .subtask_prompts import SubtaskPrompts, infer_subtask_phase

__all__ = [
    "DUAL_ARM_CAMERA_KEYS",
    "DexQueryCollator",
    "DexQueryDataset",
    "DexQueryDatasetConfig",
    "SubtaskPrompts",
    "build_outcome_index_map",
    "default_outcome_label_dir",
    "infer_subtask_phase",
    "load_outcome_labels",
    "resolve_dataset_root",
]


def __getattr__(name: str):
    if name == "DexQueryCollator":
        from .collator import DexQueryCollator

        return DexQueryCollator
    if name in {
        "DUAL_ARM_CAMERA_KEYS",
        "DexQueryDataset",
        "DexQueryDatasetConfig",
        "resolve_dataset_root",
    }:
        from .dataset import (
            DUAL_ARM_CAMERA_KEYS,
            DexQueryDataset,
            DexQueryDatasetConfig,
            resolve_dataset_root,
        )

        return {
            "DUAL_ARM_CAMERA_KEYS": DUAL_ARM_CAMERA_KEYS,
            "DexQueryDataset": DexQueryDataset,
            "DexQueryDatasetConfig": DexQueryDatasetConfig,
            "resolve_dataset_root": resolve_dataset_root,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
