"""DexQuery training dataset: LeRobot observations + sidecar outcome labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION, OBS_STATE

from .outcome_labels import (
    build_outcome_index_map,
    default_outcome_label_dir,
    load_outcome_labels,
)
from .subtask_prompts import SubtaskPrompts, infer_subtask_phase

DUAL_ARM_CAMERA_KEYS: tuple[str, ...] = (
    "observation.images.ego",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)


@dataclass(frozen=True)
class DexQueryDatasetConfig:
    task: str
    dataset_root: Path
    action_horizon: int = 30
    camera_keys: tuple[str, ...] = DUAL_ARM_CAMERA_KEYS
    outcome_fields: tuple[str, ...] = ("tray_ok", "peg_ok")
    label_dir: Path | None = None
    video_backend: str = "pyav"
    episodes: list[int] | None = None


class DexQueryDataset(Dataset):
    """Wrap ``LeRobotDataset`` and attach DexQuery outcome labels by global index."""

    def __init__(
        self,
        config: DexQueryDatasetConfig,
        *,
        subtask_prompts: SubtaskPrompts | None = None,
    ) -> None:
        self.config = config
        self.dataset_root = config.dataset_root.expanduser()
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_root}")

        self.subtask_prompts = subtask_prompts or SubtaskPrompts.for_task(config.task)
        self.camera_keys = tuple(config.camera_keys)
        self.outcome_fields = tuple(config.outcome_fields)

        self.meta = LeRobotDatasetMetadata(
            repo_id=config.task,
            root=self.dataset_root,
        )
        delta_timestamps = {
            ACTION: [i / self.meta.fps for i in range(config.action_horizon)],
        }
        self._dataset = LeRobotDataset(
            repo_id=config.task,
            root=self.dataset_root,
            episodes=config.episodes,
            delta_timestamps=delta_timestamps,
            video_backend=config.video_backend,
        )

        label_dir = config.label_dir or default_outcome_label_dir(self.dataset_root)
        self._outcomes = load_outcome_labels(label_dir)
        self._outcome_index_map = build_outcome_index_map(self._outcomes)
        self._validate_outcome_fields()

    def _validate_outcome_fields(self) -> None:
        missing = [field for field in self.outcome_fields if field not in self._outcomes]
        if missing:
            raise KeyError(f"Missing outcome columns in sidecar parquet: {missing}")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list[str] | int]:
        sample = self._dataset[idx]
        global_index = _scalar_int(sample["index"])
        label_row = self._outcome_index_map.get(global_index)
        if label_row is None:
            raise KeyError(
                f"No DexQuery outcome label for LeRobot index={global_index}. "
                "Re-run label_contact.py or filter unlabeled frames."
            )

        images = torch.stack([_to_image_tensor(sample[key]) for key in self.camera_keys], dim=0)
        state = _to_vector_tensor(sample[OBS_STATE]).float()
        actions = _to_matrix_tensor(sample[ACTION]).float()

        outcomes = {
            field: torch.tensor(float(self._outcomes[field][label_row]), dtype=torch.float32)
            for field in self.outcome_fields
        }
        tray_ok = outcomes.get("tray_ok", torch.tensor(0.0))
        peg_ok = outcomes.get("peg_ok", torch.tensor(0.0))
        subtask_phase = infer_subtask_phase(float(tray_ok), float(peg_ok))

        return {
            "images": images,
            "state": state,
            "actions": actions,
            "tray_ok": tray_ok,
            "peg_ok": peg_ok,
            "subtask_phase": torch.tensor(subtask_phase, dtype=torch.int64),
            "subtask_prompts": self.subtask_prompts.as_list(),
            "index": torch.tensor(global_index, dtype=torch.int64),
            "episode_index": _scalar_int_tensor(sample.get("episode_index")),
            "frame_index": _scalar_int_tensor(sample.get("frame_index")),
        }


def _scalar_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


def _scalar_int_tensor(value) -> torch.Tensor:
    if value is None:
        return torch.tensor(-1, dtype=torch.int64)
    if isinstance(value, torch.Tensor):
        return value.reshape(()).to(dtype=torch.int64)
    return torch.tensor(int(value), dtype=torch.int64)


def _to_image_tensor(value) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()
    if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1)
    return tensor


def _to_vector_tensor(value) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.reshape(-1)


def _to_matrix_tensor(value) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def resolve_dataset_root(root: Path | str, task: str) -> Path:
    root = Path(root).expanduser()
    if (root / "meta" / "info.json").exists():
        return root
    candidate = root / task
    if (candidate / "meta" / "info.json").exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve LeRobot dataset root for task {task!r} under {root}")
