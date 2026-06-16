"""Batch collation and optional normalization for DexQuery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.utils.constants import ACTION, OBS_STATE


@dataclass
class DexQueryCollator:
    """Stack DexQuery samples into model-ready batches."""

    normalize_state: bool = False
    normalize_action: bool = False
    state_mean: torch.Tensor | None = None
    state_std: torch.Tensor | None = None
    action_mean: torch.Tensor | None = None
    action_std: torch.Tensor | None = None

    @classmethod
    def from_dataset_stats(
        cls,
        stats: dict[str, Any],
        *,
        normalize_state: bool = True,
        normalize_action: bool = True,
    ) -> DexQueryCollator:
        return cls(
            normalize_state=normalize_state,
            normalize_action=normalize_action,
            state_mean=_stats_vector(stats, OBS_STATE, "mean"),
            state_std=_stats_vector(stats, OBS_STATE, "std"),
            action_mean=_stats_vector(stats, ACTION, "mean"),
            action_std=_stats_vector(stats, ACTION, "std"),
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("DexQueryCollator received an empty batch.")

        images = torch.stack([item["images"] for item in batch], dim=0)
        state = torch.stack([item["state"] for item in batch], dim=0)
        actions = torch.stack([item["actions"] for item in batch], dim=0)
        tray_ok = torch.stack([item["tray_ok"] for item in batch], dim=0)
        peg_ok = torch.stack([item["peg_ok"] for item in batch], dim=0)
        subtask_phase = torch.stack([item["subtask_phase"] for item in batch], dim=0)

        if self.normalize_state:
            state = _normalize(state, self.state_mean, self.state_std)
        if self.normalize_action:
            actions = _normalize(actions, self.action_mean, self.action_std)

        return {
            "images": images,
            "state": state,
            "actions": actions,
            "tray_ok": tray_ok,
            "peg_ok": peg_ok,
            "subtask_phase": subtask_phase,
            "subtask_prompts": batch[0]["subtask_prompts"],
            "index": torch.stack([item["index"] for item in batch], dim=0),
            "episode_index": torch.stack([item["episode_index"] for item in batch], dim=0),
            "frame_index": torch.stack([item["frame_index"] for item in batch], dim=0),
        }


def _stats_vector(stats: dict[str, Any], key: str, field: str) -> torch.Tensor:
    if key not in stats or field not in stats[key]:
        raise KeyError(f"Missing stats[{key!r}][{field!r}]")
    values = stats[key][field]
    tensor = torch.as_tensor(values, dtype=torch.float32)
    return tensor.reshape(-1)


def _normalize(
    tensor: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if mean is None or std is None:
        raise ValueError("Normalization requested but mean/std were not provided.")
    mean = mean.to(device=tensor.device, dtype=tensor.dtype)
    std = std.to(device=tensor.device, dtype=tensor.dtype)
    while mean.ndim < tensor.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return (tensor - mean) / (std + eps)
