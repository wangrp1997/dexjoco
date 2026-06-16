"""DiT action head interface."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import ActionHeadOutput, BaseActionHead

# Multi-Task DiT baseline in DexJoCo: configs/training/policies/multi_task_dit.yaml
# DiT block reference: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/vla_jepa/action_head.py


class DitActionHead(BaseActionHead):
    """Conditional DiT action head (DexJoCo multi_task_dit-compatible slot)."""

    action_type = "dit"

    def __init__(
        self,
        *,
        embed_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        **_kwargs,
    ) -> None:
        super().__init__()
        self._chunk_size = chunk_size
        self._action_dim = action_dim
        self.cond_proj = nn.Linear(embed_dim + state_dim, embed_dim)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def forward(
        self,
        z_subtask: torch.Tensor,
        state: torch.Tensor,
        *,
        actions: torch.Tensor | None = None,
    ) -> ActionHeadOutput:
        raise NotImplementedError(
            "DitActionHead: port LeRobot DiT action head, "
            "conditioned on cond_proj([z_subtask, state])."
        )
