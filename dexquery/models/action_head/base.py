"""Pluggable action-head interface for DexQuery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ActionHeadOutput:
    """Unified return type for all action-head backends."""

    pred_actions: torch.Tensor
    loss: torch.Tensor | None = None


class BaseActionHead(nn.Module, ABC):
    """Predict a future action chunk from subtask conditioning + proprio.

    Implementations may use direct regression, diffusion, flow matching, etc.
    DexQuery vision/query/outcome stacks stay fixed; only this module swaps.
    """

    action_type: str

    @abstractmethod
    def forward(
        self,
        z_subtask: torch.Tensor,
        state: torch.Tensor,
        *,
        actions: torch.Tensor | None = None,
    ) -> ActionHeadOutput:
        """Return predicted actions and optional training loss.

        Args:
            z_subtask: ``(B, embed_dim)`` subtask embedding.
            state: ``(B, state_dim)`` proprioception.
            actions: optional ``(B, chunk_size, action_dim)`` targets for training.
        """

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        ...

    @torch.no_grad()
    def predict(self, z_subtask: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Inference helper returning ``(B, chunk_size, action_dim)``."""
        self.eval()
        return self.forward(z_subtask, state, actions=None).pred_actions
