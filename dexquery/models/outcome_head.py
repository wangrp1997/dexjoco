"""Per-object outcome prediction heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class OutcomeHead(nn.Module):
    """Map subtask embeddings to scalar grasp outcomes."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.tray_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.peg_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(
        self,
        z_tray: torch.Tensor,
        z_peg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tray/peg logits with shape ``(B,)``."""
        tray_logit = self.tray_head(z_tray).squeeze(-1)
        peg_logit = self.peg_head(z_peg).squeeze(-1)
        return tray_logit, peg_logit
