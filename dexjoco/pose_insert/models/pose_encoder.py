"""Disentangled pose encoder (PoseInsert PoseDP, no Utils dependency)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DisentangledPoseEncoder(nn.Module):
    def __init__(self, input_dim_trans: int = 3, input_dim_rot: int = 6, output_dim: int = 64):
        super().__init__()
        self.encoder_trans = nn.Sequential(
            nn.Linear(input_dim_trans, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )
        self.encoder_rot = nn.Sequential(
            nn.Linear(input_dim_rot, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )
        self.pose_head = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=128, batch_first=True),
            nn.Linear(128, 128),
        )

    def forward(self, trans: torch.Tensor, rots: torch.Tensor) -> torch.Tensor:
        trans_feature = self.encoder_trans(trans)
        rots_feature = self.encoder_rot(rots)
        fused = torch.cat([trans_feature, rots_feature], dim=1)
        return self.pose_head(fused)
