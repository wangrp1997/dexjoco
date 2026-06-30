"""PoseDP wrapper using PoseInsert diffusion decoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from pose_insert.models.pose_encoder import DisentangledPoseEncoder
from pose_insert.poseinsert_path import ensure_poseinsert_on_path


class PoseDP(nn.Module):
    """Pose-only diffusion policy (upstream ``PoseInsert/policy/policy.py::PoseDP``)."""

    def __init__(
        self,
        num_action: int = 20,
        obs_feature_dim: int = 128,
        action_dim: int = 9,
        hidden_dim: int = 512,
        prediction: str = "sample",
    ) -> None:
        super().__init__()
        ensure_poseinsert_on_path()
        from policy.diffusion import DiffusionUNetPolicy

        self.pose_encoder = DisentangledPoseEncoder()
        self.action_decoder = DiffusionUNetPolicy(
            action_dim,
            horizon=num_action,
            n_obs_steps=1,
            obs_feature_dim=obs_feature_dim,
            prediction=prediction,
        )
        self.readout_embed = nn.Embedding(1, hidden_dim)

    def forward(self, pose: torch.Tensor, actions: torch.Tensor | None = None, batch_size: int = 24):
        trans = pose[:, 0, :3, 2]
        rots = pose[:, 0, :3, :2]
        rots = rots.reshape(rots.shape[0], rots.shape[1] * rots.shape[2])
        pose_feature = self.pose_encoder(trans, rots)
        if actions is not None:
            return self.action_decoder.compute_loss(pose_feature, actions)
        with torch.no_grad():
            return self.action_decoder.predict_action(pose_feature)
