"""Load bimanual PoseDP checkpoint (dual wrist12 actions)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pose_insert.dataset_sim import load_or_build_workspace
from pose_insert.models.pose_dp import PoseDP
from pose_insert.wrist_actions import DUAL_WRIST_DIM, denormalize_dual_wrist, load_or_build_wrist_workspace


class BimanualPoseInsertRunner:
    """PoseDP with obs pose9 and action dual wrist12."""

    def __init__(
        self,
        ckpt_path: Path | str,
        *,
        data_root: Path | str | None = None,
        num_action: int = 20,
        device: str | torch.device | None = None,
        normalize: bool = True,
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.num_action = int(num_action)
        self.normalize = bool(normalize)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.pose_workspace: np.ndarray | None = None
        self.wrist_workspace: np.ndarray | None = None
        if self.normalize:
            if data_root is None:
                raise ValueError("data_root required when normalize=True")
            root = Path(data_root)
            self.pose_workspace = load_or_build_workspace(root)
            self.wrist_workspace = load_or_build_wrist_workspace(root)

        self.policy = PoseDP(num_action=self.num_action, obs_feature_dim=128, action_dim=DUAL_WRIST_DIM)
        state = torch.load(self.ckpt_path, map_location=self.device)
        self.policy.load_state_dict(state, strict=False)
        self.policy.to(self.device)
        self.policy.eval()

    @property
    def workspace(self) -> np.ndarray | None:
        return self.pose_workspace

    def predict_wrist12_horizon(self, obs_pose9: np.ndarray) -> np.ndarray:
        """Predict (horizon, 12) dual wrist targets; denormalized if normalize=True."""
        obs = torch.from_numpy(np.asarray(obs_pose9, dtype=np.float32)).reshape(1, 1, 3, 3)
        obs = obs.to(self.device)
        with torch.inference_mode():
            actions = self.policy(obs, actions=None, batch_size=1)
        out = actions.squeeze(0).detach().cpu().numpy()
        if self.normalize and self.wrist_workspace is not None:
            out = denormalize_dual_wrist(self.wrist_workspace, out)
        return out


class BimanualAction44Runner:
    """PoseDP with obs pose9 and action dual-arm44 (wrist rotvec + hands)."""

    def __init__(
        self,
        ckpt_path: Path | str,
        *,
        data_root: Path | str | None = None,
        num_action: int = 20,
        device: str | torch.device | None = None,
        normalize: bool = True,
    ) -> None:
        from pose_insert.wrist_actions import (
            DUAL_ACTION44_DIM,
            denormalize_action44,
            load_or_build_action44_workspace,
        )

        self.ckpt_path = Path(ckpt_path)
        self.num_action = int(num_action)
        self.normalize = bool(normalize)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.pose_workspace: np.ndarray | None = None
        self.action44_workspace: np.ndarray | None = None
        if self.normalize:
            if data_root is None:
                raise ValueError("data_root required when normalize=True")
            root = Path(data_root)
            self.pose_workspace = load_or_build_workspace(root)
            self.action44_workspace = load_or_build_action44_workspace(root)

        self.policy = PoseDP(num_action=self.num_action, obs_feature_dim=128, action_dim=DUAL_ACTION44_DIM)
        state = torch.load(self.ckpt_path, map_location=self.device)
        self.policy.load_state_dict(state, strict=False)
        self.policy.to(self.device)
        self.policy.eval()

    @property
    def workspace(self) -> np.ndarray | None:
        return self.pose_workspace

    def predict_action44_horizon(self, obs_pose9: np.ndarray) -> np.ndarray:
        """Predict (horizon, 44) dual-arm targets; denormalized if normalize=True."""
        from pose_insert.wrist_actions import denormalize_action44

        obs = torch.from_numpy(np.asarray(obs_pose9, dtype=np.float32)).reshape(1, 1, 3, 3)
        obs = obs.to(self.device)
        with torch.inference_mode():
            actions = self.policy(obs, actions=None, batch_size=1)
        out = actions.squeeze(0).detach().cpu().numpy()
        if self.normalize and self.action44_workspace is not None:
            out = denormalize_action44(self.action44_workspace, out)
        return out
