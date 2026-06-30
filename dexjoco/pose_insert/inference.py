"""Load PoseDP checkpoint and run relative-pose inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pose_insert.dataset_sim import load_or_build_workspace
from pose_insert.models.pose_dp import PoseDP


class PoseInsertPolicyRunner:
    """Wrap trained PoseDP for sim rollout (privileged target pose from MuJoCo)."""

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

        self.workspace: np.ndarray | None = None
        if self.normalize:
            if data_root is None:
                raise ValueError("data_root required when normalize=True")
            self.workspace = load_or_build_workspace(Path(data_root))

        self.policy = PoseDP(num_action=self.num_action, obs_feature_dim=128, action_dim=9)
        state = torch.load(self.ckpt_path, map_location=self.device)
        self.policy.load_state_dict(state, strict=False)
        self.policy.to(self.device)
        self.policy.eval()

    def predict_pose9_horizon(self, obs_pose9: np.ndarray) -> np.ndarray:
        """Predict (horizon, 3, 3) relative pose waypoints from one obs pose9."""
        obs = torch.from_numpy(np.asarray(obs_pose9, dtype=np.float32)).reshape(1, 1, 3, 3)
        obs = obs.to(self.device)
        with torch.inference_mode():
            actions = self.policy(obs, actions=None, batch_size=1)
        return actions.squeeze(0).detach().cpu().numpy()
