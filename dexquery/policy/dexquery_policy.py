"""DexQuery checkpoint loading and closed-loop action selection."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from dexquery.data.subtask_prompts import SubtaskPrompts
from dexquery.inference.phase_controller import PhaseController, PhaseControllerConfig, PhaseControllerState
from dexquery.models.dexquery_model import DexQueryModel, DexQueryModelConfig
from lerobot.utils.constants import ACTION, OBS_STATE

POLICY_CAMERA_KEYS: tuple[str, ...] = ("ego", "wrist_left", "wrist_right")
LEROBOT_CAMERA_KEYS: tuple[str, ...] = (
    "observation.images.ego",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)


@dataclass
class DexQueryPolicyConfig:
    checkpoint_path: Path
    task: str
    device: str = "cuda"
    chunk_size: int = 30
    replan_ratio: float = 0.8
    phase_controller: PhaseControllerConfig | None = None
    subtask_prompts: SubtaskPrompts | None = None


@dataclass
class DexQueryStepInfo:
    tray_prob: float
    peg_prob: float
    tray_ok: bool
    peg_ok: bool
    subtask_phase: int
    replanned: bool


class DexQueryPolicy:
    """Run DexQuery in sim with predicted outcomes and debounced phase switching."""

    def __init__(
        self,
        model: DexQueryModel,
        *,
        dataset_stats: dict[str, Any],
        subtask_prompts: SubtaskPrompts,
        device: str,
        chunk_size: int = 30,
        replan_ratio: float = 0.8,
        phase_controller: PhaseControllerConfig | None = None,
    ) -> None:
        self.model = model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.subtask_prompts = subtask_prompts
        self.chunk_size = int(chunk_size)
        self.replan_steps = max(1, int(round(self.chunk_size * float(replan_ratio))))
        self.phase_controller = PhaseController(phase_controller)
        self._state_mean = _stats_vector(dataset_stats, OBS_STATE, "mean")
        self._state_std = _stats_vector(dataset_stats, OBS_STATE, "std")
        self._action_mean = _stats_vector(dataset_stats, ACTION, "mean")
        self._action_std = _stats_vector(dataset_stats, ACTION, "std")
        self._action_queue: deque[np.ndarray] = deque()
        self._last_phase_state: PhaseControllerState | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,
        task: str,
        device: str = "cuda",
        replan_ratio: float = 0.8,
        phase_controller: PhaseControllerConfig | None = None,
        subtask_prompts: SubtaskPrompts | None = None,
    ) -> DexQueryPolicy:
        checkpoint_path = Path(checkpoint_path).expanduser()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = DexQueryModelConfig(**payload["model_config"])
        model = DexQueryModel(model_config)
        model.load_state_dict(payload["model"])

        stats_path = checkpoint_path.parent / "dataset_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing dataset stats for inference: {stats_path}")
        with open(stats_path, "r", encoding="utf-8") as f:
            dataset_stats = json.load(f)

        prompts = subtask_prompts
        if prompts is None:
            task_cfg = checkpoint_path.parent / "config.yaml"
            if task_cfg.exists():
                prompts = SubtaskPrompts.for_task(task, config_path=task_cfg)
            else:
                prompts = SubtaskPrompts.for_task(task)

        phase_cfg = phase_controller or _phase_controller_from_payload(payload, checkpoint_path.parent)
        return cls(
            model,
            dataset_stats=dataset_stats,
            subtask_prompts=prompts,
            device=device,
            chunk_size=model.config.chunk_size,
            replan_ratio=replan_ratio,
            phase_controller=phase_cfg,
        )

    def reset(self) -> None:
        self.phase_controller.reset()
        self._action_queue.clear()
        self._last_phase_state = None

    @property
    def last_phase_state(self) -> PhaseControllerState | None:
        return self._last_phase_state

    def select_action(self, observation: dict[str, Any]) -> tuple[np.ndarray, DexQueryStepInfo]:
        """Return one 44d action and debug info for the current observation."""
        replanned = False
        if not self._action_queue:
            chunk = self._predict_action_chunk(observation)
            for action in chunk[: self.replan_steps]:
                self._action_queue.append(action)
            replanned = True

        action = self._action_queue.popleft()
        assert self._last_phase_state is not None
        info = DexQueryStepInfo(
            tray_prob=self._last_phase_state.tray_prob,
            peg_prob=self._last_phase_state.peg_prob,
            tray_ok=self._last_phase_state.tray_ok,
            peg_ok=self._last_phase_state.peg_ok,
            subtask_phase=self._last_phase_state.subtask_phase,
            replanned=replanned,
        )
        return action, info

    def _predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        images = _observation_to_images(observation).unsqueeze(0).to(self.device)
        state = _normalize_vector(
            _observation_to_state(observation),
            self._state_mean,
            self._state_std,
        ).unsqueeze(0).to(self.device)

        prompts = self.subtask_prompts.as_list()
        patch_tokens = self.model.backbone(images)
        z_subtasks = self.model.subtask_encoder(patch_tokens, prompts)
        tray_logit, peg_logit = self.model.outcome_head(z_subtasks[:, 0, :], z_subtasks[:, 1, :])
        tray_prob = torch.sigmoid(tray_logit)[0].item()
        peg_prob = torch.sigmoid(peg_logit)[0].item()
        phase_state = self.phase_controller.update(tray_prob, peg_prob)
        self._last_phase_state = phase_state

        z_action = z_subtasks[:, phase_state.subtask_phase, :]
        pred_actions = self.model.action_head(z_action, state).pred_actions[0].detach().cpu()
        pred_actions = _denormalize_matrix(pred_actions, self._action_mean, self._action_std)
        return pred_actions.numpy().astype(np.float32)


def _phase_controller_from_payload(payload: dict[str, Any], checkpoint_dir: Path) -> PhaseControllerConfig:
    cfg = payload.get("config") or {}
    inference_cfg = cfg.get("inference", {})
    phase_cfg = inference_cfg.get("phase_controller", {})
    if phase_cfg:
        return PhaseControllerConfig(**phase_cfg)

    task_cfg_path = checkpoint_dir / "config.yaml"
    if task_cfg_path.exists():
        with open(task_cfg_path, "r", encoding="utf-8") as f:
            task_cfg = yaml.safe_load(f) or {}
        phase_cfg = task_cfg.get("inference", {}).get("phase_controller", {})
        if phase_cfg:
            return PhaseControllerConfig(**phase_cfg)
    return PhaseControllerConfig()


def _stats_vector(stats: dict[str, Any], key: str, field: str) -> torch.Tensor:
    values = stats[key][field]
    return torch.as_tensor(values, dtype=torch.float32).reshape(-1)


def _normalize_vector(
    tensor: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return (tensor - mean) / (std + eps)


def _denormalize_matrix(
    tensor: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    mean = mean.unsqueeze(0)
    std = std.unsqueeze(0)
    return tensor * std + mean


def _observation_to_state(observation: dict[str, Any]) -> torch.Tensor:
    if "observation.state" in observation:
        state = observation["observation.state"]
        return torch.as_tensor(state, dtype=torch.float32).reshape(-1)
    if "state" in observation and not isinstance(observation["state"], float):
        return torch.as_tensor(observation["state"], dtype=torch.float32).reshape(-1)

    values: list[float] = []
    idx = 0
    while f"state_{idx}" in observation:
        values.append(float(observation[f"state_{idx}"]))
        idx += 1
    if not values:
        raise KeyError("Could not find state vector in observation.")
    return torch.tensor(values, dtype=torch.float32)


def _observation_to_images(observation: dict[str, Any]) -> torch.Tensor:
    images = []
    for key in POLICY_CAMERA_KEYS:
        if key in observation:
            images.append(_to_image_tensor(observation[key]))
            continue
        lerobot_key = LEROBOT_CAMERA_KEYS[len(images)]
        if lerobot_key in observation:
            images.append(_to_image_tensor(observation[lerobot_key]))
            continue
        raise KeyError(f"Missing camera observation {key!r} / {lerobot_key!r}")
    return torch.stack(images, dim=0)


def _to_image_tensor(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()
    if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1)
    return tensor


def load_checkpoint(
    checkpoint_path: Path | str,
    *,
    task: str,
    device: str = "cuda",
    replan_ratio: float = 0.8,
    phase_controller: PhaseControllerConfig | None = None,
) -> DexQueryPolicy:
    return DexQueryPolicy.from_checkpoint(
        checkpoint_path,
        task=task,
        device=device,
        replan_ratio=replan_ratio,
        phase_controller=phase_controller,
    )
