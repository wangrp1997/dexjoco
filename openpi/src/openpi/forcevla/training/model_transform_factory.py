# Source: openpi/training/config.py ModelTransformFactory (pi0.5) + force append
from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms
from openpi.forcevla.training.transforms import PadActionsOnly


class GroupFactory(Protocol):
    def __call__(self, model_config) -> _transforms.Group:
        ...


@dataclasses.dataclass(frozen=True)
class AppendForceToState(_transforms.DataTransformFn):
    """Concat privileged force tail onto state after pi0.5 prompt tokenization."""

    proprio_dim: int = 44

    def __call__(self, data: dict) -> dict:
        if "force" not in data:
            return data
        state = np.asarray(data["state"], dtype=np.float32)
        force = np.asarray(data.pop("force"), dtype=np.float32)
        data = dict(data)
        data["state"] = np.concatenate([state, force], axis=-1).astype(np.float32)
        return data


@dataclasses.dataclass(frozen=True)
class ForcePi05ModelTransformFactory(GroupFactory):
    default_prompt: str | None = None
    proprio_dim: int = 44

    def __call__(self, model_config) -> _transforms.Group:
        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=True,
                ),
                AppendForceToState(proprio_dim=self.proprio_dim),
                PadActionsOnly(model_config.action_dim),
            ],
        )
