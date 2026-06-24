"""Minimal ResFiT MLP helpers (no hydra / tabulate deps)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributions as pyd
from torch import nn


@dataclass
class ActorConfig:
    hidden_dim: int = 1024
    dropout: float = 0.0
    num_layers: int = 2
    use_layer_norm: bool = True
    action_scale: float = 0.1
    actor_last_layer_init_scale: float | None = 0.0
    actor_last_layer_init_distribution: str = "normal"


def build_fc(
    in_dim: int,
    hidden_dim: int,
    action_dim: int,
    num_layer: int,
    layer_norm: int,
    dropout: float,
    use_layer_norm: bool = True,
) -> nn.Sequential:
    dims = [in_dim, *[hidden_dim for _ in range(num_layer)]]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if use_layer_norm and layer_norm == 1:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if use_layer_norm and layer_norm == 2 and i == num_layer - 1:
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-1], action_dim))
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def initialize_layer_weights(layer: nn.Linear, distribution: str, scale: float | None = None) -> None:
    if distribution == "normal" and scale is not None:
        nn.init.normal_(layer.weight, mean=0.0, std=scale)
        if layer.bias is not None:
            nn.init.normal_(layer.bias, mean=0.0, std=scale)


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        if isinstance(scale, float):
            scale = torch.ones_like(loc) * scale
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def sample(self, clip=None, sample_shape=None):
        shape = sample_shape if sample_shape is not None else self.loc.shape
        eps = torch.randn(shape, device=self.loc.device, dtype=self.loc.dtype)
        action = self.loc + eps * self.scale
        if clip is not None:
            action = action.clamp(-clip, clip)
        return torch.clamp(action, self.low + self.eps, self.high - self.eps)
