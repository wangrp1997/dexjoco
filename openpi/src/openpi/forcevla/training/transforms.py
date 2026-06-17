# Source: openpi/forcevla/training/transforms.py (pi0.5 force path)
from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms
from openpi.shared.normalize import NormStats


@dataclasses.dataclass(frozen=True)
class NormalizeProprioAndForce(transforms.DataTransformFn):
    """Normalize proprio ``state`` and privileged ``force`` (aligned with ForceVLA state+wrench norm)."""

    norm_stats: dict[str, NormStats] | None
    proprio_dim: int = 44
    use_quantiles: bool = False

    def _normalize_array(self, values: np.ndarray, stats: NormStats) -> np.ndarray:
        dim = values.shape[-1]
        if self.use_quantiles:
            if stats.q01 is None or stats.q99 is None:
                raise ValueError("Quantile norm stats required but q01/q99 are missing.")
            q01 = stats.q01[..., :dim]
            q99 = stats.q99[..., :dim]
            ignore_mask = q99 - q01 < 1e-4
            center = (q99 + q01) / 2.0
            scaled = (values - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
            return np.where(ignore_mask, values - center, scaled).astype(np.float32)
        mean = stats.mean[..., :dim]
        std = stats.std[..., :dim]
        return ((values - mean) / (std + 1e-6)).astype(np.float32)

    def __call__(self, data: dict) -> dict:
        if self.norm_stats is None:
            return data
        data = dict(data)
        if "state" in self.norm_stats:
            state = np.asarray(data["state"], dtype=np.float32)
            stats = self.norm_stats["state"]
            stat_dim = int(np.asarray(stats.mean).shape[-1])
            if state.shape[-1] != stat_dim:
                raise ValueError(
                    f"State dim {state.shape[-1]} does not match norm stats dim {stat_dim}. "
                    "Re-run compute_norm_stats with a ForceVLA config after state46->proprio44 conversion."
                )
            data["state"] = self._normalize_array(state, stats)
        if "force" in data:
            if "force" not in self.norm_stats:
                raise ValueError(
                    "Force norm stats missing. Re-run compute_norm_stats with the matching ForceVLA config."
                )
            force = np.asarray(data["force"], dtype=np.float32)
            stats = self.norm_stats["force"]
            stat_dim = int(np.asarray(stats.mean).shape[-1])
            if force.shape[-1] != stat_dim:
                raise ValueError(
                    f"Force dim {force.shape[-1]} does not match norm stats dim {stat_dim}. "
                    "Use the norm stats file from the same force mode (wrist/finger/both)."
                )
            data["force"] = self._normalize_array(force, stats)
        return data


@dataclasses.dataclass(frozen=True)
class PadActionsOnly(transforms.DataTransformFn):
    model_action_dim: int

    def __call__(self, data: dict) -> dict:
        if "actions" in data:
            data = dict(data)
            data["actions"] = transforms.pad_to_dim(data["actions"], self.model_action_dim)
        return data
