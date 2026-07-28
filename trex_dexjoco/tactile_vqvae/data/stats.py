"""TacF6Stats — q01/q99 normalization for tactile windows.

DexJoCo: flat dim = 24 ([8,3]); upstream Sharpa was 60 ([10,6]).
Dim is taken from utils.lerobot_common.F6_DIM.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.lerobot_common import F6_DIM, F6_PER_FINGER, N_FINGERS_PER_HAND  # noqa: E402


@dataclass
class TacF6Stats:
    tacf6_min: np.ndarray
    tacf6_max: np.ndarray
    tacf6_mask: np.ndarray

    @classmethod
    def from_data_root(cls, data_root: str) -> "TacF6Stats":
        manifest_paths = sorted(
            glob.glob(os.path.join(data_root, "*", "pretrain_manifest.json"))
        )
        # Also accept manifest directly under data_root (single-task DexJoCo).
        direct = os.path.join(data_root, "pretrain_manifest.json")
        if os.path.isfile(direct):
            manifest_paths = [direct] + manifest_paths
        if not manifest_paths:
            raise FileNotFoundError(
                f"No pretrain_manifest.json under {data_root} or {data_root}/*/")

        all_q01, all_q99 = [], []
        for mp in manifest_paths:
            with open(mp, "r") as f:
                manifest = json.load(f)
            stats = manifest.get("statistics", {})
            block = stats.get("tactile_f6")
            if block:
                all_q01.append(np.array(block["q01"], dtype=np.float32))
                all_q99.append(np.array(block["q99"], dtype=np.float32))

        if all_q01:
            tacf6_min = np.min(np.stack(all_q01), axis=0)
            tacf6_max = np.max(np.stack(all_q99), axis=0)
        else:
            tacf6_min = np.full(F6_DIM, -1.0, dtype=np.float32)
            tacf6_max = np.full(F6_DIM, +1.0, dtype=np.float32)

        if tacf6_min.shape[0] != F6_DIM or tacf6_max.shape[0] != F6_DIM:
            raise ValueError(
                f"Expected F6 stats of dim {F6_DIM}, got "
                f"{tacf6_min.shape}/{tacf6_max.shape}")

        return cls(
            tacf6_min=tacf6_min,
            tacf6_max=tacf6_max,
            tacf6_mask=np.ones(F6_DIM, dtype=bool),
        )

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Min-max normalize full-hand F6 to [-1, 1].

        Accepts [..., N_FINGERS, D] or [..., F6_DIM] when size matches F6_DIM.
        For per-hand windows [T, n_fingers_per_hand, D], use `normalize_hand`.
        """
        orig_shape = x.shape
        flat = x.reshape(-1, F6_DIM).astype(np.float32, copy=False)
        denom = (self.tacf6_max - self.tacf6_min) + 1e-8
        normed = np.clip(2.0 * (flat - self.tacf6_min) / denom - 1.0, -1.0, 1.0)
        out = np.where(self.tacf6_mask, normed, flat)
        return out.reshape(orig_shape)

    def normalize_hand(self, x: np.ndarray, hand: int) -> np.ndarray:
        """Normalize a per-hand window [T, n_fingers_per_hand, D] with that hand's stats."""
        hand_dim = N_FINGERS_PER_HAND * F6_PER_FINGER
        sl = slice(hand * hand_dim, (hand + 1) * hand_dim)
        orig = x.shape
        flat = x.reshape(-1, hand_dim).astype(np.float32, copy=False)
        vmin = self.tacf6_min[sl]
        vmax = self.tacf6_max[sl]
        mask = self.tacf6_mask[sl]
        denom = (vmax - vmin) + 1e-8
        normed = np.clip(2.0 * (flat - vmin) / denom - 1.0, -1.0, 1.0)
        out = np.where(mask, normed, flat)
        return out.reshape(orig)

    def denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        orig_shape = x_norm.shape
        flat = x_norm.reshape(-1, F6_DIM).astype(np.float32, copy=False)
        denom = (self.tacf6_max - self.tacf6_min)
        out = (flat + 1.0) * 0.5 * denom + self.tacf6_min
        return out.reshape(orig_shape)

    def to_dict(self) -> dict:
        return {
            "tacf6_min":  self.tacf6_min.tolist(),
            "tacf6_max":  self.tacf6_max.tolist(),
            "tacf6_mask": self.tacf6_mask.astype(bool).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TacF6Stats":
        return cls(
            tacf6_min=np.array(d["tacf6_min"], dtype=np.float32),
            tacf6_max=np.array(d["tacf6_max"], dtype=np.float32),
            tacf6_mask=np.array(d["tacf6_mask"], dtype=bool),
        )

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)