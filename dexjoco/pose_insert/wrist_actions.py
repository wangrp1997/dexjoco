"""Dual-arm wrist (12D) actions for bimanual PoseInsert."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

DUAL_WRIST_DIM = 12
DUAL_ACTION44_DIM = 44
RIGHT_WRIST_SLICE = slice(0, 6)
LEFT_WRIST_SLICE = slice(6, 12)


def arm23_to_wrist6(action23: np.ndarray) -> np.ndarray:
    """Mocap arm23 (xyz + quat wxyz + hand16) -> wrist6 (xyz + rotvec)."""
    a = np.asarray(action23, dtype=np.float64).reshape(23)
    rotvec = R.from_quat(a[3:7], scalar_first=True).as_rotvec()
    return np.concatenate([a[0:3], rotvec], dtype=np.float64)


def zarr_flat_to_dual_wrist12(action_flat: np.ndarray) -> np.ndarray:
    """Zarr [right23, left23] -> [right_wrist6, left_wrist6]."""
    flat = np.asarray(action_flat, dtype=np.float64).reshape(-1)
    if flat.shape[0] != 46:
        raise ValueError(f"Expected 46-d zarr action, got {flat.shape[0]}")
    right6 = arm23_to_wrist6(flat[:23])
    left6 = arm23_to_wrist6(flat[23:46])
    return np.concatenate([right6, left6], dtype=np.float64)


def zarr_flat_to_action44(action_flat: np.ndarray) -> np.ndarray:
    """Zarr [right23, left23] -> action44 (wrist rotvec + hands)."""
    flat = np.asarray(action_flat, dtype=np.float64).reshape(-1)
    right23 = flat[:23]
    left23 = flat[23:46]
    r_rot = R.from_quat(right23[3:7], scalar_first=True).as_rotvec()
    l_rot = R.from_quat(left23[3:7], scalar_first=True).as_rotvec()
    return np.concatenate(
        [right23[0:3], r_rot, right23[7:23], left23[0:3], l_rot, left23[7:23]],
        dtype=np.float64,
    )


def dual_wrist12_to_action44(
    wrist12: np.ndarray,
    *,
    right_hand: np.ndarray,
    left_hand: np.ndarray,
) -> np.ndarray:
    """Build action44 from dual wrist6 + locked finger ctrl."""
    w = np.asarray(wrist12, dtype=np.float64).reshape(12)
    rh = np.asarray(right_hand, dtype=np.float64).reshape(16)
    lh = np.asarray(left_hand, dtype=np.float64).reshape(16)
    return np.concatenate([w[0:6], rh, w[6:12], lh], dtype=np.float64)


def compute_wrist_workspace(data_root: Path, split: str = "train") -> np.ndarray:
    """Per-dim min/max over all ``dual_wrist_action.npy``; shape (12, 2)."""
    split_dir = data_root / split
    mins = np.full(DUAL_WRIST_DIM, np.inf, dtype=np.float64)
    maxs = np.full(DUAL_WRIST_DIM, -np.inf, dtype=np.float64)
    found = False
    for demo_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        path = demo_dir / "dual_wrist_action.npy"
        if not path.is_file():
            continue
        w = np.asarray(np.load(path), dtype=np.float64)
        mins = np.minimum(mins, w.min(axis=0))
        maxs = np.maximum(maxs, w.max(axis=0))
        found = True
    if not found:
        raise RuntimeError(f"No dual_wrist_action.npy under {split_dir}")
    return np.stack([mins, maxs], axis=1)


def save_wrist_workspace(data_root: Path, workspace: np.ndarray) -> Path:
    path = data_root / "wrist_workspace.npy"
    np.save(path, np.asarray(workspace, dtype=np.float64))
    return path


def load_or_build_wrist_workspace(data_root: Path, split: str = "train", *, rebuild: bool = False) -> np.ndarray:
    path = data_root / "wrist_workspace.npy"
    if path.is_file() and not rebuild:
        ws = np.load(path)
        if ws.shape == (DUAL_WRIST_DIM, 2):
            return ws
    ws = compute_wrist_workspace(data_root, split=split)
    save_wrist_workspace(data_root, ws)
    return ws


def normalize_dual_wrist(workspace: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    ws = np.asarray(workspace, dtype=np.float64).reshape(DUAL_WRIST_DIM, 2)
    w = np.asarray(wrist, dtype=np.float64).copy()
    mins, maxs = ws[:, 0], ws[:, 1]
    denom = np.maximum(maxs - mins, 1e-6)
    if w.ndim == 1:
        return (w - mins) / denom * 2.0 - 1.0
    return (w - mins) / denom * 2.0 - 1.0


def denormalize_dual_wrist(workspace: np.ndarray, wrist_norm: np.ndarray) -> np.ndarray:
    ws = np.asarray(workspace, dtype=np.float64).reshape(DUAL_WRIST_DIM, 2)
    w = np.asarray(wrist_norm, dtype=np.float64).copy()
    mins, maxs = ws[:, 0], ws[:, 1]
    denom = np.maximum(maxs - mins, 1e-6)
    if w.ndim == 1:
        return (w + 1.0) / 2.0 * denom + mins
    return (w + 1.0) / 2.0 * denom + mins


def compute_action44_workspace(data_root: Path, split: str = "train") -> np.ndarray:
    """Per-dim min/max over all ``action44.npy``; shape (44, 2)."""
    split_dir = data_root / split
    mins = np.full(DUAL_ACTION44_DIM, np.inf, dtype=np.float64)
    maxs = np.full(DUAL_ACTION44_DIM, -np.inf, dtype=np.float64)
    found = False
    for demo_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        path = demo_dir / "action44.npy"
        if not path.is_file():
            continue
        a = np.asarray(np.load(path), dtype=np.float64)
        mins = np.minimum(mins, a.min(axis=0))
        maxs = np.maximum(maxs, a.max(axis=0))
        found = True
    if not found:
        raise RuntimeError(f"No action44.npy under {split_dir}")
    return np.stack([mins, maxs], axis=1)


def save_action44_workspace(data_root: Path, workspace: np.ndarray) -> Path:
    path = data_root / "action44_workspace.npy"
    np.save(path, np.asarray(workspace, dtype=np.float64))
    return path


def load_or_build_action44_workspace(data_root: Path, split: str = "train", *, rebuild: bool = False) -> np.ndarray:
    path = data_root / "action44_workspace.npy"
    if path.is_file() and not rebuild:
        ws = np.load(path)
        if ws.shape == (DUAL_ACTION44_DIM, 2):
            return ws
    ws = compute_action44_workspace(data_root, split=split)
    save_action44_workspace(data_root, ws)
    return ws


def normalize_action44(workspace: np.ndarray, action44: np.ndarray) -> np.ndarray:
    ws = np.asarray(workspace, dtype=np.float64).reshape(DUAL_ACTION44_DIM, 2)
    a = np.asarray(action44, dtype=np.float64).copy()
    mins, maxs = ws[:, 0], ws[:, 1]
    denom = np.maximum(maxs - mins, 1e-6)
    if a.ndim == 1:
        return (a - mins) / denom * 2.0 - 1.0
    return (a - mins) / denom * 2.0 - 1.0


def denormalize_action44(workspace: np.ndarray, action_norm: np.ndarray) -> np.ndarray:
    ws = np.asarray(workspace, dtype=np.float64).reshape(DUAL_ACTION44_DIM, 2)
    a = np.asarray(action_norm, dtype=np.float64).copy()
    mins, maxs = ws[:, 0], ws[:, 1]
    denom = np.maximum(maxs - mins, 1e-6)
    if a.ndim == 1:
        return (a + 1.0) / 2.0 * denom + mins
    return (a + 1.0) / 2.0 * denom + mins
