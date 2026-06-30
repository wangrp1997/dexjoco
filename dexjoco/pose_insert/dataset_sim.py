"""PyTorch dataset for exported sim PoseInsert trajectories."""

from __future__ import annotations

import collections.abc as container_abcs
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pose_insert.poses import poses7_to_matrices
from pose_insert.splits import filter_demo_dir
from pose_insert.wrist_actions import (
    load_or_build_wrist_workspace,
    normalize_dual_wrist,
)


def pose7_sequence_to_pose9(poses7: np.ndarray) -> np.ndarray:
    """Convert (T, 7) relative poses to PoseDP (T, 3, 3) tensors columns x,y,translation."""
    mats = poses7_to_matrices(np.asarray(poses7, dtype=np.float64))
    return mats[:, :3, [0, 1, 3]]


def normalize_translation(workspace: np.ndarray, poses7: np.ndarray) -> np.ndarray:
    """Normalize xyz to [-1, 1] using workspace bounds (PoseInsert style)."""
    out = np.asarray(poses7, dtype=np.float64).copy()
    trans_max = np.array(
        [workspace[:, 0].max(), workspace[:, 1].max(), workspace[:, 2].max()],
        dtype=np.float64,
    )
    trans_min = np.array(
        [workspace[:, 3].min(), workspace[:, 4].min(), workspace[:, 5].min()],
        dtype=np.float64,
    )
    denom = np.maximum(trans_max - trans_min, 1e-6)
    out[:, :3] = (out[:, :3] - trans_min) / denom * 2.0 - 1.0
    return out


def compute_source_workspace(data_root: Path, split: str = "train") -> np.ndarray:
    """Build (N, 12) workspace rows like PoseInsert ``get_workspace.py`` (xyz min/max + rot)."""
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split dir missing: {split_dir}")

    rows: list[np.ndarray] = []
    for demo_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        rel_path = demo_dir / "source_in_target.npy"
        if not rel_path.is_file():
            continue
        rel = np.asarray(np.load(rel_path), dtype=np.float64)
        rows.append(
            np.array(
                [
                    rel[:, 0].max(),
                    rel[:, 1].max(),
                    rel[:, 2].max(),
                    rel[:, 0].min(),
                    rel[:, 1].min(),
                    rel[:, 2].min(),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )
        )

    if not rows:
        raise RuntimeError(f"No source_in_target.npy under {split_dir}")
    return np.stack(rows, axis=0)


def save_workspace(data_root: Path, workspace: np.ndarray) -> Path:
    path = data_root / "source_workspace.npy"
    np.save(path, np.asarray(workspace, dtype=np.float64))
    return path


def load_or_build_workspace(data_root: Path, split: str = "train", *, rebuild: bool = False) -> np.ndarray:
    path = data_root / "source_workspace.npy"
    if path.is_file() and not rebuild:
        workspace = np.load(path)
        if workspace.ndim == 2 and workspace.shape[1] >= 6:
            return workspace
    workspace = compute_source_workspace(data_root, split=split)
    save_workspace(data_root, workspace)
    return workspace


class SimPoseInsertDataset(Dataset):
    """Load ``train/{ep}/source_in_target.npy`` with PoseInsert sliding windows."""

    def __init__(
        self,
        data_root: Path | str,
        *,
        split: str = "train",
        num_obs: int = 1,
        num_action: int = 20,
        normalize: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.num_obs = int(num_obs)
        self.num_action = int(num_action)
        self.normalize = bool(normalize)
        self.split_dir = self.data_root / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {self.split_dir}")

        self.workspace = load_or_build_workspace(self.data_root, split=split) if normalize else None
        self.demo_dirs = sorted(
            p for p in self.split_dir.iterdir() if p.is_dir() and (p / "source_in_target.npy").is_file()
        )
        if not self.demo_dirs:
            raise RuntimeError(f"No demos with source_in_target.npy in {self.split_dir}")

        self.source_obs_poses: list[np.ndarray] = []
        self.action_poses: list[np.ndarray] = []
        self.demo_ids: list[str] = []

        for demo_dir in self.demo_dirs:
            rel7 = np.load(demo_dir / "source_in_target.npy")
            if rel7.shape[0] < self.num_action + 2:
                continue
            if self.normalize and self.workspace is not None:
                rel7 = normalize_translation(self.workspace, rel7)

            rel9 = pose7_sequence_to_pose9(rel7)
            pose_ids = list(range(1, len(rel9)))
            for cur_idx in range(len(pose_ids) - 1):
                obs_pad_before = max(0, self.num_obs - cur_idx - 1)
                action_pad_after = max(0, self.num_action - (len(pose_ids) - 1 - cur_idx))
                frame_begin = max(0, cur_idx - self.num_obs + 1)
                frame_end = min(len(pose_ids), cur_idx + self.num_action + 1)

                obs_pose_ids = pose_ids[:1] * obs_pad_before + pose_ids[frame_begin : cur_idx + 1]
                action_pose_ids = pose_ids[cur_idx + 1 : frame_end] + pose_ids[-1:] * action_pad_after

                self.source_obs_poses.append(rel9[obs_pose_ids])
                self.action_poses.append(rel9[action_pose_ids])
                self.demo_ids.append(demo_dir.name)

    def __len__(self) -> int:
        return len(self.source_obs_poses)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        obs = torch.from_numpy(np.asarray(self.source_obs_poses[index], dtype=np.float32))
        action = torch.from_numpy(np.asarray(self.action_poses[index], dtype=np.float32))
        return {
            "obs_source_pose": obs,
            "action_source_pose": action,
        }


def collate_pose_batch(batch):
    if isinstance(batch[0], container_abcs.Mapping):
        return {
            "obs_source_pose": torch.stack([b["obs_source_pose"] for b in batch], 0),
            "action_source_pose": torch.stack([b["action_source_pose"] for b in batch], 0),
        }
    raise TypeError(f"Unsupported batch type: {type(batch[0])}")


class BimanualPoseInsertDataset(Dataset):
    """Obs: peg-in-socket pose9; action: dual wrist12 horizon (fingers locked at rollout)."""

    def __init__(
        self,
        data_root: Path | str,
        *,
        split: str = "train",
        num_obs: int = 1,
        num_action: int = 20,
        normalize: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.num_obs = int(num_obs)
        self.num_action = int(num_action)
        self.normalize = bool(normalize)
        self.split_dir = self.data_root / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {self.split_dir}")

        self.pose_workspace = load_or_build_workspace(self.data_root, split=split) if normalize else None
        self.wrist_workspace = (
            load_or_build_wrist_workspace(self.data_root, split=split) if normalize else None
        )
        self.demo_dirs = sorted(
            p
            for p in self.split_dir.iterdir()
            if p.is_dir()
            and (p / "source_in_target.npy").is_file()
            and (p / "dual_wrist_action.npy").is_file()
        )
        if not self.demo_dirs:
            raise RuntimeError(
                f"No demos with source_in_target.npy + dual_wrist_action.npy in {self.split_dir}; re-export"
            )

        self.source_obs_poses: list[np.ndarray] = []
        self.action_wrists: list[np.ndarray] = []
        self.demo_ids: list[str] = []

        for demo_dir in self.demo_dirs:
            rel7 = np.load(demo_dir / "source_in_target.npy")
            wrist = np.load(demo_dir / "dual_wrist_action.npy")
            if rel7.shape[0] != wrist.shape[0]:
                raise ValueError(f"{demo_dir}: pose/wrist length mismatch")
            if rel7.shape[0] < self.num_action + 2:
                continue
            if self.normalize and self.pose_workspace is not None:
                rel7 = normalize_translation(self.pose_workspace, rel7)
            if self.normalize and self.wrist_workspace is not None:
                wrist = normalize_dual_wrist(self.wrist_workspace, wrist)

            rel9 = pose7_sequence_to_pose9(rel7)
            pose_ids = list(range(1, len(rel9)))
            for cur_idx in range(len(pose_ids) - 1):
                obs_pad_before = max(0, self.num_obs - cur_idx - 1)
                action_pad_after = max(0, self.num_action - (len(pose_ids) - 1 - cur_idx))
                frame_begin = max(0, cur_idx - self.num_obs + 1)
                frame_end = min(len(pose_ids), cur_idx + self.num_action + 1)

                obs_pose_ids = pose_ids[:1] * obs_pad_before + pose_ids[frame_begin : cur_idx + 1]
                action_frame_ids = pose_ids[cur_idx + 1 : frame_end] + pose_ids[-1:] * action_pad_after

                self.source_obs_poses.append(rel9[obs_pose_ids])
                self.action_wrists.append(wrist[action_frame_ids])
                self.demo_ids.append(demo_dir.name)

    def __len__(self) -> int:
        return len(self.source_obs_poses)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        obs = torch.from_numpy(np.asarray(self.source_obs_poses[index], dtype=np.float32))
        action = torch.from_numpy(np.asarray(self.action_wrists[index], dtype=np.float32))
        return {
            "obs_source_pose": obs,
            "action_dual_wrist": action,
        }


def collate_bimanual_batch(batch):
    if isinstance(batch[0], container_abcs.Mapping):
        return {
            "obs_source_pose": torch.stack([b["obs_source_pose"] for b in batch], 0),
            "action_dual_wrist": torch.stack([b["action_dual_wrist"] for b in batch], 0),
        }
    raise TypeError(f"Unsupported batch type: {type(batch[0])}")


class BimanualAction44Dataset(Dataset):
    """Load ``train/{ep}/source_in_target.npy`` + ``action44.npy`` sliding windows."""

    def __init__(
        self,
        data_root: Path | str,
        *,
        split: str = "train",
        num_obs: int = 1,
        num_action: int = 20,
        normalize: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.num_obs = int(num_obs)
        self.num_action = int(num_action)
        self.normalize = bool(normalize)
        self.split_dir = self.data_root / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {self.split_dir}")

        from pose_insert.wrist_actions import (
            load_or_build_action44_workspace,
            normalize_action44,
        )

        self.pose_workspace = load_or_build_workspace(self.data_root, split=split) if normalize else None
        self.action44_workspace = (
            load_or_build_action44_workspace(self.data_root, split=split) if normalize else None
        )
        self.demo_dirs = sorted(
            p
            for p in self.split_dir.iterdir()
            if p.is_dir()
            and (p / "source_in_target.npy").is_file()
            and (p / "action44.npy").is_file()
            and filter_demo_dir(p, split=split)
        )
        if not self.demo_dirs:
            raise RuntimeError(
                f"No demos with source_in_target.npy + action44.npy in {self.split_dir} "
                f"(split={split}); run backfill_action44.py"
            )

        self.source_obs_poses: list[np.ndarray] = []
        self.action44s: list[np.ndarray] = []
        self.demo_ids: list[str] = []

        for demo_dir in self.demo_dirs:
            rel7 = np.load(demo_dir / "source_in_target.npy")
            act44 = np.load(demo_dir / "action44.npy")
            if rel7.shape[0] != act44.shape[0]:
                raise ValueError(f"{demo_dir}: pose/action44 length mismatch")
            if rel7.shape[0] < self.num_action + 2:
                continue
            if self.normalize and self.pose_workspace is not None:
                rel7 = normalize_translation(self.pose_workspace, rel7)
            if self.normalize and self.action44_workspace is not None:
                act44 = normalize_action44(self.action44_workspace, act44)

            rel9 = pose7_sequence_to_pose9(rel7)
            pose_ids = list(range(1, len(rel9)))
            for cur_idx in range(len(pose_ids) - 1):
                obs_pad_before = max(0, self.num_obs - cur_idx - 1)
                action_pad_after = max(0, self.num_action - (len(pose_ids) - 1 - cur_idx))
                frame_begin = max(0, cur_idx - self.num_obs + 1)
                frame_end = min(len(pose_ids), cur_idx + self.num_action + 1)

                obs_pose_ids = pose_ids[:1] * obs_pad_before + pose_ids[frame_begin : cur_idx + 1]
                action_frame_ids = pose_ids[cur_idx + 1 : frame_end] + pose_ids[-1:] * action_pad_after

                self.source_obs_poses.append(rel9[obs_pose_ids])
                self.action44s.append(act44[action_frame_ids])
                self.demo_ids.append(demo_dir.name)

    def __len__(self) -> int:
        return len(self.source_obs_poses)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        obs = torch.from_numpy(np.asarray(self.source_obs_poses[index], dtype=np.float32))
        action = torch.from_numpy(np.asarray(self.action44s[index], dtype=np.float32))
        return {
            "obs_source_pose": obs,
            "action44": action,
        }


def collate_action44_batch(batch):
    if isinstance(batch[0], container_abcs.Mapping):
        return {
            "obs_source_pose": torch.stack([b["obs_source_pose"] for b in batch], 0),
            "action44": torch.stack([b["action44"] for b in batch], 0),
        }
    raise TypeError(f"Unsupported batch type: {type(batch[0])}")
