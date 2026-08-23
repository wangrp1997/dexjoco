"""Minimal retrieval-plus-constrained-deformation DexContactRAM prototype."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .learning_data import (
    RETRIEVAL_DESCRIPTOR_DIM,
    RetrievalEntry,
    RetrievalIndex,
    state46_to_action44,
)


@dataclass(frozen=True)
class SuccessfulSkillTrajectory:
    episode_index: int
    family_id: str
    split: str
    descriptor: np.ndarray
    state46: np.ndarray
    proprio_action44: np.ndarray
    demo_action44: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.demo_action44.shape[0])


@dataclass(frozen=True)
class DataDrivenActionLimits:
    right_position_step_m: float
    left_position_step_m: float
    right_rotation_step_rad: float
    left_rotation_step_rad: float
    finger_step_rad: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray


@dataclass(frozen=True)
class PrototypeSkillPlan:
    source_episode_index: int
    source_start_frame: int
    retrieval_distance: float
    raw_actions44: np.ndarray
    adapted_actions44: np.ndarray
    action_limits: DataDrivenActionLimits


def belief_to_retrieval_descriptor(
    belief18: np.ndarray,
    *,
    target_depth_m: float,
    nominal_peg_size_m: float,
) -> np.ndarray:
    """Convert the estimated 18D belief into the existing 14D skill query."""
    belief = np.asarray(belief18, dtype=np.float32).reshape(18)
    descriptor = np.concatenate(
        [
            belief[6:12],
            belief[12:18],
            np.asarray([target_depth_m, nominal_peg_size_m], dtype=np.float32),
        ]
    ).astype(np.float32)
    if descriptor.shape != (RETRIEVAL_DESCRIPTOR_DIM,):
        raise RuntimeError(f"unexpected retrieval descriptor shape {descriptor.shape}")
    return descriptor


def _rotation_step_angles(rotvec: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64))
    relative = rotations[1:] * rotations[:-1].inv()
    return relative.magnitude()


def _positive_quantile(values: np.ndarray, quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    positive = array[array > 1e-8]
    if positive.size == 0:
        return 1e-6
    return float(np.quantile(positive, quantile))


class SuccessfulSkillMemory:
    """Train-only successful trajectory gallery with learned motion limits."""

    def __init__(
        self,
        trajectories: list[SuccessfulSkillTrajectory],
        *,
        step_quantile: float = 0.95,
    ) -> None:
        if not trajectories:
            raise ValueError("skill memory requires at least one trajectory")
        if not 0.5 <= step_quantile < 1.0:
            raise ValueError("step_quantile must be in [0.5, 1.0)")
        self.trajectories = {item.episode_index: item for item in trajectories}
        entries = [
            RetrievalEntry(
                episode_index=item.episode_index,
                family_id=item.family_id,
                split=item.split,
                descriptor=item.descriptor,
            )
            for item in trajectories
        ]
        self.index = RetrievalIndex(entries)
        self.family_constants = self._family_constants(trajectories)
        self.limits = self._fit_limits(trajectories, step_quantile)

    @classmethod
    def load(
        cls,
        learning_dir: Path,
        *,
        gallery_split: str = "train",
        step_quantile: float = 0.95,
    ) -> "SuccessfulSkillMemory":
        import pyarrow.parquet as parquet

        root = Path(learning_dir)
        manifest = json.loads((root / "manifest.json").read_text())
        metadata = {
            int(item["episode_index"]): item for item in manifest["episodes"]
        }
        trajectories = []
        for path in sorted((root / "episodes").glob("episode_*.parquet")):
            episode_index = int(path.stem.rsplit("_", 1)[-1])
            item = metadata[episode_index]
            if str(item["split"]) != gallery_split:
                continue
            table = parquet.read_table(
                path,
                columns=["state46", "proprio_action44", "demo_action44"],
            )
            trajectories.append(
                SuccessfulSkillTrajectory(
                    episode_index=episode_index,
                    family_id=str(item["family_id"]),
                    split=str(item["split"]),
                    descriptor=np.asarray(item["descriptor"], dtype=np.float32),
                    state46=np.asarray(table["state46"].to_pylist(), dtype=np.float32),
                    proprio_action44=np.asarray(
                        table["proprio_action44"].to_pylist(), dtype=np.float32
                    ),
                    demo_action44=np.asarray(
                        table["demo_action44"].to_pylist(), dtype=np.float32
                    ),
                )
            )
        return cls(trajectories, step_quantile=step_quantile)

    @staticmethod
    def _family_constants(
        trajectories: list[SuccessfulSkillTrajectory],
    ) -> dict[str, tuple[float, float]]:
        grouped: dict[str, list[np.ndarray]] = {}
        for trajectory in trajectories:
            grouped.setdefault(trajectory.family_id, []).append(trajectory.descriptor[-2:])
        return {
            family_id: tuple(np.mean(np.stack(values), axis=0).tolist())
            for family_id, values in grouped.items()
        }

    @staticmethod
    def _fit_limits(
        trajectories: list[SuccessfulSkillTrajectory],
        quantile: float,
    ) -> DataDrivenActionLimits:
        actions = [trajectory.demo_action44 for trajectory in trajectories]
        deltas = [np.diff(action, axis=0) for action in actions if len(action) > 1]
        right_position = np.concatenate(
            [np.linalg.norm(delta[:, :3], axis=1) for delta in deltas]
        )
        left_position = np.concatenate(
            [np.linalg.norm(delta[:, 22:25], axis=1) for delta in deltas]
        )
        right_rotation = np.concatenate(
            [_rotation_step_angles(action[:, 3:6]) for action in actions]
        )
        left_rotation = np.concatenate(
            [_rotation_step_angles(action[:, 25:28]) for action in actions]
        )
        finger_delta = np.concatenate(
            [
                np.concatenate([np.abs(delta[:, 6:22]), np.abs(delta[:, 28:44])], axis=1)
                for delta in deltas
            ],
            axis=0,
        )
        all_actions = np.concatenate(actions, axis=0)
        return DataDrivenActionLimits(
            right_position_step_m=_positive_quantile(right_position, quantile),
            left_position_step_m=_positive_quantile(left_position, quantile),
            right_rotation_step_rad=_positive_quantile(right_rotation, quantile),
            left_rotation_step_rad=_positive_quantile(left_rotation, quantile),
            finger_step_rad=np.quantile(finger_delta, quantile, axis=0).astype(np.float32),
            action_min=all_actions.min(axis=0).astype(np.float32),
            action_max=all_actions.max(axis=0).astype(np.float32),
        )


def _align_rotation_sequence(
    source_rotvec: np.ndarray,
    source_start_rotvec: np.ndarray,
    current_rotvec: np.ndarray,
) -> np.ndarray:
    source_start = Rotation.from_rotvec(source_start_rotvec)
    current = Rotation.from_rotvec(current_rotvec)
    alignment = current * source_start.inv()
    source = Rotation.from_rotvec(source_rotvec)
    return (alignment * source).as_rotvec().astype(np.float32)


def _project_position(target: np.ndarray, previous: np.ndarray, limit: float) -> np.ndarray:
    delta = target - previous
    norm = float(np.linalg.norm(delta))
    if norm <= limit:
        return target
    return previous + delta * (limit / max(norm, 1e-8))


def _project_rotation(target: np.ndarray, previous: np.ndarray, limit: float) -> np.ndarray:
    previous_rotation = Rotation.from_rotvec(previous)
    target_rotation = Rotation.from_rotvec(target)
    delta = target_rotation * previous_rotation.inv()
    angle = float(delta.magnitude())
    if angle <= limit:
        return target
    clipped = delta.as_rotvec() * (limit / max(angle, 1e-8))
    return (Rotation.from_rotvec(clipped) * previous_rotation).as_rotvec()


class RetrievalAugmentedSkillPrototype:
    """Retrieve one successful trajectory and project it around current proprioception."""

    def __init__(self, memory: SuccessfulSkillMemory) -> None:
        self.memory = memory

    def plan(
        self,
        belief18: np.ndarray,
        state46: np.ndarray,
        *,
        family_id: str,
        horizon: int = 32,
    ) -> PrototypeSkillPlan:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if family_id not in self.memory.family_constants:
            raise KeyError(f"unknown geometry family {family_id!r}")
        target_depth, peg_size = self.memory.family_constants[family_id]
        descriptor = belief_to_retrieval_descriptor(
            belief18,
            target_depth_m=target_depth,
            nominal_peg_size_m=peg_size,
        )
        matches = self.memory.index.query(
            descriptor,
            family_id=family_id,
            top_k=1,
            gallery_split="train",
        )
        if not matches:
            raise RuntimeError("no successful skill matches the current belief")
        entry, distance = matches[0]
        source = self.memory.trajectories[entry.episode_index]
        count = min(horizon, source.num_frames)
        raw = source.demo_action44[:count].copy()
        source_proprio = source.proprio_action44[0]
        current = state46_to_action44(np.asarray(state46, dtype=np.float32).reshape(46))
        aligned = raw.copy()
        aligned[:, :3] += current[:3] - source_proprio[:3]
        aligned[:, 22:25] += current[22:25] - source_proprio[22:25]
        aligned[:, 3:6] = _align_rotation_sequence(
            raw[:, 3:6],
            source_proprio[3:6],
            current[3:6],
        )
        aligned[:, 25:28] = _align_rotation_sequence(
            raw[:, 25:28],
            source_proprio[25:28],
            current[25:28],
        )
        aligned[:, 6:22] += current[6:22] - source_proprio[6:22]
        aligned[:, 28:44] += current[28:44] - source_proprio[28:44]
        adapted = self._project(aligned, current)
        return PrototypeSkillPlan(
            source_episode_index=source.episode_index,
            source_start_frame=0,
            retrieval_distance=float(distance),
            raw_actions44=raw,
            adapted_actions44=adapted,
            action_limits=self.memory.limits,
        )

    def _project(self, aligned: np.ndarray, current: np.ndarray) -> np.ndarray:
        limits = self.memory.limits
        result = np.empty_like(aligned)
        previous = current.astype(np.float64).copy()
        finger_limits = np.maximum(limits.finger_step_rad, 1e-6)
        for row, target in enumerate(aligned.astype(np.float64)):
            projected = target.copy()
            projected[:3] = _project_position(
                target[:3], previous[:3], limits.right_position_step_m
            )
            projected[22:25] = _project_position(
                target[22:25], previous[22:25], limits.left_position_step_m
            )
            projected[3:6] = _project_rotation(
                target[3:6], previous[3:6], limits.right_rotation_step_rad
            )
            projected[25:28] = _project_rotation(
                target[25:28], previous[25:28], limits.left_rotation_step_rad
            )
            projected[6:22] = previous[6:22] + np.clip(
                target[6:22] - previous[6:22],
                -finger_limits[:16],
                finger_limits[:16],
            )
            projected[28:44] = previous[28:44] + np.clip(
                target[28:44] - previous[28:44],
                -finger_limits[16:],
                finger_limits[16:],
            )
            result[row] = projected
            previous = projected
        return result.astype(np.float32)
