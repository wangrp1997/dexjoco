"""Adapt successful DexContactRAM memories into RC-HB-SQP candidates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .belief_space_sqp import (
    ASSEMBLY_STATE_DIM,
    GRASP_WRENCH_DIM,
    RetrievedInsertionCandidate,
)
from .assembly_kinematics import wrist_twists_from_action44
from .learning_data import RetrievalEntry, RetrievalIndex
from .skill_prototype import belief_to_retrieval_descriptor


_PEG_IN_HOLE_POSITION = slice(0, 3)
_PEG_IN_HOLE_ROTVEC = slice(3, 6)
_GEOMETRY_SCALAR_OFFSET = 18
_APPROACH_HEIGHT_INDEX = _GEOMETRY_SCALAR_OFFSET + 2
_TARGET_DEPTH_INDEX = _GEOMETRY_SCALAR_OFFSET + 4
_PEG_SIZE_INDEX = _GEOMETRY_SCALAR_OFFSET + 5
_INSERT_OK_INDEX = _GEOMETRY_SCALAR_OFFSET + 8
_PEG_CONTACT_COUNT_INDEX = _GEOMETRY_SCALAR_OFFSET + 9
_TRAY_CONTACT_COUNT_INDEX = _GEOMETRY_SCALAR_OFFSET + 10


def assembly_state_from_belief18(belief18: np.ndarray) -> np.ndarray:
    """Return [lateral x/y, tilt x/y, signed axial progress] from belief18."""
    belief = np.asarray(belief18, dtype=np.float64).reshape(18)
    return np.asarray(
        [belief[0], belief[1], belief[3], belief[4], -belief[2]],
        dtype=np.float64,
    )


def assembly_states_from_geometry_features(features: np.ndarray) -> np.ndarray:
    """Convert stored geometry features into the five-dimensional SQP state."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] <= _TRAY_CONTACT_COUNT_INDEX:
        raise ValueError(f"geometry features have unexpected shape {values.shape}")
    position = values[:, _PEG_IN_HOLE_POSITION]
    rotation = values[:, _PEG_IN_HOLE_ROTVEC]
    signed_progress = -values[:, _APPROACH_HEIGHT_INDEX]
    return np.column_stack(
        [position[:, 0], position[:, 1], rotation[:, 0], rotation[:, 1], signed_progress]
    )


def _resample(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    source = np.arange(array.shape[0], dtype=np.float64)
    if array.ndim == 1:
        return np.interp(positions, source, array)
    return np.column_stack(
        [np.interp(positions, source, array[:, column]) for column in range(array.shape[1])]
    )


def candidate_sample_positions(
    suffix_length: int,
    horizon: int,
    *,
    source_span_steps: int | None = None,
) -> np.ndarray:
    if suffix_length < 2:
        raise ValueError("suffix_length must be at least two")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    final_source_index = suffix_length - 1
    if source_span_steps is not None:
        if source_span_steps <= 0:
            raise ValueError("source_span_steps must be positive")
        final_source_index = min(final_source_index, source_span_steps)
    return np.linspace(0.0, final_source_index, horizon + 1)


def _contact_modes(features: np.ndarray) -> tuple[str, ...]:
    values = np.asarray(features, dtype=np.float64)
    progress = -values[:, _APPROACH_HEIGHT_INDEX]
    insert = values[:, _INSERT_OK_INDEX] > 0.5
    peg_contact = values[:, _PEG_CONTACT_COUNT_INDEX] > 0.0
    tray_contact = values[:, _TRAY_CONTACT_COUNT_INDEX] > 0.0
    modes = []
    for depth, bottom, peg_ok, tray_ok in zip(
        progress,
        insert,
        peg_contact,
        tray_contact,
        strict=True,
    ):
        if bottom:
            modes.append("bottom")
        elif depth > 0.0:
            modes.append("inserted")
        elif peg_ok and tray_ok:
            modes.append("bimanual_align")
        else:
            modes.append("approach")
    return tuple(modes)


@dataclass(frozen=True)
class SuccessfulInsertionSkillRecord:
    episode_index: int
    family_id: str
    split: str
    descriptor: np.ndarray
    frame_index: np.ndarray
    states: np.ndarray
    geometry_features: np.ndarray
    demo_actions44: np.ndarray
    right_wrenches: np.ndarray
    left_wrenches: np.ndarray
    contact_modes: tuple[str, ...]
    target_depth_m: float
    nominal_peg_size_m: float


class SuccessfulSkillSQPAdapter:
    """Train-only successful memory with phase matching and wrench envelopes."""

    def __init__(
        self,
        records: list[SuccessfulInsertionSkillRecord],
        *,
        wrench_capacity_quantile: float = 0.99,
        capacity_confidence_multiplier: float = 2.326347874,
    ) -> None:
        if not records:
            raise ValueError("at least one successful skill record is required")
        if not 0.5 <= wrench_capacity_quantile < 1.0:
            raise ValueError("wrench_capacity_quantile must be in [0.5, 1.0)")
        self.records = {record.episode_index: record for record in records}
        self.index = RetrievalIndex(
            [
                RetrievalEntry(
                    episode_index=record.episode_index,
                    family_id=record.family_id,
                    split=record.split,
                    descriptor=record.descriptor,
                )
                for record in records
            ]
        )
        self.family_constants = self._family_constants(records)
        all_states = np.concatenate([record.states for record in records], axis=0)
        scale = np.std(all_states, axis=0)
        self.phase_scale = np.where(scale > 1e-8, scale, 1.0)
        right_norm = np.concatenate(
            [np.linalg.norm(record.right_wrenches, axis=1) for record in records]
        )
        left_norm = np.concatenate(
            [np.linalg.norm(record.left_wrenches, axis=1) for record in records]
        )
        self.right_capacity, self.right_capacity_std = self._wrench_envelope(
            right_norm,
            wrench_capacity_quantile,
            capacity_confidence_multiplier,
        )
        self.left_capacity, self.left_capacity_std = self._wrench_envelope(
            left_norm,
            wrench_capacity_quantile,
            capacity_confidence_multiplier,
        )
        finger_delta = np.concatenate(
            [
                np.abs(
                    np.diff(
                        np.concatenate(
                            [
                                record.demo_actions44[:, 6:22],
                                record.demo_actions44[:, 28:44],
                            ],
                            axis=1,
                        ),
                        axis=0,
                    )
                )
                for record in records
            ],
            axis=0,
        )
        self.finger_step_limit = np.maximum(
            np.quantile(finger_delta, 0.95, axis=0),
            1e-6,
        )

    @classmethod
    def load(
        cls,
        learning_dir: Path,
        estimation_dir: Path,
        *,
        gallery_split: str = "train",
        wrench_capacity_quantile: float = 0.99,
        capacity_confidence_multiplier: float = 2.326347874,
    ) -> "SuccessfulSkillSQPAdapter":
        import pyarrow.parquet as parquet

        learning_root = Path(learning_dir)
        estimation_root = Path(estimation_dir)
        manifest = json.loads((learning_root / "manifest.json").read_text())
        metadata = {
            int(item["episode_index"]): item for item in manifest["episodes"]
        }
        records = []
        for learning_path in sorted((learning_root / "episodes").glob("episode_*.parquet")):
            episode_index = int(learning_path.stem.rsplit("_", 1)[-1])
            item = metadata[episode_index]
            if str(item["split"]) != gallery_split:
                continue
            estimation_path = (
                estimation_root / "episodes" / f"episode_{episode_index:06d}.parquet"
            )
            if not estimation_path.is_file():
                raise FileNotFoundError(
                    f"missing aligned estimation episode {estimation_path}"
                )
            learning = parquet.read_table(
                learning_path,
                columns=["frame_index", "geometry_features", "demo_action44"],
            )
            estimation = parquet.read_table(
                estimation_path,
                columns=["frame_index", "sensor_wrist_wrench_world"],
            )
            learning_frame = np.asarray(learning["frame_index"].to_numpy(), dtype=np.int64)
            estimation_frame = np.asarray(
                estimation["frame_index"].to_numpy(), dtype=np.int64
            )
            if not np.array_equal(learning_frame, estimation_frame):
                raise ValueError(f"episode {episode_index} frame alignment mismatch")
            features = np.asarray(
                learning["geometry_features"].to_pylist(), dtype=np.float64
            )
            demo_actions44 = np.asarray(
                learning["demo_action44"].to_pylist(), dtype=np.float64
            )
            wrenches = np.asarray(
                estimation["sensor_wrist_wrench_world"].to_pylist(),
                dtype=np.float64,
            ).reshape(-1, 2, GRASP_WRENCH_DIM)
            records.append(
                SuccessfulInsertionSkillRecord(
                    episode_index=episode_index,
                    family_id=str(item["family_id"]),
                    split=str(item["split"]),
                    descriptor=np.asarray(item["descriptor"], dtype=np.float32),
                    frame_index=learning_frame,
                    states=assembly_states_from_geometry_features(features),
                    geometry_features=features,
                    demo_actions44=demo_actions44,
                    right_wrenches=wrenches[:, 0],
                    left_wrenches=wrenches[:, 1],
                    contact_modes=_contact_modes(features),
                    target_depth_m=float(np.median(features[:, _TARGET_DEPTH_INDEX])),
                    nominal_peg_size_m=float(np.median(features[:, _PEG_SIZE_INDEX])),
                )
            )
        return cls(
            records,
            wrench_capacity_quantile=wrench_capacity_quantile,
            capacity_confidence_multiplier=capacity_confidence_multiplier,
        )

    @staticmethod
    def _family_constants(
        records: list[SuccessfulInsertionSkillRecord],
    ) -> dict[str, tuple[float, float]]:
        grouped: dict[str, list[tuple[float, float]]] = {}
        for record in records:
            grouped.setdefault(record.family_id, []).append(
                (record.target_depth_m, record.nominal_peg_size_m)
            )
        return {
            family_id: tuple(np.mean(values, axis=0).tolist())
            for family_id, values in grouped.items()
        }

    @staticmethod
    def _wrench_envelope(
        norms: np.ndarray,
        quantile: float,
        confidence_multiplier: float,
    ) -> tuple[float, float]:
        values = np.asarray(norms, dtype=np.float64).reshape(-1)
        median = float(np.median(values))
        robust_std = 1.4826 * float(np.median(np.abs(values - median)))
        robust_std = max(robust_std, 1e-6)
        successful_quantile = float(np.quantile(values, quantile))
        return (
            successful_quantile + confidence_multiplier * robust_std,
            robust_std,
        )

    def retrieve_candidates(
        self,
        belief18: np.ndarray,
        *,
        family_id: str,
        top_k: int = 4,
        horizon: int = 12,
        source_span_steps: int | None = None,
    ) -> list[RetrievedInsertionCandidate]:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if family_id not in self.family_constants:
            raise KeyError(f"unknown geometry family {family_id!r}")
        target_depth, peg_size = self.family_constants[family_id]
        descriptor = belief_to_retrieval_descriptor(
            belief18,
            target_depth_m=target_depth,
            nominal_peg_size_m=peg_size,
        )
        matches = self.index.query(
            descriptor,
            family_id=family_id,
            top_k=top_k,
            gallery_split="train",
        )
        current_state = assembly_state_from_belief18(belief18)
        candidates = []
        for entry, retrieval_distance in matches:
            record = self.records[entry.episode_index]
            phase_distance = np.linalg.norm(
                (record.states - current_state) / self.phase_scale,
                axis=1,
            )
            anchor = int(np.argmin(phase_distance))
            anchor = min(anchor, len(record.states) - 2)
            suffix_states = record.states[anchor:]
            suffix_actions = record.demo_actions44[anchor:]
            suffix_right_wrench = record.right_wrenches[anchor:]
            suffix_left_wrench = record.left_wrenches[anchor:]
            suffix_modes = record.contact_modes[anchor:]
            positions = candidate_sample_positions(
                len(suffix_states),
                horizon,
                source_span_steps=source_span_steps,
            )
            control_positions = positions[1:]
            mode_positions = np.rint(control_positions).astype(np.int64)
            nominal_states = _resample(suffix_states, positions)
            nominal_actions = _resample(suffix_actions, positions)
            right_wrenches = _resample(suffix_right_wrench, control_positions)
            left_wrenches = _resample(suffix_left_wrench, control_positions)
            nominal_right_controls = wrist_twists_from_action44(
                nominal_actions,
                side="right",
            )
            nominal_left_controls = wrist_twists_from_action44(
                nominal_actions,
                side="left",
            )
            candidates.append(
                RetrievedInsertionCandidate(
                    skill_id=f"episode_{record.episode_index:06d}@{anchor}",
                    retrieval_distance=float(retrieval_distance),
                    nominal_states=nominal_states,
                    nominal_actions44=nominal_actions,
                    nominal_right_controls=nominal_right_controls,
                    nominal_left_controls=nominal_left_controls,
                    nominal_right_wrenches=right_wrenches,
                    nominal_left_wrenches=left_wrenches,
                    right_wrench_capacity=np.full(horizon, self.right_capacity),
                    left_wrench_capacity=np.full(horizon, self.left_capacity),
                    right_capacity_std=np.full(horizon, self.right_capacity_std),
                    left_capacity_std=np.full(horizon, self.left_capacity_std),
                    mean_state_disturbance=np.zeros(
                        (horizon, ASSEMBLY_STATE_DIM), dtype=np.float64
                    ),
                    contact_modes=tuple(suffix_modes[index] for index in mode_positions),
                )
            )
        return candidates
