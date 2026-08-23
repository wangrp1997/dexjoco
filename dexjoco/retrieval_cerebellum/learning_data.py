"""Object-centric training data for a retrieval-conditioned cerebellum."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


STATE_DIM = 46
ACTION_DIM = 44

GEOMETRY_VECTOR_COLUMNS = (
    "peg_tip_in_hole_position",
    "peg_in_hole_rotvec",
    "peg_in_right_palm_position",
    "peg_in_right_palm_rotvec",
    "tray_in_left_palm_position",
    "tray_in_left_palm_rotvec",
)
GEOMETRY_SCALAR_COLUMNS = (
    "lateral_error_m",
    "axis_error_rad",
    "approach_height_m",
    "insertion_depth_m",
    "target_depth_m",
    "nominal_peg_size_m",
    "peg_ok",
    "tray_ok",
    "insert_ok",
    "peg_contact_count",
    "tray_contact_count",
)
GEOMETRY_FEATURE_DIM = 3 * len(GEOMETRY_VECTOR_COLUMNS) + len(
    GEOMETRY_SCALAR_COLUMNS
)

RETRIEVAL_VECTOR_COLUMNS = (
    "peg_in_right_palm_position",
    "peg_in_right_palm_rotvec",
    "tray_in_left_palm_position",
    "tray_in_left_palm_rotvec",
)
RETRIEVAL_SCALAR_COLUMNS = (
    "target_depth_m",
    "nominal_peg_size_m",
)
RETRIEVAL_DESCRIPTOR_DIM = 3 * len(RETRIEVAL_VECTOR_COLUMNS) + len(
    RETRIEVAL_SCALAR_COLUMNS
)


@dataclass(frozen=True)
class RetrievalEntry:
    episode_index: int
    family_id: str
    split: str
    descriptor: np.ndarray

    def __post_init__(self) -> None:
        descriptor = np.asarray(self.descriptor, dtype=np.float32).reshape(-1)
        if descriptor.shape != (RETRIEVAL_DESCRIPTOR_DIM,):
            raise ValueError(
                f"descriptor must have shape ({RETRIEVAL_DESCRIPTOR_DIM},), "
                f"got {descriptor.shape}"
            )
        object.__setattr__(self, "descriptor", descriptor)


def state46_to_action44(state46: np.ndarray) -> np.ndarray:
    """Express 46D proprioception in the policy's 44D rotvec layout."""
    state = np.asarray(state46, dtype=np.float64)
    if state.shape[-1] != STATE_DIM:
        raise ValueError(f"state46 must end in {STATE_DIM} values, got {state.shape}")

    flat = state.reshape(-1, STATE_DIM)
    result = np.empty((flat.shape[0], ACTION_DIM), dtype=np.float64)
    result[:, :3] = flat[:, :3]
    result[:, 3:6] = Rotation.from_quat(
        flat[:, 3:7], scalar_first=True
    ).as_rotvec()
    result[:, 6:22] = flat[:, 14:30]
    result[:, 22:25] = flat[:, 7:10]
    result[:, 25:28] = Rotation.from_quat(
        flat[:, 10:14], scalar_first=True
    ).as_rotvec()
    result[:, 28:44] = flat[:, 30:46]
    return result.astype(np.float32).reshape(state.shape[:-1] + (ACTION_DIM,))


def geometry_feature_matrix(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    """Flatten object-centric primitive and contact columns per frame."""
    parts: list[np.ndarray] = []
    row_count: int | None = None
    for name in GEOMETRY_VECTOR_COLUMNS:
        values = np.asarray(columns[name], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"{name} must have shape (T, 3), got {values.shape}")
        row_count = values.shape[0] if row_count is None else row_count
        if values.shape[0] != row_count:
            raise ValueError("geometry columns must have equal row counts")
        parts.append(values)
    for name in GEOMETRY_SCALAR_COLUMNS:
        values = np.asarray(columns[name], dtype=np.float32).reshape(-1, 1)
        row_count = values.shape[0] if row_count is None else row_count
        if values.shape[0] != row_count:
            raise ValueError("geometry columns must have equal row counts")
        parts.append(values)
    features = np.concatenate(parts, axis=1)
    if features.shape[1] != GEOMETRY_FEATURE_DIM:
        raise RuntimeError(f"unexpected geometry feature shape {features.shape}")
    return features


def retrieval_descriptor(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    """Build the handoff-time query used to retrieve similar demonstrations."""
    parts: list[np.ndarray] = []
    for name in RETRIEVAL_VECTOR_COLUMNS:
        values = np.asarray(columns[name], dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 3:
            raise ValueError(f"{name} must have non-empty shape (T, 3), got {values.shape}")
        parts.append(values[0])
    for name in RETRIEVAL_SCALAR_COLUMNS:
        values = np.asarray(columns[name], dtype=np.float32).reshape(-1)
        if values.size == 0:
            raise ValueError(f"{name} must be non-empty")
        parts.append(values[:1])
    descriptor = np.concatenate(parts).astype(np.float32)
    if descriptor.shape != (RETRIEVAL_DESCRIPTOR_DIM,):
        raise RuntimeError(f"unexpected retrieval descriptor shape {descriptor.shape}")
    return descriptor


def episode_split(
    episode_index: int,
    family_id: str,
    *,
    seed: int = 0,
) -> str:
    """Assign a deterministic 80/10/10 episode-level split."""
    token = f"{seed}:{family_id}:{int(episode_index)}".encode()
    bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"


class RetrievalIndex:
    """Family-aware nearest-neighbor index with train-only gallery support."""

    def __init__(self, entries: Sequence[RetrievalEntry]) -> None:
        if not entries:
            raise ValueError("retrieval index requires at least one entry")
        self.entries = tuple(entries)
        normalization_entries = [entry for entry in entries if entry.split == "train"]
        if not normalization_entries:
            normalization_entries = list(entries)
        descriptors = np.stack(
            [entry.descriptor for entry in normalization_entries]
        ).astype(np.float64)
        self.mean = descriptors.mean(axis=0)
        scale = descriptors.std(axis=0)
        self.scale = np.where(scale > 1e-8, scale, 1.0)

    def query(
        self,
        descriptor: np.ndarray,
        *,
        family_id: str,
        top_k: int = 4,
        exclude_episode: int | None = None,
        gallery_split: str | None = "train",
    ) -> list[tuple[RetrievalEntry, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = np.asarray(descriptor, dtype=np.float64).reshape(-1)
        if query.shape != (RETRIEVAL_DESCRIPTOR_DIM,):
            raise ValueError(
                f"descriptor must have shape ({RETRIEVAL_DESCRIPTOR_DIM},), got {query.shape}"
            )
        normalized_query = (query - self.mean) / self.scale
        candidates: list[tuple[RetrievalEntry, float]] = []
        for entry in self.entries:
            if entry.family_id != family_id:
                continue
            if gallery_split is not None and entry.split != gallery_split:
                continue
            if exclude_episode is not None and entry.episode_index == exclude_episode:
                continue
            normalized = (entry.descriptor.astype(np.float64) - self.mean) / self.scale
            distance = float(np.linalg.norm(normalized - normalized_query))
            candidates.append((entry, distance))
        candidates.sort(key=lambda item: (item[1], item[0].episode_index))
        return candidates[:top_k]


def table_columns(path: Path) -> dict[str, np.ndarray]:
    """Read geometry sidecar columns into NumPy without exposing Arrow types."""
    import pyarrow.parquet as parquet

    table = parquet.read_table(path)
    result: dict[str, np.ndarray] = {}
    for name in GEOMETRY_VECTOR_COLUMNS:
        result[name] = np.asarray(table[name].to_pylist(), dtype=np.float32)
    for name in GEOMETRY_SCALAR_COLUMNS:
        result[name] = np.asarray(table[name].to_numpy())
    result["index"] = np.asarray(table["index"].to_numpy(), dtype=np.int64)
    result["episode_index"] = np.asarray(
        table["episode_index"].to_numpy(), dtype=np.int64
    )
    result["frame_index"] = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    result["family_id"] = np.asarray(table["family_id"].to_pylist(), dtype=object)
    return result
