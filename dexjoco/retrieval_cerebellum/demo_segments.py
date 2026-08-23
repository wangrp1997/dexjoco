"""Offline segmentation of post-grasp cerebellum training demonstrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DemoSegmentationConfig:
    grasp_confirm_frames: int = 5
    grasp_loss_confirm_frames: int = 8
    min_segment_frames: int = 16

    def __post_init__(self) -> None:
        if self.grasp_confirm_frames <= 0:
            raise ValueError("grasp_confirm_frames must be positive")
        if self.grasp_loss_confirm_frames <= 0:
            raise ValueError("grasp_loss_confirm_frames must be positive")
        if self.min_segment_frames <= 0:
            raise ValueError("min_segment_frames must be positive")


@dataclass(frozen=True)
class PostGraspSegment:
    episode_index: int
    start_frame: int | None
    end_frame: int | None
    insert_frame: int | None
    start_index: int | None
    end_index: int | None
    insert_index: int | None
    num_frames: int
    grasp_retained_to_insert: bool
    eligible: bool
    rejection_reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeLabels:
    episode_index: int
    frame_index: np.ndarray
    global_index: np.ndarray
    peg_ok: np.ndarray
    tray_ok: np.ndarray
    insert_ok: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "frame_index": np.asarray(self.frame_index, dtype=np.int64).reshape(-1),
            "global_index": np.asarray(self.global_index, dtype=np.int64).reshape(-1),
            "peg_ok": np.asarray(self.peg_ok, dtype=bool).reshape(-1),
            "tray_ok": np.asarray(self.tray_ok, dtype=bool).reshape(-1),
            "insert_ok": np.asarray(self.insert_ok, dtype=bool).reshape(-1),
        }
        lengths = {array.shape[0] for array in arrays.values()}
        if len(lengths) != 1:
            raise ValueError(f"Episode label arrays must share one length, got {lengths}")
        if arrays["frame_index"].size == 0:
            raise ValueError("Episode labels must not be empty")
        if np.any(np.diff(arrays["frame_index"]) < 0):
            raise ValueError("frame_index must be sorted")
        for name, array in arrays.items():
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def _first_confirmed_true(mask: np.ndarray, frames: int, *, start: int = 0) -> int | None:
    mask = np.asarray(mask, dtype=bool)
    streak = 0
    for index in range(max(0, int(start)), mask.shape[0]):
        streak = streak + 1 if mask[index] else 0
        if streak >= frames:
            return index - frames + 1
    return None


def segment_post_grasp_episode(
    labels: EpisodeLabels,
    config: DemoSegmentationConfig | None = None,
) -> PostGraspSegment:
    """Find a clean both-grasped segment ending at first insert contact."""
    cfg = config or DemoSegmentationConfig()
    both_grasped = labels.peg_ok & labels.tray_ok
    start_position = _first_confirmed_true(both_grasped, cfg.grasp_confirm_frames)
    if start_position is None:
        return _rejected(labels.episode_index, "both_grasp_not_confirmed")

    insert_positions = np.flatnonzero(labels.insert_ok & (np.arange(labels.insert_ok.size) >= start_position))
    insert_position = int(insert_positions[0]) if insert_positions.size else None
    loss_position = _first_confirmed_true(
        ~both_grasped,
        cfg.grasp_loss_confirm_frames,
        start=start_position + cfg.grasp_confirm_frames,
    )
    loss_before_insert = loss_position is not None and (
        insert_position is None or loss_position <= insert_position
    )

    if insert_position is not None:
        end_position = insert_position
        rejection_reason = None
    else:
        end_position = labels.frame_index.size - 1
        rejection_reason = "insert_not_observed"

    num_frames = end_position - start_position + 1
    eligible = bool(
        insert_position is not None
        and num_frames >= cfg.min_segment_frames
    )
    if rejection_reason is None and not eligible:
        rejection_reason = "segment_too_short"

    return PostGraspSegment(
        episode_index=labels.episode_index,
        start_frame=int(labels.frame_index[start_position]),
        end_frame=int(labels.frame_index[end_position]),
        insert_frame=(
            None if insert_position is None else int(labels.frame_index[insert_position])
        ),
        start_index=int(labels.global_index[start_position]),
        end_index=int(labels.global_index[end_position]),
        insert_index=(
            None if insert_position is None else int(labels.global_index[insert_position])
        ),
        num_frames=int(num_frames),
        grasp_retained_to_insert=not loss_before_insert,
        eligible=eligible,
        rejection_reason=rejection_reason,
    )


def _rejected(episode_index: int, reason: str) -> PostGraspSegment:
    return PostGraspSegment(
        episode_index=episode_index,
        start_frame=None,
        end_frame=None,
        insert_frame=None,
        start_index=None,
        end_index=None,
        insert_index=None,
        num_frames=0,
        grasp_retained_to_insert=False,
        eligible=False,
        rejection_reason=reason,
    )


def load_episode_labels(path: Path, *, insert_path: Path | None = None) -> EpisodeLabels:
    """Load one replay-label parquet without importing LeRobot."""
    import pyarrow.parquet as parquet

    table = parquet.read_table(
        path,
        columns=["index", "episode_index", "frame_index", "peg_ok", "tray_ok", "insert_ok"],
    )
    data = table.to_pydict()
    episode_values = np.asarray(data["episode_index"], dtype=np.int64)
    unique_episodes = np.unique(episode_values)
    if unique_episodes.shape != (1,):
        raise ValueError(f"Expected one episode in {path}, got {unique_episodes.tolist()}")
    insert_ok = np.asarray(data["insert_ok"], dtype=np.float64) > 0.5
    if insert_path is not None:
        insert_table = parquet.read_table(
            insert_path,
            columns=["index", "episode_index", "frame_index", "insert_ok"],
        )
        insert_data = insert_table.to_pydict()
        for key in ("index", "episode_index", "frame_index"):
            if not np.array_equal(
                np.asarray(data[key], dtype=np.int64),
                np.asarray(insert_data[key], dtype=np.int64),
            ):
                raise ValueError(f"{key} mismatch between {path} and {insert_path}")
        insert_ok = np.asarray(insert_data["insert_ok"], dtype=np.float64) > 0.5

    return EpisodeLabels(
        episode_index=int(unique_episodes[0]),
        frame_index=np.asarray(data["frame_index"], dtype=np.int64),
        global_index=np.asarray(data["index"], dtype=np.int64),
        peg_ok=np.asarray(data["peg_ok"], dtype=np.float64) > 0.5,
        tray_ok=np.asarray(data["tray_ok"], dtype=np.float64) > 0.5,
        insert_ok=insert_ok,
    )


def audit_label_files(
    paths: Iterable[Path],
    config: DemoSegmentationConfig | None = None,
    *,
    insert_label_dir: Path | None = None,
) -> list[PostGraspSegment]:
    cfg = config or DemoSegmentationConfig()
    return [
        segment_post_grasp_episode(
            load_episode_labels(
                path,
                insert_path=(None if insert_label_dir is None else insert_label_dir / path.name),
            ),
            cfg,
        )
        for path in paths
    ]


def load_state_action_segment(dataset_root: Path, segment: PostGraspSegment) -> dict[str, np.ndarray]:
    """Load the 46D state and 44D action rows referenced by an eligible segment."""
    if not segment.eligible or segment.start_index is None or segment.end_index is None:
        raise ValueError("segment must be eligible and have global index bounds")

    import pyarrow.compute as compute
    import pyarrow.dataset as dataset

    source = dataset.dataset(Path(dataset_root) / "data", format="parquet")
    index_field = dataset.field("index")
    table = source.to_table(
        columns=["index", "episode_index", "frame_index", "observation.state", "action"],
        filter=(index_field >= segment.start_index) & (index_field <= segment.end_index),
    )
    if table.num_rows != segment.num_frames:
        raise ValueError(
            f"Expected {segment.num_frames} rows for episode {segment.episode_index}, "
            f"got {table.num_rows}"
        )
    order = compute.sort_indices(table, sort_keys=[("index", "ascending")])
    table = compute.take(table, order)
    return {
        "index": np.asarray(table["index"].to_numpy(), dtype=np.int64),
        "episode_index": np.asarray(table["episode_index"].to_numpy(), dtype=np.int64),
        "frame_index": np.asarray(table["frame_index"].to_numpy(), dtype=np.int64),
        "state": np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
        "action": np.asarray(table["action"].to_pylist(), dtype=np.float32),
    }
