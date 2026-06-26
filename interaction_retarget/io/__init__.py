"""Sidecar / zarr I/O."""

from interaction_retarget.io.npz import adjacency_from_padded, load_interaction_npz
from interaction_retarget.io.sidecar import (
    EpisodeSidecar,
    ObjectInteractionSnapshot,
    build_episode_sidecar,
    load_episode_sidecar_npz,
    save_episode_sidecar,
    timing_from_trace,
)
from interaction_retarget.io.zarr_io import discover_zarr_demos, iter_zarr_episodes, load_zarr_episode

__all__ = [
    "EpisodeSidecar",
    "ObjectInteractionSnapshot",
    "adjacency_from_padded",
    "build_episode_sidecar",
    "discover_zarr_demos",
    "iter_zarr_episodes",
    "load_episode_sidecar_npz",
    "load_interaction_npz",
    "load_zarr_episode",
    "save_episode_sidecar",
    "timing_from_trace",
]
