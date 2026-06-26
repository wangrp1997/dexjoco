"""Distill canonical grasp prototypes (δ*) from episode sidecars."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from interaction_retarget.constants import (
    CONTACT_SAMPLE_SIGMA_M,
    NUM_HAND_KEYPOINTS,
    NUM_INTERACTION_VERTICES,
    NUM_OBJECT_SAMPLES,
    PEG_MESH_PATH,
    TRAY_MESH_PATH,
)
from interaction_retarget.laplacian import create_interaction_adjacency, laplacian_coordinates
from interaction_retarget.mesh.sampling import contact_weighted_surface_sampling, load_object_mesh
from interaction_retarget.io.npz import adjacency_from_padded, load_interaction_npz


@dataclass
class DistillReport:
    object_name: str
    num_episodes_used: int
    episode_indices: list[int]
    excluded_episode_indices: list[int]
    hand_points_std_mean_m: float
    hand_points_std_max_m: float
    laplacian_spread_mean_m: float
    representative_episode_index: int
    representative_laplacian_rmse: float


@dataclass
class CanonicalGraspPrototype:
    object_name: str
    hand_side: str
    hand_points_obj: np.ndarray
    object_samples_obj: np.ndarray
    interaction_vertices_obj: np.ndarray
    laplacian_coords: np.ndarray
    adjacency: list[list[int]]
    hand_points_std: np.ndarray
    per_episode_laplacian_rmse: np.ndarray
    source_episode_indices: np.ndarray
    report: DistillReport


def _mesh_path(object_name: str) -> Path:
    if object_name == "tray":
        return TRAY_MESH_PATH
    if object_name == "peg":
        return PEG_MESH_PATH
    raise ValueError(object_name)


def _hand_side(object_name: str) -> str:
    return "left" if object_name == "tray" else "right"


def _pooled_contact_centers(contact_list: list[np.ndarray], *, max_points: int = 256) -> np.ndarray:
    chunks = [np.asarray(c, dtype=np.float64).reshape(-1, 3) for c in contact_list if c.size]
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64)
    pooled = np.concatenate(chunks, axis=0)
    if pooled.shape[0] <= max_points:
        return pooled
    rng = np.random.default_rng(0)
    idx = rng.choice(pooled.shape[0], size=max_points, replace=False)
    return pooled[idx]


def _adjacency_to_padded(adjacency: list[list[int]]) -> np.ndarray:
    max_deg = max((len(n) for n in adjacency), default=0)
    adj_mat = -np.ones((len(adjacency), max_deg), dtype=np.int32)
    for i, neighbors in enumerate(adjacency):
        adj_mat[i, : len(neighbors)] = neighbors
    return adj_mat


def _episode_excluded(entry: dict[str, Any], *, object_name: str, exclude_fallback: bool) -> bool:
    if not entry.get(f"has_{object_name}", False):
        return True
    if not exclude_fallback:
        return False
    timing = entry.get("timing", {})
    if object_name == "tray" and timing.get("left_grasp_fallback"):
        return True
    if object_name == "peg" and timing.get("right_grasp_fallback"):
        return True
    warnings = entry.get("timing_warnings", [])
    flag = f"{object_name}_grasp_used_fallback"
    return flag in warnings


def load_episode_snapshots(
    sidecar_dir: Path,
    *,
    object_name: str,
    exclude_fallback: bool = False,
) -> tuple[list[int], list[dict[str, np.ndarray | list[list[int]]]], list[int]]:
    """Return (used_indices, snapshots, excluded_indices)."""
    manifest_path = sidecar_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    used: list[int] = []
    excluded: list[int] = []
    snaps: list[dict[str, np.ndarray | list[list[int]]]] = []
    for entry in manifest["episodes"]:
        ep_idx = int(entry["episode_index"])
        if _episode_excluded(entry, object_name=object_name, exclude_fallback=exclude_fallback):
            excluded.append(ep_idx)
            continue
        npz_path = Path(entry["npz_path"])
        if not npz_path.is_file():
            npz_path = sidecar_dir / f"episode_{ep_idx:03d}" / "interaction_sidecar.npz"
        snaps.append(load_interaction_npz(npz_path, prefix=object_name))
        used.append(ep_idx)
    return used, snaps, excluded


def distill_canonical_grasp(
    snapshots: list[dict[str, np.ndarray | list[list[int]]]],
    *,
    object_name: str,
    episode_indices: list[int],
    excluded_episode_indices: list[int],
    sample_seed: int = 0,
) -> CanonicalGraspPrototype:
    """Build one canonical δ* from aligned episode snapshots."""
    if not snapshots:
        raise ValueError(f"No snapshots to distill for {object_name}")

    hands = np.stack([np.asarray(s["hand_obj"], dtype=np.float64) for s in snapshots], axis=0)
    hand_median = np.median(hands, axis=0)
    hand_std = np.std(hands, axis=0)

    contacts = [np.asarray(s["contact_centers_obj"], dtype=np.float64) for s in snapshots]
    pooled_contacts = _pooled_contact_centers(contacts)
    mesh = load_object_mesh(_mesh_path(object_name))
    object_samples = contact_weighted_surface_sampling(
        mesh,
        pooled_contacts,
        NUM_OBJECT_SAMPLES,
        sigma=CONTACT_SAMPLE_SIGMA_M,
        seed=sample_seed,
    )

    vertices = np.concatenate([hand_median, object_samples], axis=0)
    if vertices.shape[0] != NUM_INTERACTION_VERTICES:
        raise ValueError(f"Expected {NUM_INTERACTION_VERTICES} vertices, got {vertices.shape[0]}")

    adjacency = create_interaction_adjacency(vertices)
    laplacian = laplacian_coordinates(vertices, adjacency)

    # Per-episode Laplacian RMSE vs canonical (each ep uses its own graph).
    rmse_list: list[float] = []
    for snap in snapshots:
        ep_lap = np.asarray(snap["vertices_obj"], dtype=np.float64)
        ep_adj = snap["adjacency"]
        ep_delta = laplacian_coordinates(ep_lap, ep_adj)
        diff = ep_delta - laplacian
        rmse_list.append(float(np.sqrt(np.mean(diff**2))))
    rmse_arr = np.asarray(rmse_list, dtype=np.float64)
    best_i = int(np.argmin(rmse_arr))

    report = DistillReport(
        object_name=object_name,
        num_episodes_used=len(snapshots),
        episode_indices=list(episode_indices),
        excluded_episode_indices=list(excluded_episode_indices),
        hand_points_std_mean_m=float(np.mean(np.linalg.norm(hand_std, axis=1))),
        hand_points_std_max_m=float(np.max(np.linalg.norm(hand_std, axis=1))),
        laplacian_spread_mean_m=float(np.mean(rmse_arr)),
        representative_episode_index=int(episode_indices[best_i]),
        representative_laplacian_rmse=float(rmse_arr[best_i]),
    )

    return CanonicalGraspPrototype(
        object_name=object_name,
        hand_side=_hand_side(object_name),
        hand_points_obj=hand_median,
        object_samples_obj=object_samples,
        interaction_vertices_obj=vertices,
        laplacian_coords=laplacian,
        adjacency=adjacency,
        hand_points_std=hand_std,
        per_episode_laplacian_rmse=rmse_arr,
        source_episode_indices=np.asarray(episode_indices, dtype=np.int32),
        report=report,
    )


def save_canonical_grasp(prototype: CanonicalGraspPrototype, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adj_mat = _adjacency_to_padded(prototype.adjacency)
    np.savez_compressed(
        out_path,
        object_name=np.asarray([prototype.object_name]),
        hand_side=np.asarray([prototype.hand_side]),
        hand_points_obj=prototype.hand_points_obj,
        object_samples_obj=prototype.object_samples_obj,
        interaction_vertices_obj=prototype.interaction_vertices_obj,
        laplacian_coords=prototype.laplacian_coords,
        adjacency=adj_mat,
        hand_points_std=prototype.hand_points_std,
        per_episode_laplacian_rmse=prototype.per_episode_laplacian_rmse,
        source_episode_indices=prototype.source_episode_indices,
        num_hand_keypoints=np.asarray([NUM_HAND_KEYPOINTS], dtype=np.int32),
        num_object_samples=np.asarray([NUM_OBJECT_SAMPLES], dtype=np.int32),
    )
    meta_path = out_path.with_suffix(".json")
    meta = {
        "object_name": prototype.object_name,
        "hand_side": prototype.hand_side,
        "npz_path": str(out_path),
        "report": asdict(prototype.report),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_path


def load_canonical_grasp(npz_path: Path) -> dict[str, np.ndarray | list[list[int]]]:
    data = np.load(npz_path)
    adj = adjacency_from_padded(data["adjacency"])
    return {
        "object_name": str(data["object_name"][0]),
        "hand_side": str(data["hand_side"][0]),
        "hand_points_obj": np.asarray(data["hand_points_obj"], dtype=np.float64),
        "object_samples_obj": np.asarray(data["object_samples_obj"], dtype=np.float64),
        "interaction_vertices_obj": np.asarray(data["interaction_vertices_obj"], dtype=np.float64),
        "laplacian_coords": np.asarray(data["laplacian_coords"], dtype=np.float64),
        "adjacency": adj,
        "hand_points_std": np.asarray(data["hand_points_std"], dtype=np.float64),
        "per_episode_laplacian_rmse": np.asarray(data["per_episode_laplacian_rmse"], dtype=np.float64),
        "source_episode_indices": np.asarray(data["source_episode_indices"], dtype=np.int32),
    }


def distill_from_sidecar_dir(
    sidecar_dir: Path,
    *,
    out_dir: Path | None = None,
    exclude_fallback: bool = False,
    sample_seed: int = 0,
) -> dict[str, CanonicalGraspPrototype]:
    sidecar_dir = Path(sidecar_dir)
    out_dir = out_dir if out_dir is not None else sidecar_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prototypes: dict[str, CanonicalGraspPrototype] = {}
    summary: dict[str, Any] = {"sidecar_dir": str(sidecar_dir), "objects": {}}

    for object_name in ("tray", "peg"):
        used, snaps, excluded = load_episode_snapshots(
            sidecar_dir,
            object_name=object_name,
            exclude_fallback=exclude_fallback,
        )
        proto = distill_canonical_grasp(
            snaps,
            object_name=object_name,
            episode_indices=used,
            excluded_episode_indices=excluded,
            sample_seed=sample_seed,
        )
        out_npz = out_dir / f"canonical_{object_name}_grasp.npz"
        save_canonical_grasp(proto, out_npz)
        prototypes[object_name] = proto
        summary["objects"][object_name] = {
            "npz_path": str(out_npz),
            "meta_path": str(out_npz.with_suffix(".json")),
            "report": asdict(proto.report),
        }

    summary_path = out_dir / "canonical_grasp_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return prototypes
