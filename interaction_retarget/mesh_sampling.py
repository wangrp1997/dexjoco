"""Surface sampling adapted from holosoma_retargeting/src/utils.py."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import trimesh


def load_object_mesh(mesh_path: str | Path, *, scale: float = 1.0) -> trimesh.Trimesh:
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh at {mesh_path}, got {type(mesh)}")
    if scale != 1.0:
        mesh = mesh.copy()
        mesh.apply_scale(float(scale))
    return mesh


def weighted_surface_sampling(
    mesh: trimesh.Trimesh,
    sample_count: int,
    weight_func: Callable[[np.ndarray], float],
    *,
    seed: int = 42,
) -> np.ndarray:
    """Sample triangle surface points; face weight from ``weight_func(face_center)``."""
    rng = np.random.default_rng(seed)
    faces = mesh.faces
    vertices = mesh.vertices

    face_areas: list[float] = []
    face_centers: list[np.ndarray] = []
    for face in faces:
        v1, v2, v3 = vertices[face]
        face_areas.append(0.5 * float(np.linalg.norm(np.cross(v2 - v1, v3 - v1))))
        face_centers.append((v1 + v2 + v3) / 3.0)

    face_areas_arr = np.asarray(face_areas, dtype=np.float64)
    face_centers_arr = np.asarray(face_centers, dtype=np.float64)
    weights = np.asarray([weight_func(c) for c in face_centers_arr], dtype=np.float64)
    weights = np.maximum(weights, 1e-8)
    face_probs = face_areas_arr * weights
    face_probs /= face_probs.sum()

    sampled_indices = rng.choice(len(faces), size=sample_count, p=face_probs)
    points: list[np.ndarray] = []
    for face_idx in sampled_indices:
        v1, v2, v3 = vertices[faces[face_idx]]
        r1, r2 = rng.random(2)
        if r1 + r2 > 1.0:
            r1, r2 = 1.0 - r1, 1.0 - r2
        points.append(v1 + r1 * (v2 - v1) + r2 * (v3 - v1))
    return np.asarray(points, dtype=np.float64)


def contact_weighted_surface_sampling(
    mesh: trimesh.Trimesh,
    contact_centers_obj: np.ndarray,
    sample_count: int,
    *,
    sigma: float,
    seed: int = 42,
) -> np.ndarray:
    """Boost mesh faces near demo contact centers (object rest/body frame)."""
    centers = np.asarray(contact_centers_obj, dtype=np.float64).reshape(-1, 3)
    if centers.size == 0:
        points, _ = trimesh.sample.sample_surface_even(mesh, sample_count, seed=seed)
        return np.asarray(points, dtype=np.float64)

    def weight_func(face_center: np.ndarray) -> float:
        dists = np.linalg.norm(centers - face_center.reshape(1, 3), axis=1)
        return float(np.exp(-np.min(dists) ** 2 / (2.0 * sigma**2)))

    return weighted_surface_sampling(mesh, sample_count, weight_func, seed=seed)
