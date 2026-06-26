"""Interaction mesh Laplacian (holosoma / TopoRetarget Sec. 3.3–3.4, uniform weights)."""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay


def create_interaction_adjacency(vertices: np.ndarray) -> list[list[int]]:
    """Delaunay tetrahedra → undirected adjacency (holosoma ``get_adjacency_list``)."""
    tri = Delaunay(np.asarray(vertices, dtype=np.float64))
    num_vertices = int(vertices.shape[0])
    adj_sets = [set() for _ in range(num_vertices)]
    for tet in tri.simplices:
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = int(tet[i]), int(tet[j])
                adj_sets[u].add(v)
                adj_sets[v].add(u)
    return [list(s) for s in adj_sets]


def laplacian_coordinates(
    vertices: np.ndarray,
    adj_list: list[list[int]],
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Uniform-weight Laplacian coords (holosoma ``calculate_laplacian_coordinates``)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    laplacian = np.zeros_like(vertices)
    for i, neighbors in enumerate(adj_list):
        if not neighbors:
            continue
        vi = vertices[i]
        neighbor_positions = vertices[neighbors]
        center = neighbor_positions.mean(axis=0)
        laplacian[i] = vi - center
    _ = epsilon  # reserved for distance-weighted variant (TopoRetarget Eq. 5)
    return laplacian
