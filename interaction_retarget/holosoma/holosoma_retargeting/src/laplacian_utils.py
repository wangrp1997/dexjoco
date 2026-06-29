"""Laplacian + interaction mesh utils (holosoma ``utils.py`` L394–461, numpy only).

Source: refs/holosoma/src/holosoma_retargeting/holosoma_retargeting/src/utils.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay


def create_interaction_mesh(vertices: np.ndarray):
    """Delaunay tetrahedra on interaction vertices."""
    tri = Delaunay(np.asarray(vertices, dtype=np.float64))
    return vertices, tri.simplices


def get_adjacency_list(tetrahedra, num_vertices: int) -> list[list[int]]:
    """Tetrahedra → undirected adjacency list."""
    adj = [set() for _ in range(num_vertices)]
    for tet in tetrahedra:
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = int(tet[i]), int(tet[j])
                adj[u].add(v)
                adj[v].add(u)
    return [list(s) for s in adj]


def calculate_laplacian_coordinates(
    vertices: np.ndarray,
    adj_list: list[list[int]],
    epsilon: float = 1e-6,
    uniform_weight: bool = True,
) -> np.ndarray:
    """Uniform or distance-weighted Laplacian coords (TopoRetarget Eq. 5 when uniform_weight=False)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    laplacian = np.zeros_like(vertices)
    for i, neighbors_indices in enumerate(adj_list):
        if not neighbors_indices:
            continue
        vi = vertices[i]
        neighbor_positions = vertices[neighbors_indices]
        if uniform_weight:
            weights = np.ones(len(neighbors_indices), dtype=np.float64)
        else:
            distances = np.linalg.norm(vi - neighbor_positions, axis=1)
            weights = 1.0 / (1.5 * distances + epsilon)
        sum_of_weights = np.sum(weights)
        center = np.sum(weights[:, np.newaxis] * neighbor_positions, axis=0) / sum_of_weights
        laplacian[i] = vi - center
    return laplacian


def create_interaction_adjacency(vertices: np.ndarray) -> list[list[int]]:
    """Convenience: Delaunay → adjacency (dexjoco alias)."""
    _, simplices = create_interaction_mesh(vertices)
    return get_adjacency_list(simplices, int(vertices.shape[0]))
