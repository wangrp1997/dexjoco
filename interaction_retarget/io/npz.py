"""Load interaction sidecar / canonical grasp npz snapshots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from interaction_retarget.constants import NUM_HAND_KEYPOINTS, NUM_OBJECT_SAMPLES


def adjacency_from_padded(adj_mat: np.ndarray) -> list[list[int]]:
    adj_mat = np.asarray(adj_mat, dtype=np.int32)
    out: list[list[int]] = []
    for row in adj_mat:
        neighbors = [int(j) for j in row if j >= 0]
        out.append(neighbors)
    return out


def unique_undirected_edges(adj_list: list[list[int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            if j <= i:
                continue
            key = (i, j)
            if key not in seen:
                seen.add(key)
                edges.append(key)
    return edges


def load_interaction_npz(npz_path: Path, *, prefix: str) -> dict[str, np.ndarray | list[list[int]]]:
    """Load one hand-object interaction snapshot from a sidecar npz."""
    data = np.load(npz_path)
    p = prefix
    hand_key = f"{p}_hand_points_obj"
    if hand_key not in data:
        raise KeyError(f"{npz_path}: missing {hand_key!r}")
    hand = np.asarray(data[hand_key], dtype=np.float64)
    obj = np.asarray(data[f"{p}_object_samples_obj"], dtype=np.float64)
    vertices = np.asarray(data[f"{p}_interaction_vertices_obj"], dtype=np.float64)
    adj = adjacency_from_padded(data[f"{p}_adjacency"])
    contact = np.asarray(data.get(f"{p}_contact_centers_obj", np.zeros((0, 3))), dtype=np.float64)
    grasp_frame = int(data[f"{p}_grasp_frame"][0])
    if hand.shape[0] != NUM_HAND_KEYPOINTS:
        raise ValueError(f"Expected {NUM_HAND_KEYPOINTS} hand points, got {hand.shape[0]}")
    if obj.shape[0] != NUM_OBJECT_SAMPLES:
        raise ValueError(f"Expected {NUM_OBJECT_SAMPLES} object samples, got {obj.shape[0]}")
    return {
        "hand_obj": hand,
        "object_samples_obj": obj,
        "vertices_obj": vertices,
        "adjacency": adj,
        "contact_centers_obj": contact,
        "grasp_frame": grasp_frame,
        "edges": unique_undirected_edges(adj),
    }
