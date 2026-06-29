"""Port of ContactOpt ``diffcontact.py`` (capsule SFD → contact value), numpy only.

Source: refs/contactopt/contactopt/diffcontact.py
"""

from __future__ import annotations

import numpy as np


def _normalize_rows(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _knn_query_to_target(
    query: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per query point: dist, index, nearest position. (Q,3),(T,3) → (Q,),(Q,),(Q,3)."""
    query = np.asarray(query, dtype=np.float64).reshape(-1, 3)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    if query.size == 0 or targets.size == 0:
        z = np.zeros((query.shape[0],), dtype=np.float64)
        return z, np.zeros((query.shape[0],), dtype=np.int64), np.zeros((query.shape[0], 3), dtype=np.float64)
    diff = query[:, None, :] - targets[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    idx = np.argmin(d2, axis=1)
    dist = np.sqrt(d2[np.arange(query.shape[0]), idx])
    return dist, idx, targets[idx]


def _knn_target_to_query(
    mesh: np.ndarray, queries: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per mesh point: dist, index, nearest query. (V,3),(Q,3) → (V,),(V,),(V,3)."""
    mesh = np.asarray(mesh, dtype=np.float64).reshape(-1, 3)
    queries = np.asarray(queries, dtype=np.float64).reshape(-1, 3)
    if mesh.size == 0 or queries.size == 0:
        z = np.zeros((mesh.shape[0],), dtype=np.float64)
        return z, np.zeros((mesh.shape[0],), dtype=np.int64), np.zeros((mesh.shape[0], 3), dtype=np.float64)
    diff = mesh[:, None, :] - queries[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    idx = np.argmin(d2, axis=1)
    dist = np.sqrt(d2[np.arange(mesh.shape[0]), idx])
    return dist, idx, queries[idx]


def capsule_sdf(
    mesh_verts: np.ndarray,
    mesh_normals: np.ndarray,
    query_points: np.ndarray,
    query_normals: np.ndarray,
    *,
    caps_rad: float,
    caps_top: float,
    caps_bot: float,
    foreach_on_mesh: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct port of ContactOpt ``capsule_sdf``."""
    mesh_verts = np.asarray(mesh_verts, dtype=np.float64).reshape(-1, 3)
    mesh_normals = _normalize_rows(mesh_normals.reshape(-1, 3))
    query_points = np.asarray(query_points, dtype=np.float64).reshape(-1, 3)
    query_normals = _normalize_rows(query_normals.reshape(-1, 3))

    if foreach_on_mesh:
        _, nn_idx, nearest_pos = _knn_target_to_query(mesh_verts, query_points)
        capsule_tops = mesh_verts + mesh_normals * float(caps_top)
        capsule_bots = mesh_verts + mesh_normals * float(caps_bot)
        delta_top = nearest_pos - capsule_tops
        normal_dot = np.sum(
            mesh_normals * query_normals[np.clip(nn_idx, 0, max(query_normals.shape[0] - 1, 0))],
            axis=1,
        )
    else:
        _, nn_idx, nearest_pos = _knn_query_to_target(query_points, mesh_verts)
        closest = mesh_verts[np.clip(nn_idx, 0, max(mesh_verts.shape[0] - 1, 0))]
        closest_normals = mesh_normals[np.clip(nn_idx, 0, max(mesh_normals.shape[0] - 1, 0))]
        capsule_tops = closest + closest_normals * float(caps_top)
        capsule_bots = closest + closest_normals * float(caps_bot)
        delta_top = query_points - capsule_tops
        normal_dot = np.sum(query_normals * closest_normals, axis=1)

    bot_to_top = capsule_bots - capsule_tops
    along_axis = np.sum(delta_top * bot_to_top, axis=1)
    top_to_bot_sq = np.sum(bot_to_top * bot_to_top, axis=1)
    h = np.clip(along_axis / np.maximum(top_to_bot_sq, 1e-12), 0.0, 1.0)
    dist_to_axis = np.linalg.norm(delta_top - bot_to_top * h[:, None], axis=1)
    sdf = dist_to_axis / max(float(caps_rad), 1e-8)
    return sdf, normal_dot


def sdf_to_contact(
    sdf: np.ndarray,
    dot_normal: np.ndarray | None = None,
    *,
    method: int = 0,
) -> np.ndarray:
    """Port of ContactOpt ``sdf_to_contact``."""
    sdf = np.asarray(sdf, dtype=np.float64)
    if method == 0:
        c = 1.0 / (sdf + 1e-4)
    elif method == 1:
        c = -sdf + 2.0
    elif method == 2:
        c = np.power(1.0 / (sdf + 1e-4), 2)
    elif method == 3:
        c = 1.0 / (1.0 + np.exp(sdf - 2.5))
    elif method == 4:
        dn = np.zeros_like(sdf) if dot_normal is None else np.asarray(dot_normal, dtype=np.float64)
        c = (-dn / 2.0 + 0.5) / (sdf + 1e-4)
    else:
        c = 1.0 / (sdf + 1e-4)
    return np.clip(c, 0.0, 1.0)


def calculate_contact_capsule(
    hand_verts: np.ndarray,
    hand_normals: np.ndarray,
    object_verts: np.ndarray,
    object_normals: np.ndarray,
    *,
    caps_top: float = 0.0005,
    caps_bot: float = -0.0015,
    caps_rad: float = 0.001,
    caps_on_hand: bool = False,
    contact_norm_method: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of ContactOpt ``calculate_contact_capsule``. Returns (obj_contact, hand_contact) each (N,)."""
    if caps_on_hand:
        sdf_obj, dot_obj = capsule_sdf(
            hand_verts, hand_normals, object_verts, object_normals,
            caps_rad=caps_rad, caps_top=caps_top, caps_bot=caps_bot, foreach_on_mesh=False,
        )
        sdf_hand, dot_hand = capsule_sdf(
            hand_verts, hand_normals, object_verts, object_normals,
            caps_rad=caps_rad, caps_top=caps_top, caps_bot=caps_bot, foreach_on_mesh=True,
        )
    else:
        sdf_obj, dot_obj = capsule_sdf(
            object_verts, object_normals, hand_verts, hand_normals,
            caps_rad=caps_rad, caps_top=caps_top, caps_bot=caps_bot, foreach_on_mesh=True,
        )
        sdf_hand, dot_hand = capsule_sdf(
            object_verts, object_normals, hand_verts, hand_normals,
            caps_rad=caps_rad, caps_top=caps_top, caps_bot=caps_bot, foreach_on_mesh=False,
        )
    obj_contact = sdf_to_contact(sdf_obj, dot_obj, method=contact_norm_method)
    hand_contact = sdf_to_contact(sdf_hand, dot_hand, method=contact_norm_method)
    return obj_contact, hand_contact


# Alias used by integration code
capsule_contact_on_object = calculate_contact_capsule
