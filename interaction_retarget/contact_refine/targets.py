"""Demo privileged contact targets (ContactOpt GT + GraspTTA cmap)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from interaction_retarget.constants import INDUSTREAL_MESH_SCALE, PEG_MESH_PATH, TRAY_MESH_PATH
from interaction_retarget.contact_refine.config import ContactCapsConfig, DEFAULT_CAPS
from interaction_retarget.contactopt.diffcontact import calculate_contact_capsule
from interaction_retarget.mesh.sampling import load_object_mesh


def _mesh_path(object_name: str) -> Path:
    return TRAY_MESH_PATH if object_name == "tray" else PEG_MESH_PATH


def object_normals_at_points(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Face normals at closest points on mesh (object frame)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    _, _, face_idx = trimesh.proximity.closest_point(mesh, pts)
    face_idx = np.clip(np.asarray(face_idx, dtype=np.int64), 0, len(mesh.face_normals) - 1)
    return np.asarray(mesh.face_normals[face_idx], dtype=np.float64)


def demo_object_contact_cmap(
    object_samples_obj: np.ndarray,
    contact_centers_obj: np.ndarray,
    *,
    radius_m: float = 0.025,
) -> np.ndarray:
    """GraspTTA-style binary cmap: 1 on object samples near demo contact centers."""
    samples = np.asarray(object_samples_obj, dtype=np.float64).reshape(-1, 3)
    centers = np.asarray(contact_centers_obj, dtype=np.float64).reshape(-1, 3)
    if samples.size == 0 or centers.size == 0:
        return np.zeros((samples.shape[0],), dtype=np.float64)
    cmap = np.zeros((samples.shape[0],), dtype=np.float64)
    for i, p in enumerate(samples):
        d = np.linalg.norm(centers - p, axis=1)
        cmap[i] = 1.0 if float(np.min(d)) <= float(radius_m) else 0.0
    return cmap


def demo_contact_targets_from_canonical(
    canonical: dict[str, Any],
    *,
    object_name: str,
    cmap_radius_m: float = 0.025,
) -> dict[str, np.ndarray]:
    """Build privileged targets for one demo δ* (sidecar / demo_grasp fields)."""
    obj_samples = np.asarray(canonical["object_samples_obj"], dtype=np.float64)
    centers = np.asarray(
        canonical.get("contact_centers_obj", canonical.get("contact_sites_obj", np.zeros((0, 3)))),
        dtype=np.float64,
    ).reshape(-1, 3)
    sites = np.asarray(canonical.get("contact_sites_obj", centers), dtype=np.float64).reshape(-1, 3)

    mesh = load_object_mesh(_mesh_path(object_name), scale=INDUSTREAL_MESH_SCALE)
    obj_normals = object_normals_at_points(mesh, obj_samples)

    # ContactOpt target: high contact on demo regions (soft map from cmap)
    cmap = demo_object_contact_cmap(obj_samples, centers, radius_m=cmap_radius_m)
    target_obj_contact = np.clip(cmap, 0.0, 1.0)

    # Hand target: uniform contact on fingertip regions (filled during sim from geometry)
    hand_pts = np.asarray(canonical["hand_points_obj"], dtype=np.float64)
    hand_normals = object_normals_at_points(mesh, hand_pts)  # proxy; refined in sim

    return {
        "object_samples_obj": obj_samples,
        "object_normals_obj": obj_normals,
        "contact_centers_obj": centers,
        "contact_sites_obj": sites,
        "object_cmap": cmap,
        "target_obj_contact": target_obj_contact,
        "hand_points_obj": hand_pts,
        "hand_normals_proxy_obj": hand_normals,
    }


def predicted_contact_maps_obj_frame(
    hand_obj: np.ndarray,
    hand_normals_obj: np.ndarray,
    targets: dict[str, np.ndarray],
    *,
    caps: ContactCapsConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Current ContactOpt capsule contact on demo object samples."""
    caps = caps or DEFAULT_CAPS
    obj_c, hand_c = calculate_contact_capsule(
        hand_obj,
        hand_normals_obj,
        targets["object_samples_obj"],
        targets["object_normals_obj"],
        caps_top=caps.caps_top,
        caps_bot=caps.caps_bot,
        caps_rad=caps.caps_rad,
        caps_on_hand=False,
        contact_norm_method=caps.contact_norm_method,
    )
    return obj_c, hand_c
