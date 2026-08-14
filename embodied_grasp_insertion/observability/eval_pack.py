"""P0-Obs-D1 evaluation pack schema, slicing, and validation (no MuJoCo)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from embodied_grasp_insertion.labels.privileged_schema import (
    EXCLUDED_FIELDS,
    SCHEMA_VERSION,
)
from embodied_grasp_insertion.observability.feasibility import (
    HORIZONS,
    SPLIT_SEED,
    atomic_episode_split,
    check_split_leakage,
    digest_obj,
)

PROTOCOL = "P0-Obs-D1"
PACK_NAME = "observability_eval_v1"
PRIMARY_H = 8
STORE_H = 16
FORBIDDEN_DEPLOY_KEYS = (
    "peg7",
    "tray7",
    "lat_vec3",
    "along_tip_axis",
    "hole_axis3",
    "peg_axis3",
    "flags3",
    "tip",
    "tip_dist",
)


def slice_view(arr: np.ndarray, h: int) -> np.ndarray:
    """H1/H4/H8/H16 views into leading frames of stored H16 history."""
    if h not in HORIZONS:
        raise ValueError(f"unsupported H={h}")
    if arr.shape[0] < h:
        raise ValueError(f"array length {arr.shape[0]} < H={h}")
    return arr[:h]


def sample_meta(
    *,
    episode_index: int,
    root_id: str,
    root_phase: str,
    root_frame: int,
    split: str,
    geometry_family_id: str,
    target_instance_id: str,
    socket_site: str,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "pack_name": PACK_NAME,
        "evaluation_only": True,
        "training_authorized": False,
        "single_geometry_only": True,
        "claims_observability_p0_pass": False,
        "primary_horizon": PRIMARY_H,
        "stored_horizon": STORE_H,
        "schema_version": schema_version,
        "episode_index": int(episode_index),
        "root_id": str(root_id),
        "root_phase": str(root_phase),
        "root_frame": int(root_frame),
        "split": str(split),
        "geometry_family_id": str(geometry_family_id),
        "target_instance_id": str(target_instance_id),
        "socket_site": str(socket_site),
        "student_inputs": ["A_act44", "B_act44_ft12"],
        "forbidden_deploy_inputs": list(FORBIDDEN_DEPLOY_KEYS),
        "excluded_labels": list(EXCLUDED_FIELDS),
    }


def validate_sample_arrays(data: dict[str, np.ndarray], meta: dict[str, Any]) -> list[str]:
    """Return list of issues (empty = ok)."""
    issues: list[str] = []
    t = int(data["frames"].shape[0])
    if t != STORE_H:
        issues.append(f"stored_len!={STORE_H}:{t}")
    frames = np.asarray(data["frames"], dtype=np.int64)
    if frames.tolist() != list(range(int(frames[0]), int(frames[0]) + t)):
        issues.append("frames_not_contiguous")
    for key, shape in (
        ("act44", (STORE_H, 44)),
        ("ft12", (STORE_H, 12)),
        ("o2h_translation_m", (STORE_H, 3)),
        ("o2h_rotvec_rad", (STORE_H, 3)),
        ("finger_force_norm_N", (STORE_H, 4)),
        ("contact_active", (STORE_H, 4)),
    ):
        a = np.asarray(data[key])
        if a.shape != shape:
            issues.append(f"{key}_shape:{a.shape}")
        if key in ("act44", "ft12", "o2h_translation_m", "o2h_rotvec_rad", "finger_force_norm_N"):
            if not np.isfinite(a).all():
                issues.append(f"{key}_nonfinite")
    # velocity: first unavailable
    avail = np.asarray(data["o2h_vel_available"], dtype=bool)
    if avail.shape != (STORE_H,) or bool(avail[0]) is not False:
        issues.append("vel_first_must_be_unavailable")
    if not bool(np.all(avail[1:])):
        issues.append("vel_rest_must_be_available")
    for bad in FORBIDDEN_DEPLOY_KEYS:
        if bad in data or bad in meta:
            issues.append(f"forbidden_deploy:{bad}")
    for bad in EXCLUDED_FIELDS:
        if bad in data:
            issues.append(f"excluded_label:{bad}")
    if meta.get("evaluation_only") is not True:
        issues.append("evaluation_only_false")
    if meta.get("training_authorized") is not False:
        issues.append("training_authorized_not_false")
    if meta.get("claims_observability_p0_pass") is not False:
        issues.append("claims_obs_p0_true")
    # primary H=8 slice
    try:
        a8 = slice_view(data["act44"], PRIMARY_H)
        if a8.shape != (PRIMARY_H, 44):
            issues.append("primary_slice_bad")
    except Exception as e:
        issues.append(f"primary_slice_err:{e}")
    return issues


def pack_banner() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "pack_name": PACK_NAME,
        "evaluation_only": True,
        "training_authorized": False,
        "single_geometry_only": True,
        "claims_observability_p0_pass": False,
        "WRITE_IMPLEMENTATION_ENABLED": False,
        "primary_horizon": PRIMARY_H,
        "stored_horizon": STORE_H,
        "schema_version": SCHEMA_VERSION,
        "split_seed": SPLIT_SEED,
        "note": "Not a training dataset. Not Observability P0 pass.",
    }


def split_digest(episode_split: dict[str, list[int]]) -> str:
    return digest_obj({k: list(v) for k, v in sorted(episode_split.items())})


def build_fixed_split(episode_ids: list[int] | None = None) -> dict[str, Any]:
    ids = list(range(100)) if episode_ids is None else list(episode_ids)
    ep_split = atomic_episode_split(ids, seed=SPLIT_SEED)
    return {
        "method": "sha256(seed:episode) ranked; episode-atomic 70/15/15",
        "seed": SPLIT_SEED,
        "episodes": ep_split,
        "digest": split_digest(ep_split),
        "counts": {k: len(v) for k, v in ep_split.items()},
    }
