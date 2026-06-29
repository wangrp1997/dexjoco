"""Per-demo δ* from episode sidecar (one-demo learning, L1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from interaction_retarget.constants import PEG_BODY, TRAY_BODY
from interaction_retarget.grasp.distill import _cluster_contact_sites
from interaction_retarget.io.npz import adjacency_from_padded, load_interaction_npz
from interaction_retarget.sim.settle import read_arm_action
from interaction_retarget.sim.state import restore_sim, snapshot_sim
from interaction_retarget.skill_replay.demo_grasp import apply_demo_grasp_frame
from interaction_retarget.transforms import relative_mocap_in_object_frame

ObjectName = Literal["tray", "peg"]


def _episode_sidecar_npz(sidecar_dir: Path, entry: dict[str, Any]) -> Path:
    ep_idx = int(entry["episode_index"])
    npz_path = Path(entry["npz_path"]) if entry.get("npz_path") else None
    if npz_path is None or not npz_path.is_file():
        npz_path = sidecar_dir / f"episode_{ep_idx:03d}" / "interaction_sidecar.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing episode sidecar: {npz_path}")
    return npz_path


def load_demo_canonical_grasp(
    sidecar_dir: Path,
    entry: dict[str, Any],
    object_name: str,
    *,
    seed_base: int = 0,
) -> dict[str, np.ndarray | list[list[int]] | str | None | int]:
    """Topology/contact δ* from sidecar; mocap filled via ``enrich_canonical_on_env``."""
    del seed_base
    sidecar_dir = Path(sidecar_dir)
    npz_path = _episode_sidecar_npz(sidecar_dir, entry)
    snap = load_interaction_npz(npz_path, prefix=object_name)
    data = np.load(npz_path)
    prefix = object_name
    contacts = np.asarray(snap["contact_centers_obj"], dtype=np.float64)
    contact_sites = _cluster_contact_sites(contacts)
    return {
        "object_name": object_name,
        "hand_side": "left" if object_name == "tray" else "right",
        "hand_points_obj": np.asarray(snap["hand_obj"], dtype=np.float64),
        "object_samples_obj": np.asarray(snap["object_samples_obj"], dtype=np.float64),
        "interaction_vertices_obj": np.asarray(
            data[f"{prefix}_interaction_vertices_obj"], dtype=np.float64
        ),
        "laplacian_coords": np.asarray(data[f"{prefix}_laplacian_coords"], dtype=np.float64),
        "adjacency": adjacency_from_padded(data[f"{prefix}_adjacency"]),
        "hand_points_std": np.zeros_like(snap["hand_obj"], dtype=np.float64),
        "hand_joint_median": np.zeros(16, dtype=np.float64),
        "mocap_pos_obj": np.zeros(3, dtype=np.float64),
        "mocap_quat_obj": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "contact_sites_obj": contact_sites,
        "source_episode_index": int(entry["episode_index"]),
    }


def enrich_canonical_on_env(
    raw_env,
    entry: dict[str, Any],
    canonical: dict[str, Any],
    object_name: ObjectName,
) -> dict[str, Any]:
    """Set mocap/hand joint at grasp frame on the live env (no extra EGL context)."""
    side: Literal["left", "right"] = "left" if object_name == "tray" else "right"
    body = TRAY_BODY if object_name == "tray" else PEG_BODY
    snap = snapshot_sim(raw_env)
    try:
        apply_demo_grasp_frame(raw_env, entry, object_name)
        arm23 = read_arm_action(raw_env, side)
        obj_id = int(raw_env._model.body(body).id)
        obj_pos = np.asarray(raw_env._data.xpos[obj_id], dtype=np.float64)
        obj_quat = np.asarray(raw_env._data.xquat[obj_id], dtype=np.float64)
        pos_obj, quat_obj = relative_mocap_in_object_frame(
            arm23[0:3], arm23[3:7], obj_pos, obj_quat
        )
        canonical["mocap_pos_obj"] = np.asarray(pos_obj, dtype=np.float64)
        canonical["mocap_quat_obj"] = np.asarray(quat_obj, dtype=np.float64)
        canonical["hand_joint_median"] = np.asarray(arm23[7:23], dtype=np.float64)
    finally:
        restore_sim(raw_env, snap)
    return canonical
