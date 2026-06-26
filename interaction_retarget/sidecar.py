"""Build per-episode interaction sidecar (δ* ingredients)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from interaction_retarget.constants import (
    CONTACT_SAMPLE_SIGMA_M,
    INDUSTREAL_MESH_SCALE,
    LEFT_HAND_BODIES,
    NUM_HAND_KEYPOINTS,
    NUM_INTERACTION_VERTICES,
    NUM_OBJECT_SAMPLES,
    PEG_BODY,
    PEG_MESH_PATH,
    RIGHT_HAND_BODIES,
    TRAY_BODY,
    TRAY_MESH_PATH,
)
from interaction_retarget.grasp_timing import GraspTiming, detect_grasp_timing
from interaction_retarget.laplacian import create_interaction_adjacency, laplacian_coordinates
from interaction_retarget.mesh_sampling import contact_weighted_surface_sampling, load_object_mesh
from interaction_retarget.replay import ReplayTrace, ReplayStep
from interaction_retarget.transforms import world_to_object

@dataclass
class ObjectInteractionSnapshot:
    object_name: str
    grasp_frame: int
    lift_start: int | None
    hand_points_obj: np.ndarray
    contact_centers_obj: np.ndarray
    object_samples_obj: np.ndarray
    interaction_vertices_obj: np.ndarray
    laplacian_coords: np.ndarray
    adjacency: list[list[int]]


@dataclass
class EpisodeSidecar:
    episode_index: int
    zarr_path: str
    num_steps: int
    timing: GraspTiming
    tray: ObjectInteractionSnapshot | None
    peg: ObjectInteractionSnapshot | None
    replay_info: dict[str, Any]


_MESH_CACHE: dict[str, trimesh.Trimesh] = {}


def _get_mesh(path: Path) -> trimesh.Trimesh:
    key = str(path)
    if key not in _MESH_CACHE:
        _MESH_CACHE[key] = load_object_mesh(path, scale=INDUSTREAL_MESH_SCALE)
    return _MESH_CACHE[key]


def timing_from_trace(trace: ReplayTrace) -> GraspTiming:
    steps = trace.steps
    tray_contact = np.asarray([s.contact.tray_contact for s in steps], dtype=bool)
    peg_contact = np.asarray([s.contact.peg_contact for s in steps], dtype=bool)
    tray_counts = np.asarray([s.contact.tray_contact_count for s in steps], dtype=np.int32)
    peg_counts = np.asarray([s.contact.peg_contact_count for s in steps], dtype=np.int32)
    left_grip = np.asarray([s.left_gripper_speed for s in steps], dtype=np.float64)
    right_grip = np.asarray([s.right_gripper_speed for s in steps], dtype=np.float64)
    tray_z = np.asarray([s.tray_z for s in steps], dtype=np.float64)
    peg_z = np.asarray([s.peg_z for s in steps], dtype=np.float64)
    tray_rest_z = float(tray_z[0])
    peg_rest_z = float(peg_z[0])
    return detect_grasp_timing(
        tray_contact=tray_contact,
        peg_contact=peg_contact,
        tray_contact_count=tray_counts,
        peg_contact_count=peg_counts,
        left_gripper_speed=left_grip,
        right_gripper_speed=right_grip,
        tray_z=tray_z,
        peg_z=peg_z,
        tray_rest_z=tray_rest_z,
        peg_rest_z=peg_rest_z,
    )


def _snap_contacts_to_mesh(mesh: trimesh.Trimesh, points_obj: np.ndarray) -> np.ndarray:
    points = np.asarray(points_obj, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return points
    try:
        closest, _, _ = trimesh.proximity.closest_point(mesh, points)
        return np.asarray(closest, dtype=np.float64)
    except (ImportError, ModuleNotFoundError):
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        d2 = ((points[:, None, :] - verts[None, :, :]) ** 2).sum(axis=2)
        return verts[np.argmin(d2, axis=1)]


def _contact_centers_obj(step: ReplayStep, *, object_name: str, mesh: trimesh.Trimesh) -> np.ndarray:
    if object_name == "tray":
        pos_w = step.contact.tray_contact_pos_world
        obj_pos, obj_quat = step.tray_pos, step.tray_quat
    elif object_name == "peg":
        pos_w = step.contact.peg_contact_pos_world
        obj_pos, obj_quat = step.peg_pos, step.peg_quat
    else:
        raise ValueError(object_name)
    if pos_w.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    obj_pts = world_to_object(pos_w, obj_pos, obj_quat)
    return _snap_contacts_to_mesh(mesh, obj_pts)


def _snapshot_for_object(
    trace: ReplayTrace,
    *,
    object_name: str,
    grasp_frame: int,
    lift_start: int | None,
    hand_world_fn,
    mesh_path: Path,
    seed: int,
) -> ObjectInteractionSnapshot:
    step = trace.steps[grasp_frame]
    if object_name == "tray":
        obj_pos, obj_quat = step.tray_pos, step.tray_quat
    else:
        obj_pos, obj_quat = step.peg_pos, step.peg_quat

    hand_world = hand_world_fn(step)
    hand_obj = world_to_object(hand_world, obj_pos, obj_quat)
    mesh = _get_mesh(mesh_path)
    contact_centers = _contact_centers_obj(step, object_name=object_name, mesh=mesh)
    object_samples = contact_weighted_surface_sampling(
        mesh,
        contact_centers,
        NUM_OBJECT_SAMPLES,
        sigma=CONTACT_SAMPLE_SIGMA_M,
        seed=seed,
    )
    vertices = np.concatenate([hand_obj, object_samples], axis=0)
    adjacency = create_interaction_adjacency(vertices)
    laplacian = laplacian_coordinates(vertices, adjacency)

    return ObjectInteractionSnapshot(
        object_name=object_name,
        grasp_frame=int(grasp_frame),
        lift_start=lift_start,
        hand_points_obj=hand_obj,
        contact_centers_obj=contact_centers,
        object_samples_obj=object_samples,
        interaction_vertices_obj=vertices,
        laplacian_coords=laplacian,
        adjacency=adjacency,
    )


def build_episode_sidecar(
    trace: ReplayTrace,
    *,
    episode_index: int,
    zarr_path: Path,
    seed: int = 0,
) -> EpisodeSidecar:
    timing = timing_from_trace(trace)
    tray_snap = None
    peg_snap = None

    if timing.left_grasp_frame is not None:
        tray_snap = _snapshot_for_object(
            trace,
            object_name="tray",
            grasp_frame=timing.left_grasp_frame,
            lift_start=timing.tray_lift_start,
            hand_world_fn=lambda s: s.left_hand_world,
            mesh_path=TRAY_MESH_PATH,
            seed=seed + episode_index * 2,
        )
    if timing.right_grasp_frame is not None:
        peg_snap = _snapshot_for_object(
            trace,
            object_name="peg",
            grasp_frame=timing.right_grasp_frame,
            lift_start=timing.peg_lift_start,
            hand_world_fn=lambda s: s.right_hand_world,
            mesh_path=PEG_MESH_PATH,
            seed=seed + episode_index * 2 + 1,
        )

    return EpisodeSidecar(
        episode_index=int(episode_index),
        zarr_path=str(zarr_path),
        num_steps=len(trace.steps),
        timing=timing,
        tray=tray_snap,
        peg=peg_snap,
        replay_info=dict(trace.info),
    )


def _snapshot_to_npz_group(prefix: str, snap: ObjectInteractionSnapshot | None) -> dict[str, np.ndarray]:
    if snap is None:
        return {}
    adj_array = np.asarray([len(n) for n in snap.adjacency], dtype=np.int32)
    # Store adjacency as padded matrix for simple IO
    max_deg = max((len(n) for n in snap.adjacency), default=0)
    adj_mat = -np.ones((len(snap.adjacency), max_deg), dtype=np.int32)
    for i, neighbors in enumerate(snap.adjacency):
        adj_mat[i, : len(neighbors)] = neighbors
    return {
        f"{prefix}/grasp_frame": np.asarray([snap.grasp_frame], dtype=np.int32),
        f"{prefix}/lift_start": np.asarray([-1 if snap.lift_start is None else snap.lift_start], dtype=np.int32),
        f"{prefix}/hand_points_obj": snap.hand_points_obj,
        f"{prefix}/contact_centers_obj": snap.contact_centers_obj,
        f"{prefix}/object_samples_obj": snap.object_samples_obj,
        f"{prefix}/interaction_vertices_obj": snap.interaction_vertices_obj,
        f"{prefix}/laplacian_coords": snap.laplacian_coords,
        f"{prefix}/adjacency_degrees": adj_array,
        f"{prefix}/adjacency": adj_mat,
    }


def save_episode_sidecar(sidecar: EpisodeSidecar, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_dir = out_dir / f"episode_{sidecar.episode_index:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "episode_index": np.asarray([sidecar.episode_index], dtype=np.int32),
        "num_steps": np.asarray([sidecar.num_steps], dtype=np.int32),
        "left_grasp_frame": np.asarray(
            [-1 if sidecar.timing.left_grasp_frame is None else sidecar.timing.left_grasp_frame],
            dtype=np.int32,
        ),
        "right_grasp_frame": np.asarray(
            [-1 if sidecar.timing.right_grasp_frame is None else sidecar.timing.right_grasp_frame],
            dtype=np.int32,
        ),
        "tray_lift_start": np.asarray(
            [-1 if sidecar.timing.tray_lift_start is None else sidecar.timing.tray_lift_start],
            dtype=np.int32,
        ),
        "peg_lift_start": np.asarray(
            [-1 if sidecar.timing.peg_lift_start is None else sidecar.timing.peg_lift_start],
            dtype=np.int32,
        ),
    }
    for k, v in _snapshot_to_npz_group("tray", sidecar.tray).items():
        arrays[k.replace("/", "_")] = v
    for k, v in _snapshot_to_npz_group("peg", sidecar.peg).items():
        arrays[k.replace("/", "_")] = v

    npz_path = ep_dir / "interaction_sidecar.npz"
    np.savez_compressed(npz_path, **arrays)

    meta = {
        "episode_index": sidecar.episode_index,
        "zarr_path": sidecar.zarr_path,
        "num_steps": sidecar.num_steps,
        "timing": asdict(sidecar.timing),
        "has_tray": sidecar.tray is not None,
        "has_peg": sidecar.peg is not None,
        "replay_info": sidecar.replay_info,
        "num_hand_keypoints": NUM_HAND_KEYPOINTS,
        "num_object_samples": NUM_OBJECT_SAMPLES,
        "num_interaction_vertices": NUM_INTERACTION_VERTICES,
        "left_hand_body_names": list(LEFT_HAND_BODIES),
        "right_hand_body_names": list(RIGHT_HAND_BODIES),
        "tray_body": TRAY_BODY,
        "peg_body": PEG_BODY,
    }
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npz_path


def load_episode_sidecar_npz(npz_path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(npz_path))
