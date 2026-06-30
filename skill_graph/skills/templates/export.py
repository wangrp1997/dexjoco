"""Export object-frame grasp templates from sidecar demos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from skill_graph.adapters.assembly import make_assembly_sim
from skill_graph.adapters.contacts import hand_object_contacts
from skill_graph.adapters.control import read_arm23
from skill_graph.constants import APPROACH_LOOKBACK, MAX_APPROACH_WAYPOINTS, ObjectName, Side
from skill_graph.io.actions import zarr_to_raw_dict
from skill_graph.io.zarr import load_zarr_episode
from skill_graph.math.se3 import relative_mocap_in_object_frame, world_to_object
from skill_graph.skills.templates.bank import save_template
from skill_graph.skills.templates.schema import GraspTemplate


def _side_for_object(object_name: ObjectName) -> Side:
    return "left" if object_name == "tray" else "right"


def _grasp_frame(timing: dict, object_name: ObjectName) -> int:
    key = "left_grasp_frame" if object_name == "tray" else "right_grasp_frame"
    return int(timing[key])


def _squeeze_frame(timing: dict, object_name: ObjectName, n_actions: int) -> int:
    grasp = _grasp_frame(timing, object_name)
    lift_key = "tray_lift_start" if object_name == "tray" else "peg_lift_start"
    lift = timing.get(lift_key)
    if lift is None:
        return int(np.clip(grasp + 8, 0, n_actions - 1))
    return int(np.clip(min(grasp + 8, int(lift) - 1), grasp, n_actions - 1))


def _subsample(start: int, end: int, max_pts: int) -> np.ndarray:
    idx = np.arange(int(start), int(end) + 1, dtype=int)
    if idx.size <= max_pts:
        return idx
    return np.unique(np.linspace(start, end, max_pts, dtype=int))


def _record_contacts_obj(sim, side: Side, object_name: ObjectName) -> tuple[np.ndarray, ...]:
    obj_pos, obj_quat = sim.object_pose(object_name)
    contacts = hand_object_contacts(sim, side=side, object_name=object_name)
    if not contacts:
        z = np.zeros((0, 3), dtype=np.float64)
        return z, z, z, (), ()
    pos_obj = world_to_object(np.stack([c.pos_world for c in contacts], axis=0), obj_pos, obj_quat)
    normal_obj = world_to_object(np.stack([c.normal_world for c in contacts], axis=0), obj_pos, obj_quat)
    nrm = np.linalg.norm(normal_obj, axis=1, keepdims=True)
    normal_obj = normal_obj / np.maximum(nrm, 1e-8)
    force_obj = world_to_object(np.stack([c.force_world for c in contacts], axis=0), obj_pos, obj_quat)
    return (
        pos_obj,
        normal_obj,
        force_obj,
        tuple(c.hand_body for c in contacts),
        tuple(c.object_body for c in contacts),
    )


def export_template_from_entry(
    entry: dict[str, Any],
    *,
    object_name: ObjectName,
    seed: int = 0,
    bank_root: Path | None = None,
) -> GraspTemplate | None:
    timing = entry["timing"]
    actions, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    n = len(actions)
    side = _side_for_object(object_name)
    grasp_f = int(np.clip(_grasp_frame(timing, object_name), 0, n - 1))
    squeeze_f = _squeeze_frame(timing, object_name, n)
    start_f = max(0, grasp_f - APPROACH_LOOKBACK)
    approach_frames = set(_subsample(start_f, grasp_f, MAX_APPROACH_WAYPOINTS).tolist())
    approach_frames.update({grasp_f, squeeze_f, 0})

    end_replay = max(grasp_f, squeeze_f)
    recorded: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    demo_obj_pos: np.ndarray | None = None
    demo_obj_quat: np.ndarray | None = None

    sim = make_assembly_sim(seed=int(seed) + int(entry["episode_index"]))
    try:
        sim.env.reset()
        sim.restore_initial_state(initial_state)
        for fi, action in enumerate(actions[: end_replay + 1]):
            sim.raw.step(zarr_to_raw_dict(action))
            sim.on_physics_step()
            if fi not in approach_frames:
                continue
            arm = read_arm23(sim, side)
            obj_pos, obj_quat = sim.object_pose(object_name)
            pos_obj, quat_obj = relative_mocap_in_object_frame(arm[:3], arm[3:7], obj_pos, obj_quat)
            recorded[fi] = (pos_obj, quat_obj, arm[7:23].copy())
            if fi == 0:
                demo_obj_pos, demo_obj_quat = obj_pos.copy(), obj_quat.copy()

        if grasp_f not in recorded or squeeze_f not in recorded:
            return None

        g_pos, g_quat, g_hand = recorded[grasp_f]
        s_pos, s_quat, s_hand = recorded[squeeze_f]
        pos_lst, quat_lst, hand_lst = [], [], []
        for fi in sorted(f for f in approach_frames if f < grasp_f and f in recorded):
            p, q, h = recorded[fi]
            pos_lst.append(p)
            quat_lst.append(q)
            hand_lst.append(h)
        if not pos_lst:
            pos_lst, quat_lst, hand_lst = [g_pos], [g_quat], [g_hand]

        sim.restore_initial_state(initial_state)
        for fi, action in enumerate(actions[: squeeze_f + 1]):
            sim.raw.step(zarr_to_raw_dict(action))
        c_pos, c_n, c_f, c_h, c_o = _record_contacts_obj(sim, side, object_name)

        template = GraspTemplate(
            episode_index=int(entry["episode_index"]),
            side=side,
            object_name=object_name,
            demo_obj_pos=demo_obj_pos if demo_obj_pos is not None else sim.object_pose(object_name)[0],
            demo_obj_quat=demo_obj_quat if demo_obj_quat is not None else sim.object_pose(object_name)[1],
            grasp_mocap_pos_obj=g_pos,
            grasp_mocap_quat_obj=g_quat,
            grasp_hand=g_hand,
            squeeze_mocap_pos_obj=s_pos,
            squeeze_mocap_quat_obj=s_quat,
            squeeze_hand=s_hand,
            approach_mocap_pos_obj=np.stack(pos_lst, axis=0),
            approach_mocap_quat_obj=np.stack(quat_lst, axis=0),
            approach_hand=np.stack(hand_lst, axis=0),
            contact_pos_obj=c_pos,
            contact_normal_obj=c_n,
            contact_force_obj=c_f,
            contact_hand_bodies=c_h,
            contact_object_bodies=c_o,
            export_contact_count=int(c_pos.shape[0]),
            zarr_path=str(entry["zarr_path"]),
        )
        save_template(template, bank_root)
        return template
    finally:
        sim.close()


def export_all_from_manifest(
    entries: list[dict[str, Any]],
    *,
    seed: int = 0,
    bank_root: Path | None = None,
    max_episodes: int | None = None,
) -> list[GraspTemplate]:
    out: list[GraspTemplate] = []
    for i, entry in enumerate(entries):
        if max_episodes is not None and i >= max_episodes:
            break
        ep = int(entry["episode_index"])
        for obj in ("tray", "peg"):
            print(f"export ep{ep} {obj} ...", flush=True)
            t = export_template_from_entry(entry, object_name=obj, seed=seed, bank_root=bank_root)  # type: ignore[arg-type]
            if t is not None:
                out.append(t)
    return out
