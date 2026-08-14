"""Generic contact/insertion helpers driven by GeometryFamilySpec (parallel smoke only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from embodied_grasp_insertion.geometry.family_spec import GeometryFamilySpec


@dataclass
class NamedIds:
    peg_body: int
    peg_joint: int
    peg_tip: int
    peg_grasp: int
    peg_collision: int
    socket_body: int
    socket_joint: int
    socket_site: int
    insert_bottom: int


def lookup_ids(model: mujoco.MjModel, spec: GeometryFamilySpec) -> NamedIds:
    """Resolve bodies/sites/geoms by unified semantic names — no hardcoded 8mm strings."""
    return NamedIds(
        peg_body=int(model.body(spec.peg_body).id),
        peg_joint=int(model.joint("peg_joint").id),
        peg_tip=int(model.site(spec.peg_tip_site).id),
        peg_grasp=int(model.site(spec.peg_grasp_site).id),
        peg_collision=int(model.geom(spec.peg_collision_geom).id),
        socket_body=int(model.body(spec.socket_body).id),
        socket_joint=int(model.joint("socket_joint").id),
        socket_site=int(model.site(spec.socket_site).id),
        insert_bottom=int(model.geom(spec.insert_bottom_geom).id),
    )


def freejoint_qpos_adr(model: mujoco.MjModel, joint_id: int) -> int:
    return int(model.jnt_qposadr[joint_id])


def set_free_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_id: int,
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
) -> None:
    adr = freejoint_qpos_adr(model, joint_id)
    data.qpos[adr : adr + 3] = np.asarray(pos, dtype=np.float64)
    data.qpos[adr + 3 : adr + 7] = np.asarray(quat_wxyz, dtype=np.float64)


def tip_to_socket_site(data: mujoco.MjData, ids: NamedIds) -> np.ndarray:
    return np.asarray(data.site_xpos[ids.peg_tip], dtype=np.float64) - np.asarray(
        data.site_xpos[ids.socket_site], dtype=np.float64
    )


def count_contacts_involving(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> int:
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if int(c.geom1) == geom_id or int(c.geom2) == geom_id:
            n += 1
    return n


def any_penetrating_contact(data: mujoco.MjData, depth_thresh: float = -1e-5) -> bool:
    """True if any contact has substantial penetration (negative dist beyond thresh)."""
    for i in range(data.ncon):
        if float(data.contact[i].dist) < depth_thresh:
            return True
    return False


def settle(model: mujoco.MjModel, data: mujoco.MjData, steps: int = 50) -> None:
    for _ in range(int(steps)):
        mujoco.mj_step(model, data)


@dataclass
class InsertionProbeResult:
    lateral_offset_m: float
    depth_m: float
    tip_site_delta: list[float]
    ncon: int
    peg_collision_contacts: int
    bottom_contacts: int
    penetrating: bool
    notes: str = ""


def probe_insertion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: GeometryFamilySpec,
    *,
    lateral_offset_m: float,
    depth_m: float,
    settle_steps: int = 30,
) -> InsertionProbeResult:
    """Place peg above/into hole along socket site +z with optional lateral offset; short settle."""
    ids = lookup_ids(model, spec)
    # Freeze socket near origin.
    set_free_pose(model, data, ids.socket_joint, np.array([0.0, 0.0, 0.05]), np.array([1, 0, 0, 0.0]))
    mujoco.mj_forward(model, data)
    sock = np.asarray(data.site_xpos[ids.socket_site], dtype=np.float64).copy()
    # Align peg tip to socket site + offset - depth along world z (sites defined +z in body).
    tip_target = sock + np.array([lateral_offset_m, 0.0, -depth_m])
    # Peg tip site local z = collision tip_z; place body so tip reaches tip_target.
    tip_local_z = float(spec.collision["peg_tip_site_z_m"])
    peg_pos = tip_target - np.array([0.0, 0.0, tip_local_z])
    set_free_pose(model, data, ids.peg_joint, peg_pos, np.array([1, 0, 0, 0.0]))
    mujoco.mj_forward(model, data)
    settle(model, data, settle_steps)
    delta = tip_to_socket_site(data, ids)
    return InsertionProbeResult(
        lateral_offset_m=float(lateral_offset_m),
        depth_m=float(depth_m),
        tip_site_delta=delta.tolist(),
        ncon=int(data.ncon),
        peg_collision_contacts=count_contacts_involving(model, data, ids.peg_collision),
        bottom_contacts=count_contacts_involving(model, data, ids.insert_bottom),
        penetrating=any_penetrating_contact(data),
    )


def labeler_design_notes() -> dict[str, Any]:
    return {
        "hardcoded_8mm_forbidden": True,
        "lookup": "GeometryFamilySpec semantic names via lookup_ids()",
        "adapter": "embodied_grasp_insertion.geometry.labeler_adapter",
        "parallel_only": "Does not modify hybrid_insert.assembly_contacts or official arena",
        "fields": [
            "peg_body",
            "peg_tip_site",
            "peg_grasp_site",
            "socket_body",
            "socket_site",
            "insert_bottom_geom",
        ],
    }
