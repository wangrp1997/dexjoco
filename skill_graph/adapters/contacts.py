"""MuJoCo hand–object contacts (privileged sim)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from skill_graph.constants import LEFT_HAND_ROOT, PEG_BODY, RIGHT_HAND_ROOT, TRAY_BODY, ObjectName, Side

if TYPE_CHECKING:
    from skill_graph.adapters.assembly import AssemblySim


def _object_body(object_name: ObjectName) -> str:
    return TRAY_BODY if object_name == "tray" else PEG_BODY


def _hand_root(side: Side) -> str:
    return LEFT_HAND_ROOT if side == "left" else RIGHT_HAND_ROOT


def _subtree_geom_ids(model, body_id: int) -> set[int]:
    bodies = {int(body_id)}
    changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            if int(model.body_parentid[b]) in bodies and b not in bodies:
                bodies.add(b)
                changed = True
    geoms: set[int] = set()
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) in bodies:
            geoms.add(g)
    return geoms


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


@dataclass(frozen=True)
class ContactRecord:
    pos_world: np.ndarray
    normal_world: np.ndarray
    force_world: np.ndarray
    dist_m: float
    hand_body: str
    object_body: str


def hand_object_contacts(
    sim: "AssemblySim",
    *,
    side: Side,
    object_name: ObjectName,
    contact_dist_m: float = 0.002,
) -> list[ContactRecord]:
    model = sim.model
    data = sim.data
    obj_id = int(model.body(_object_body(object_name)).id)
    hand_id = int(model.body(_hand_root(side)).id)
    obj_geoms = _subtree_geom_ids(model, obj_id)
    hand_geoms = _subtree_geom_ids(model, hand_id)
    out: list[ContactRecord] = []
    force_buf = np.zeros(6, dtype=np.float64)

    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        ho = (g1 in obj_geoms and g2 in hand_geoms) or (g2 in obj_geoms and g1 in hand_geoms)
        if not ho:
            continue
        frame = np.asarray(c.frame, dtype=np.float64).reshape(9)
        n01 = frame[0:3]
        if g1 in hand_geoms and g2 in obj_geoms:
            normal = _normalize(n01)
            hand_body = model.body(model.geom_bodyid[g1]).name
            object_body = model.body(model.geom_bodyid[g2]).name
        else:
            normal = _normalize(-n01)
            hand_body = model.body(model.geom_bodyid[g2]).name
            object_body = model.body(model.geom_bodyid[g1]).name
        d = float(c.dist)
        if d > contact_dist_m:
            continue
        pos = np.asarray(c.pos, dtype=np.float64) - d * normal
        mujoco.mj_contactForce(model, data, i, force_buf)
        force = force_buf[:3].copy()
        out.append(
            ContactRecord(
                pos_world=pos,
                normal_world=normal,
                force_world=force,
                dist_m=d,
                hand_body=hand_body,
                object_body=object_body,
            )
        )
    return out
