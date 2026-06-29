"""Hand–object contact distances from MuJoCo (DexGraspBench / Dexonomy style)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from interaction_retarget.constants import LEFT_HAND_ROOT, PEG_BODY, RIGHT_HAND_ROOT, TRAY_BODY

ObjectName = Literal["tray", "peg"]
Side = Literal["left", "right"]


def _object_body(object_name: ObjectName) -> str:
    return TRAY_BODY if object_name == "tray" else PEG_BODY


def _hand_root(side: Side) -> str:
    return LEFT_HAND_ROOT if side == "left" else RIGHT_HAND_ROOT


def _subtree_geom_ids(model, root_body_id: int) -> set[int]:
    bodies = {int(root_body_id)}
    changed = True
    while changed:
        changed = False
        for bid in range(model.nbody):
            parent = int(model.body_parentid[bid])
            if parent in bodies and bid not in bodies:
                bodies.add(bid)
                changed = True
    return {gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in bodies}


def hand_object_contact_dists(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
) -> tuple[np.ndarray, int]:
    """Return (distances, count) for hand–object geom pairs. dist<0 ⇒ penetration."""
    model = raw_env._model
    data = raw_env._data
    obj_id = int(model.body(_object_body(object_name)).id)
    hand_id = int(model.body(_hand_root(side)).id)
    obj_geoms = _subtree_geom_ids(model, obj_id)
    hand_geoms = _subtree_geom_ids(model, hand_id)
    dists: list[float] = []
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in obj_geoms and g2 in hand_geoms) or (g2 in obj_geoms and g1 in hand_geoms):
            dists.append(float(c.dist))
    arr = np.asarray(dists, dtype=np.float64) if dists else np.zeros(0, dtype=np.float64)
    return arr, len(dists)


def max_penetration_m(raw_env, *, side: Side, object_name: ObjectName) -> float:
    dists, _ = hand_object_contact_dists(raw_env, side=side, object_name=object_name)
    if dists.size == 0:
        return 0.0
    return float(max(0.0, -np.min(dists)))


def hand_hand_contact_dists(raw_env, *, side: Side) -> np.ndarray:
    """Hand–hand geom contact distances (Dexonomy hh_contact)."""
    model = raw_env._model
    data = raw_env._data
    left_id = int(model.body(LEFT_HAND_ROOT).id)
    right_id = int(model.body(RIGHT_HAND_ROOT).id)
    left_geoms = _subtree_geom_ids(model, left_id)
    right_geoms = _subtree_geom_ids(model, right_id)
    dists: list[float] = []
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in left_geoms and g2 in right_geoms) or (g2 in left_geoms and g1 in right_geoms):
            dists.append(float(c.dist))
    return np.asarray(dists, dtype=np.float64) if dists else np.zeros(0, dtype=np.float64)
