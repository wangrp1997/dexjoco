"""Hand keypoints and collision geometry from MuJoCo geoms."""

from __future__ import annotations

import mujoco
import numpy as np

from interaction_retarget.constants import LEFT_HAND_BODIES, NUM_HAND_KEYPOINTS, RIGHT_HAND_BODIES


def _subtree_bodies(model, root_body_id: int) -> set[int]:
    bodies = {int(root_body_id)}
    changed = True
    while changed:
        changed = False
        for bid in range(model.nbody):
            parent = int(model.body_parentid[bid])
            if parent in bodies and bid not in bodies:
                bodies.add(bid)
                changed = True
    return bodies


def _primary_geom_for_body(model, body_id: int) -> int | None:
    candidates = [gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) == int(body_id)]
    if not candidates:
        return None
    for gid in candidates:
        if int(model.geom_group[gid]) == 1:
            return gid
    return candidates[0]


def hand_keypoints_world(model, data, body_names: tuple[str, ...]) -> np.ndarray:
    """Use geom world positions (visual if available) instead of body origins."""
    if len(body_names) != NUM_HAND_KEYPOINTS:
        raise ValueError(f"Expected {NUM_HAND_KEYPOINTS} hand bodies, got {len(body_names)}")
    pts: list[np.ndarray] = []
    for name in body_names:
        body_id = int(model.body(name).id)
        gid = _primary_geom_for_body(model, body_id)
        if gid is not None:
            pts.append(np.asarray(data.geom_xpos[gid], dtype=np.float64).copy())
        else:
            pts.append(np.asarray(data.xpos[body_id], dtype=np.float64).copy())
    return np.stack(pts, axis=0)


def hand_collision_segments_world(
    model,
    data,
    root_body_name: str,
    *,
    max_segments: int = 800,
) -> np.ndarray:
    """Wireframe of hand collision geoms (boxes/capsules) in world frame."""
    root_id = int(model.body(root_body_name).id)
    bodies = _subtree_bodies(model, root_id)
    segments: list[np.ndarray] = []

    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) not in bodies:
            continue
        if int(model.geom_group[gid]) != 3:
            continue
        seg = _geom_wireframe_segments(model, data, gid)
        if seg is not None and seg.size:
            segments.append(seg)

    if not segments:
        return np.zeros((0, 2, 3), dtype=np.float64)
    out = np.concatenate(segments, axis=0)
    if out.shape[0] > max_segments:
        idx = np.linspace(0, out.shape[0] - 1, max_segments, dtype=int)
        out = out[idx]
    return out


def _geom_wireframe_segments(model, data, gid: int) -> np.ndarray | None:
    pos = np.asarray(data.geom_xpos[gid], dtype=np.float64)
    mat = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    gtype = int(model.geom_type[gid])
    size = np.asarray(model.geom_size[gid], dtype=np.float64)

    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        hx, hy, hz = size
        corners = np.array(
            [
                [-hx, -hy, -hz],
                [hx, -hy, -hz],
                [hx, hy, -hz],
                [-hx, hy, -hz],
                [-hx, -hy, hz],
                [hx, -hy, hz],
                [hx, hy, hz],
                [-hx, hy, hz],
            ],
            dtype=np.float64,
        )
        world = corners @ mat.T + pos
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        return np.stack([[world[a], world[b]] for a, b in edges], axis=0)

    if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius, half_len = size[0], size[1]
        axis = mat[:, 2]
        p0 = pos - axis * half_len
        p1 = pos + axis * half_len
        return np.stack([[p0, p1]], axis=0)

    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return np.stack([[pos, pos + mat[:, 0] * size[0]]], axis=0)

    return None
