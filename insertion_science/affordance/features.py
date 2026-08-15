"""Feature extractors without geometry ID leakage."""

from __future__ import annotations

from typing import Any

import numpy as np

from insertion_science.affordance.geometry_scene import SceneHandles, characteristic_length
from interaction_retarget.skill_replay.insert import _insert_geometry
from reach_insert_rl.env.full_obs import privileged_full_features


def _geom_extents(scene: SceneHandles) -> dict[str, float]:
    spec = scene.spec
    c = spec.collision
    if spec.section == "round":
        peg_rx = peg_ry = float(c["peg_radius_m"])
    else:
        peg_rx = float(c["peg_half_width_m"])
        peg_ry = float(c["peg_half_depth_m"])
    hole_r = float(c["hole_half_xy_m"])
    clearance = hole_r - max(peg_rx, peg_ry)
    return {
        "peg_rx": peg_rx,
        "peg_ry": peg_ry,
        "hole_r": hole_r,
        "clearance": clearance,
        "char_len": float(scene.char_len),
    }


def tip_lat_axis_features(env) -> np.ndarray:
    feat = privileged_full_features(env)
    return np.asarray(
        [feat["tip_dist"], feat["lat_err"], feat["along"], feat["axis_err"]],
        dtype=np.float64,
    )


def raw_relation_features(scene: SceneHandles) -> np.ndarray:
    env = scene.env
    feat = privileged_full_features(env)
    tip, socket, hole, _ = _insert_geometry(env)
    origin = np.asarray(socket, dtype=np.float64)
    Rm_z = np.asarray(hole, dtype=np.float64)
    Rm_z = Rm_z / max(float(np.linalg.norm(Rm_z)), 1e-9)
    # tip in a partial socket frame (along = z)
    tip_rel = tip - origin
    along = float(np.dot(tip_rel, Rm_z))
    lat_vec = tip_rel - along * Rm_z
    ext = _geom_extents(scene)
    # NO family_id / section one-hot
    return np.asarray(
        [
            *lat_vec.tolist(),
            along,
            float(feat["tip_dist"]),
            float(feat["axis_err"]),
            float(np.dot(np.asarray(feat["peg_axis"]), Rm_z)),
            ext["peg_rx"],
            ext["peg_ry"],
            ext["hole_r"],
            ext["clearance"],
            ext["char_len"],
        ],
        dtype=np.float64,
    )


def contact_affordance_features(
    scene: SceneHandles,
    *,
    twist_dir_world: np.ndarray,
    n_rim: int = 8,
) -> np.ndarray:
    """Object-target contact field: rim clearances + contact stats + tip_lat_axis."""
    env = scene.env
    tip, socket, hole, _ = _insert_geometry(env)
    origin = np.asarray(socket, dtype=np.float64)
    z = np.asarray(hole, dtype=np.float64)
    z = z / max(float(np.linalg.norm(z)), 1e-9)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, z))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= max(float(np.linalg.norm(x)), 1e-9)
    y = np.cross(z, x)
    ext = _geom_extents(scene)
    # sample peg rim points projected in socket xy at tip height
    peg_r = max(ext["peg_rx"], ext["peg_ry"])
    hole_r = ext["hole_r"]
    dists = []
    for k in range(n_rim):
        ang = 2 * np.pi * k / n_rim
        # elliptical rim for rectangular approx
        px = ext["peg_rx"] * np.cos(ang)
        py = ext["peg_ry"] * np.sin(ang)
        p = tip + x * px + y * py
        # radial distance from hole axis in socket plane
        v = p - origin
        along = float(np.dot(v, z))
        radial = v - along * z
        r = float(np.linalg.norm(radial))
        # signed clearance to hole wall (positive = inside clearance)
        dists.append(hole_r - r)
    dists = np.asarray(dists, dtype=np.float64)
    # contact count peg-socket
    model, data = env.model, env.data
    peg_id = int(env._peg_body_id)
    sock_id = int(env._socket_body_id)
    peg_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == peg_id}
    sock_geoms = set()
    for bid in range(model.nbody):
        if bid == sock_id or int(model.body_parentid[bid]) == sock_id:
            for g in range(model.ngeom):
                if int(model.geom_bodyid[g]) == bid:
                    sock_geoms.add(g)
    n_con = 0
    depths = []
    for i in range(int(data.ncon)):
        g1, g2 = int(data.contact[i].geom1), int(data.contact[i].geom2)
        if {g1, g2} & peg_geoms and {g1, g2} & sock_geoms:
            n_con += 1
            depths.append(float(data.contact[i].dist))
    depth_mean = float(np.mean(depths)) if depths else 0.0
    twist = np.asarray(twist_dir_world, dtype=np.float64)
    tn = float(np.linalg.norm(twist))
    twist_u = twist / tn if tn > 1e-9 else np.zeros(3)
    # alignment of twist with opening axis and with radial outward
    align_axis = float(np.dot(twist_u, z))
    tip_lat = tip - origin - float(np.dot(tip - origin, z)) * z
    lat_n = float(np.linalg.norm(tip_lat))
    lat_u = tip_lat / lat_n if lat_n > 1e-9 else np.zeros(3)
    align_radial = float(np.dot(twist_u, lat_u))

    base = tip_lat_axis_features(env)
    aff = np.asarray(
        [
            *base.tolist(),
            float(np.min(dists)),
            float(np.mean(dists)),
            float(np.max(dists)),
            float(np.std(dists)),
            float(n_con),
            depth_mean,
            align_axis,
            align_radial,
            ext["clearance"],
            peg_r,
            hole_r,
            ext["char_len"],
        ],
        dtype=np.float64,
    )
    return aff


def split_socket_channels(feat_name: str, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (object_or_shared, socket_side) for shuffle control."""
    if feat_name == "tip_lat_axis":
        # all pose-relative; shuffle entire vector as weak control
        return x[:0], x
    if feat_name == "raw_relation":
        # last 4 are peg_rx,peg_ry,hole_r,clearance,char — socket-ish: hole_r, clearance
        # indices: 0:3 lat, 3 along, 4 tip, 5 axis_err, 6 peg·hole, 7 peg_rx, 8 peg_ry, 9 hole_r, 10 clearance, 11 char
        obj = x.copy()
        sock = x.copy()
        # mark socket channels
        return obj, sock
    if feat_name == "contact_affordance":
        return x.copy(), x.copy()
    return x, x


def shuffle_pairing(X: np.ndarray, feat_name: str, rng: np.random.Generator) -> np.ndarray:
    """Shuffle socket-related channels across rows (destroy object-target pairing)."""
    X2 = np.asarray(X, dtype=np.float64).copy()
    n = X2.shape[0]
    perm = rng.permutation(n)
    if feat_name == "tip_lat_axis":
        # tip/lat/axis are already relative; shuffling rows' features vs labels is done outside
        return X2[perm]
    if feat_name == "raw_relation":
        # shuffle hole_r, clearance (indices 9,10) and optionally hole-alignment channel 6
        for idx in (6, 9, 10):
            X2[:, idx] = X2[perm, idx]
        return X2
    if feat_name == "contact_affordance":
        # shuffle rim clearance stats + hole_r + clearance (indices after tip_lat 4:)
        # layout: 0:4 tip_lat, 4:8 rim stats, 8 n_con, 9 depth, 10 align_axis, 11 align_rad, 12 clearance, 13 peg_r, 14 hole_r, 15 char
        for idx in (4, 5, 6, 7, 12, 14):
            X2[:, idx] = X2[perm, idx]
        return X2
    return X2[perm]
