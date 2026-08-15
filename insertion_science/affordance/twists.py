"""Matched twist perturbations and free/blocked/jam labels."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from insertion_science.affordance.geometry_scene import (
    SceneHandles,
    peg_tip_pose,
    pin_socket,
    set_peg_pose,
)
from interaction_retarget.skill_replay.insert import _insert_geometry


def _peg_socket_contact_force(env) -> float:
    model, data = env.model, env.data
    peg_id = int(env._peg_body_id)
    sock_id = int(env._socket_body_id)
    peg_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == peg_id}
    sock_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == sock_id}
    # include socket subtree
    for bid in range(model.nbody):
        if int(model.body_parentid[bid]) == sock_id or bid == sock_id:
            for g in range(model.ngeom):
                if int(model.geom_bodyid[g]) == bid:
                    sock_geoms.add(g)
    total = 0.0
    force = np.zeros(6, dtype=np.float64)
    for i in range(int(data.ncon)):
        g1, g2 = int(data.contact[i].geom1), int(data.contact[i].geom2)
        pair = {g1, g2}
        if peg_geoms & pair and sock_geoms & pair:
            mujoco.mj_contactForce(model, data, i, force)
            total += float(np.linalg.norm(force[:3]))
    return total


def apply_twist_and_label(
    scene: SceneHandles,
    *,
    sock_pos: np.ndarray,
    sock_quat: np.ndarray,
    R_socket: np.ndarray,
    d_L: np.ndarray,
    r_rad: np.ndarray,
    settle_steps: int,
    gravity_off: bool,
    label_cfg: dict[str, float],
) -> dict[str, Any]:
    env = scene.env
    L = scene.char_len
    tip0, R0 = peg_tip_pose(env)
    # body pose before
    body_pos0 = np.asarray(env.data.qpos[scene.peg_qpos_adr : scene.peg_qpos_adr + 3]).copy()
    body_quat0 = np.asarray(env.data.qpos[scene.peg_qpos_adr + 3 : scene.peg_qpos_adr + 7]).copy()

    d_world = R_socket @ (np.asarray(d_L, dtype=np.float64) * L)
    r_world = R_socket @ np.asarray(r_rad, dtype=np.float64)
    R_delta = R.from_rotvec(r_world)
    tip_cmd = tip0 + d_world
    R_cmd = R_delta.as_matrix() @ R0
    # map tip command to body
    from insertion_science.affordance.geometry_scene import body_pose_from_tip

    body_pos, body_quat = body_pose_from_tip(env, tip_cmd, R_cmd)

    # capture gravity
    g0 = np.asarray(env.model.opt.gravity, dtype=np.float64).copy()
    if gravity_off:
        env.model.opt.gravity[:] = 0.0

    set_peg_pose(env, body_pos, body_quat)
    pin_socket(scene, sock_pos, sock_quat)
    # zero peg vel
    env.data.qvel[scene.peg_qvel_adr : scene.peg_qvel_adr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)

    forces = []
    for _ in range(int(settle_steps)):
        pin_socket(scene, sock_pos, sock_quat)
        # hold arms: zero policy action via raw ctrl keep / no step through opspace —
        # use mj_step only on free objects by stepping model with current ctrl
        mujoco.mj_step(env.model, env.data)
        forces.append(_peg_socket_contact_force(env))

    tip1, _ = peg_tip_pose(env)
    tip_delta = tip1 - tip0
    cmd = d_world
    cmd_n = float(np.linalg.norm(cmd))
    if cmd_n < 1e-9:
        # pure rotation: use tip lateral motion magnitude vs tilt severity
        progress = 0.0
        intended = np.zeros(3)
        lat_dev = float(np.linalg.norm(tip_delta))
        commanded = max(L * float(np.linalg.norm(r_rad)), 1e-6)
        progress_frac = 1.0 - min(lat_dev / (0.5 * L + 1e-9), 1.0)  # less lateral blowup ~ freer? 
        # For rotation, prefer force-based: low force after tilt = free (clearance), high = jam
        progress_frac = float("nan")
    else:
        intended = cmd / cmd_n
        progress = float(np.dot(tip_delta, intended))
        lat_dev = float(np.linalg.norm(tip_delta - progress * intended))
        commanded = cmd_n
        progress_frac = progress / commanded

    force_mean = float(np.mean(forces)) if forces else 0.0
    # restore gravity
    env.model.opt.gravity[:] = g0

    free_p = float(label_cfg["free_progress_frac"])
    jam_p = float(label_cfg["jam_progress_frac"])
    free_f = float(label_cfg["free_force_n"])
    jam_f = float(label_cfg["jam_force_n"])
    lat_lim = float(label_cfg["lat_dev_L"]) * L

    if cmd_n < 1e-9:
        # rotation twists
        if force_mean <= free_f and lat_dev <= lat_lim:
            label = "free"
        elif force_mean >= jam_f:
            label = "jam"
        else:
            label = "blocked"
        progress_frac = 0.0 if label != "free" else 1.0
    else:
        if progress_frac >= free_p and force_mean <= free_f and lat_dev <= lat_lim:
            label = "free"
        elif progress_frac <= jam_p and force_mean >= jam_f:
            label = "jam"
        else:
            label = "blocked"

    # restore peg for next matched branch caller responsibility
    return {
        "label": label,
        "feasible": label == "free",
        "progress": float(progress) if cmd_n >= 1e-9 else 0.0,
        "progress_frac": float(progress_frac) if progress_frac == progress_frac else 0.0,
        "lat_dev": float(lat_dev),
        "force_mean_n": force_mean,
        "commanded_m": float(commanded),
        "tip0": tip0.tolist(),
        "tip1": tip1.tolist(),
        "d_world": d_world.tolist(),
        "body_pos0": body_pos0.tolist(),
        "body_quat0": body_quat0.tolist(),
    }


def restore_peg(scene: SceneHandles, body_pos: np.ndarray, body_quat: np.ndarray,
                sock_pos: np.ndarray, sock_quat: np.ndarray) -> None:
    set_peg_pose(scene.env, body_pos, body_quat)
    pin_socket(scene, sock_pos, sock_quat)
    scene.env.data.qvel[scene.peg_qvel_adr : scene.peg_qvel_adr + 6] = 0.0
    mujoco.mj_forward(scene.env.model, scene.env.data)
