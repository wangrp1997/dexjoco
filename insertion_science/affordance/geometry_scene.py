"""Scene helpers: load formal geometry env, place peg tip in socket frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from dexjoco.sim.envs.assembly_geometry import names_from_raw
from dexjoco.sim.envs.panda_bimanual_assembly_env import PandaBimanualAssemblyGymEnv
from embodied_grasp_insertion.geometry.family_spec import GeometryFamilySpec
from embodied_grasp_insertion.geometry.formal_xml_builder import (
    write_dual_socket_family_assets,
    write_formal_family_assets,
)
from interaction_retarget.skill_replay.insert import _insert_geometry


@dataclass
class SceneHandles:
    env: Any
    spec: GeometryFamilySpec
    char_len: float
    peg_qpos_adr: int
    peg_qvel_adr: int
    socket_qpos_adr: int
    socket_qvel_adr: int
    instance_key: str


def characteristic_length(spec: GeometryFamilySpec) -> float:
    c = spec.collision
    if spec.section == "round":
        return float(c["peg_radius_m"])
    return float(max(c["peg_half_width_m"], c["peg_half_depth_m"]))


def make_scene(
    spec: GeometryFamilySpec,
    *,
    seed: int = 0,
    dual_socket: bool = False,
    secondary_key: str = "b",
) -> SceneHandles:
    if dual_socket:
        write_dual_socket_family_assets(spec, secondary_key=secondary_key, overwrite=False)
    else:
        write_formal_family_assets(spec, overwrite=False)
    env = PandaBimanualAssemblyGymEnv(
        geometry_family=spec.family_id,
        image_obs=False,
        randomize=False,
        hz=0,
        seed=seed,
    )
    env.reset()
    L = characteristic_length(spec)
    return SceneHandles(
        env=env,
        spec=spec,
        char_len=L,
        peg_qpos_adr=int(env._peg_qpos_adr),
        peg_qvel_adr=int(env._peg_qvel_adr),
        socket_qpos_adr=int(env._socket_qpos_adr),
        socket_qvel_adr=int(env._socket_qvel_adr),
        instance_key="primary",
    )


def socket_frame(env) -> tuple[np.ndarray, np.ndarray]:
    """Return (origin, R_world_from_socket) with +z = hole axis."""
    tip, socket, hole, _ = _insert_geometry(env)
    z = np.asarray(hole, dtype=np.float64).reshape(3)
    z = z / max(float(np.linalg.norm(z)), 1e-9)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, z))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= max(float(np.linalg.norm(x)), 1e-9)
    y = np.cross(z, x)
    Rm = np.stack([x, y, z], axis=1)
    return np.asarray(socket, dtype=np.float64).copy(), Rm


def peg_tip_pose(env) -> tuple[np.ndarray, np.ndarray]:
    tip, _, _, _ = _insert_geometry(env)
    names = names_from_raw(env)
    tip_id = int(env.model.site(names.peg_tip_site).id)
    # tip site xmat
    xmat = np.asarray(env.data.site_xmat[tip_id], dtype=np.float64).reshape(3, 3)
    return np.asarray(tip, dtype=np.float64).copy(), xmat


def body_pose_from_tip(
    env,
    tip_pos: np.ndarray,
    tip_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer peg freejoint body pose so tip site matches desired tip pose."""
    names = names_from_raw(env)
    tip_id = int(env.model.site(names.peg_tip_site).id)
    site_pos_local = np.asarray(env.model.site_pos[tip_id], dtype=np.float64)
    site_quat = np.asarray(env.model.site_quat[tip_id], dtype=np.float64)  # wxyz
    site_mat_local = R.from_quat(site_quat, scalar_first=True).as_matrix()
    R_tip = np.asarray(tip_rot, dtype=np.float64).reshape(3, 3)
    R_body = R_tip @ site_mat_local.T
    body_pos = np.asarray(tip_pos, dtype=np.float64) - R_body @ site_pos_local
    body_quat = R.from_matrix(R_body).as_quat(scalar_first=True)
    return body_pos, body_quat


def set_peg_pose(env, body_pos: np.ndarray, body_quat_wxyz: np.ndarray) -> None:
    env._set_free_joint_pose(env._peg_qpos_adr, env._peg_qvel_adr, body_pos, body_quat_wxyz)
    mujoco.mj_forward(env.model, env.data)


def pin_socket(scene: SceneHandles, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    scene.env._set_free_joint_pose(
        scene.socket_qpos_adr, scene.socket_qvel_adr, pos, quat_wxyz
    )


def capture_socket_pose(scene: SceneHandles) -> tuple[np.ndarray, np.ndarray]:
    q = scene.env.data.qpos
    adr = scene.socket_qpos_adr
    pos = np.asarray(q[adr : adr + 3], dtype=np.float64).copy()
    quat = np.asarray(q[adr + 3 : adr + 7], dtype=np.float64).copy()
    return pos, quat


def place_tip_in_socket_frame(
    scene: SceneHandles,
    *,
    tip_offset_L: np.ndarray,
    tilt_rad: np.ndarray,
) -> dict[str, Any]:
    env = scene.env
    origin, Rm = socket_frame(env)
    L = scene.char_len
    tip_pos = origin + Rm @ (np.asarray(tip_offset_L, dtype=np.float64) * L)
    R_tilt = R.from_rotvec(Rm @ np.asarray(tilt_rad, dtype=np.float64))
    R_tip = R_tilt.as_matrix() @ Rm
    body_pos, body_quat = body_pose_from_tip(env, tip_pos, R_tip)
    sock_pos, sock_quat = capture_socket_pose(scene)
    set_peg_pose(env, body_pos, body_quat)
    pin_socket(scene, sock_pos, sock_quat)
    mujoco.mj_forward(env.model, env.data)
    tip2, _ = peg_tip_pose(env)
    return {
        "tip_pos": tip2,
        "socket_origin": origin,
        "R_socket": Rm,
        "sock_pos": sock_pos,
        "sock_quat": sock_quat,
    }
