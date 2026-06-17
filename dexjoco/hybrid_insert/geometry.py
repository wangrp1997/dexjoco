"""Pose helpers for peg-in-hole geometry in MuJoCo."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
    return R.from_quat(quat_xyzw).as_matrix()


def body_z_axis(xmat: np.ndarray) -> np.ndarray:
    axis = np.asarray(xmat, dtype=np.float64).reshape(3, 3)[:, 2]
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return axis / norm


def site_z_axis(site_xmat: np.ndarray) -> np.ndarray:
    return body_z_axis(site_xmat)


def project_onto_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    return vector - normal * np.dot(vector, normal)


def height_along_axis(point: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> float:
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return float(np.dot(point - origin, axis))


def lateral_error(
    point: np.ndarray, origin: np.ndarray, axis: np.ndarray
) -> tuple[float, np.ndarray]:
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    rel = point - origin
    lateral = project_onto_plane(rel, axis)
    return float(np.linalg.norm(lateral)), lateral


def hole_opening_axis(
    socket_pos: np.ndarray,
    socket_xmat: np.ndarray,
    bottom_pos: np.ndarray,
) -> np.ndarray:
    """Unit axis pointing out of the hole (socket opening toward approach side)."""
    axis = site_z_axis(socket_xmat)
    if float(np.dot(axis, socket_pos - bottom_pos)) < 0.0:
        axis = -axis
    return axis


def lateral_align_delta(
    tip: np.ndarray,
    socket: np.ndarray,
    hole_axis: np.ndarray,
    *,
    gain: float,
) -> np.ndarray:
    """Move tip toward socket center in the plane perpendicular to hole_axis."""
    lat_vec = project_onto_plane(tip - socket, hole_axis)
    return -gain * lat_vec


def insert_along_hole_delta(
    hole_axis: np.ndarray,
    *,
    step_m: float,
) -> np.ndarray:
    """Advance peg tip into the hole along -hole_axis."""
    axis = np.asarray(hole_axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return -axis * step_m


def toward_socket_delta(
    tip: np.ndarray,
    socket: np.ndarray,
    *,
    gain: float,
    max_step_m: float,
) -> np.ndarray:
    """Move tip toward socket in 3D (reduces tip-socket Euclidean distance)."""
    vec = np.asarray(socket, dtype=np.float64) - np.asarray(tip, dtype=np.float64)
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        return np.zeros(3, dtype=np.float64)
    step = min(dist, max_step_m)
    return gain * vec * (step / dist)


def tip_socket_distance(tip: np.ndarray, socket: np.ndarray) -> float:
    """Euclidean distance between a peg reference point and socket site (world frame)."""
    return float(np.linalg.norm(np.asarray(tip, dtype=np.float64) - np.asarray(socket, dtype=np.float64)))


# Bottom of industreal_round_peg_8mm_collision cylinder in peg body frame (0.081 - 0.0675).
PEG_INSERT_END_BODY_OFFSET = np.array([0.0, 0.0, 0.0135], dtype=np.float64)


def peg_insert_end_pos(body_pos: np.ndarray, body_xmat: np.ndarray) -> np.ndarray:
    """World position of the peg pin end that enters the hole (not tip_site / grasp end)."""
    rot = np.asarray(body_xmat, dtype=np.float64).reshape(3, 3)
    origin = np.asarray(body_pos, dtype=np.float64)
    return origin + rot @ PEG_INSERT_END_BODY_OFFSET


def line_align_target_axis(peg_axis: np.ndarray, hole_axis: np.ndarray) -> np.ndarray:
    """Pick ±hole_axis on the same line, whichever is closer to peg_axis."""
    peg_axis = np.asarray(peg_axis, dtype=np.float64)
    hole_axis = np.asarray(hole_axis, dtype=np.float64)
    peg_axis = peg_axis / (np.linalg.norm(peg_axis) + 1e-8)
    hole_axis = hole_axis / (np.linalg.norm(hole_axis) + 1e-8)
    if float(np.dot(peg_axis, hole_axis)) < 0.0:
        return -hole_axis
    return hole_axis


def rotation_world_from_to(from_vec: np.ndarray, to_vec: np.ndarray) -> R:
    """World-frame rotation mapping one unit vector toward another (shortest path)."""
    from_vec = np.asarray(from_vec, dtype=np.float64)
    to_vec = np.asarray(to_vec, dtype=np.float64)
    from_vec = from_vec / (np.linalg.norm(from_vec) + 1e-8)
    to_vec = to_vec / (np.linalg.norm(to_vec) + 1e-8)
    cross = np.cross(from_vec, to_vec)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(from_vec, to_vec), -1.0, 1.0))
    if cross_norm < 1e-8:
        if dot > 0.0:
            return R.identity()
        # 180°: rotate around any axis perpendicular to from_vec.
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(from_vec[0]) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(from_vec, helper)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        return R.from_rotvec(axis * np.pi)
    return R.from_rotvec(cross / cross_norm * float(np.arctan2(cross_norm, dot)))


def wrist_rotvec_align_peg_axis(
    peg_axis: np.ndarray,
    hole_axis: np.ndarray,
    wrist_rotvec: np.ndarray,
    *,
    angle_tol_rad: float = 0.15,
    gain: float = 0.15,
    max_step_rad: float = 0.015,
) -> np.ndarray | None:
    """Return new wrist rotvec that slightly turns peg_axis toward ±hole (line align).

    Opspace tracks the palm/mocap frame, so the correction is composed as
    R_corr @ R_wrist (not R_wrist @ R_corr).
    """
    if axis_parallel_error_rad(peg_axis, hole_axis) <= angle_tol_rad:
        return None

    target = line_align_target_axis(peg_axis, hole_axis)
    r_wrist = R.from_rotvec(np.asarray(wrist_rotvec, dtype=np.float64))
    r_corr = rotation_world_from_to(peg_axis, target)
    corr_rotvec = r_corr.as_rotvec()
    angle = float(np.linalg.norm(corr_rotvec))
    if angle < 1e-8:
        return wrist_rotvec.copy()

    step = min(max_step_rad, gain * angle)
    r_step = R.from_rotvec(corr_rotvec * (step / angle))
    return (r_step * r_wrist).as_rotvec()


def rotation_align_delta_no_flip(
    peg_axis: np.ndarray,
    hole_axis: np.ndarray,
    *,
    angle_tol_rad: float = 0.15,
) -> np.ndarray:
    """Legacy delta-rot helper; prefer wrist_rotvec_align_peg_axis."""
    target = line_align_target_axis(peg_axis, hole_axis)
    r_corr = rotation_world_from_to(peg_axis, target)
    if axis_parallel_error_rad(peg_axis, hole_axis) <= angle_tol_rad:
        return np.zeros(3, dtype=np.float64)
    return r_corr.as_rotvec()


def axis_parallel_error_rad(peg_axis: np.ndarray, hole_axis: np.ndarray) -> float:
    """Angle between axis lines, ignoring arrow direction (0 = parallel)."""
    peg_axis = peg_axis / (np.linalg.norm(peg_axis) + 1e-8)
    hole_axis = hole_axis / (np.linalg.norm(hole_axis) + 1e-8)
    return float(np.arccos(np.clip(abs(float(np.dot(peg_axis, hole_axis))), -1.0, 1.0)))


def in_approach_cylinder(
    tip: np.ndarray,
    socket: np.ndarray,
    hole_axis: np.ndarray,
    *,
    xy_tol_m: float,
    z_min_m: float,
    z_max_m: float | None = None,
) -> bool:
    """True when peg tip is above the socket and within lateral tolerance."""
    along = height_along_axis(tip, socket, hole_axis)
    lat, _ = lateral_error(tip, socket, hole_axis)
    if lat > xy_tol_m or along < z_min_m:
        return False
    if z_max_m is None:
        return True
    return along <= z_max_m
