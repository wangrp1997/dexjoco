"""CAD landmarks and deterministic geometric solvers for V2 vision."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .spatial_visual_supervision import CameraCalibration, PEG_CLASS, SOCKET_CLASS


@dataclass(frozen=True)
class CADLandmarkSet:
    names: tuple[str, ...]
    object_kinds: tuple[str, ...]
    support_classes: tuple[int, ...]
    body_names: tuple[str, ...]
    points_body: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.names)
        if not (
            len(self.object_kinds)
            == len(self.support_classes)
            == len(self.body_names)
            == count
        ):
            raise ValueError("CAD landmark metadata lengths differ")
        points = np.asarray(self.points_body, dtype=np.float64)
        if points.shape != (count, 3):
            raise ValueError(f"points_body must have shape ({count}, 3), got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("CAD landmark points must be finite")
        object.__setattr__(self, "points_body", points.copy())

    def world_points(self, raw_env) -> np.ndarray:
        output = np.empty_like(self.points_body)
        for row, (body_name, point_body) in enumerate(
            zip(self.body_names, self.points_body, strict=True)
        ):
            body_id = int(raw_env._model.body(body_name).id)
            rotation = np.asarray(raw_env._data.xmat[body_id]).reshape(3, 3)
            position = np.asarray(raw_env._data.xpos[body_id])
            output[row] = position + rotation @ point_body
        return output


def _circle_points(radius_x: float, radius_y: float, z: float, count: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack(
        [radius_x * np.cos(angles), radius_y * np.sin(angles), np.full(count, z)]
    )


def _box_corners(half_x: float, half_y: float, z: float) -> np.ndarray:
    return np.asarray(
        [
            [-half_x, -half_y, z],
            [half_x, -half_y, z],
            [half_x, half_y, z],
            [-half_x, half_y, z],
        ],
        dtype=np.float64,
    )


def assembly_cad_landmarks(raw_env) -> CADLandmarkSet:
    """Derive stable peg and socket CAD points from MuJoCo collision geometry."""
    from dexjoco.sim.envs.assembly_geometry import names_from_raw

    names = names_from_raw(raw_env)
    model = raw_env._model
    peg_geom_id = int(model.geom(names.peg_collision).id)
    peg_center = np.asarray(model.geom_pos[peg_geom_id], dtype=np.float64)
    peg_size = np.asarray(model.geom_size[peg_geom_id], dtype=np.float64)
    if names.section == "round":
        peg_half_x = peg_half_y = float(peg_size[0])
        peg_half_z = float(peg_size[1])
    else:
        peg_half_x, peg_half_y, peg_half_z = map(float, peg_size[:3])
    peg_tip_z = float(peg_center[2] - peg_half_z)
    peg_far_z = float(peg_center[2] + peg_half_z)
    if names.section == "round":
        peg_tip_rim = _circle_points(peg_half_x, peg_half_y, peg_tip_z, 8)
        peg_far_rim = _circle_points(peg_half_x, peg_half_y, peg_far_z, 8)
    else:
        peg_tip_rim = _box_corners(peg_half_x, peg_half_y, peg_tip_z)
        peg_far_rim = _box_corners(peg_half_x, peg_half_y, peg_far_z)

    socket_site_id = int(model.site(names.socket_site).id)
    hole_center = np.asarray(model.site_pos[socket_site_id], dtype=np.float64)
    wall_pos_x = int(model.geom(f"{names.socket_body}_wall_pos_x").id)
    wall_neg_x = int(model.geom(f"{names.socket_body}_wall_neg_x").id)
    wall_pos_y = int(model.geom(f"{names.socket_body}_wall_pos_y").id)
    wall_neg_y = int(model.geom(f"{names.socket_body}_wall_neg_y").id)
    inner_pos_x = float(model.geom_pos[wall_pos_x, 0] - model.geom_size[wall_pos_x, 0])
    inner_neg_x = float(model.geom_pos[wall_neg_x, 0] + model.geom_size[wall_neg_x, 0])
    inner_pos_y = float(model.geom_pos[wall_pos_y, 1] - model.geom_size[wall_pos_y, 1])
    inner_neg_y = float(model.geom_pos[wall_neg_y, 1] + model.geom_size[wall_neg_y, 1])
    hole_half_x = 0.5 * (inner_pos_x - inner_neg_x)
    hole_half_y = 0.5 * (inner_pos_y - inner_neg_y)
    if hole_half_x <= 0.0 or hole_half_y <= 0.0:
        raise ValueError("socket wall geometry produced non-positive hole extents")
    if names.section == "round":
        hole_rim = _circle_points(hole_half_x, hole_half_y, hole_center[2], 8)
    else:
        hole_rim = _box_corners(hole_half_x, hole_half_y, hole_center[2])

    base_geom_id = int(model.geom(names.socket_base).id)
    base_center = np.asarray(model.geom_pos[base_geom_id], dtype=np.float64)
    base_size = np.asarray(model.geom_size[base_geom_id], dtype=np.float64)
    base_top = _box_corners(
        float(base_size[0]),
        float(base_size[1]),
        float(base_center[2] + base_size[2]),
    )
    wall_ids = (wall_pos_x, wall_neg_x, wall_pos_y, wall_neg_y)
    outer_x = max(
        abs(float(model.geom_pos[geom_id, 0])) + float(model.geom_size[geom_id, 0])
        for geom_id in wall_ids
    )
    outer_y = max(
        abs(float(model.geom_pos[geom_id, 1])) + float(model.geom_size[geom_id, 1])
        for geom_id in wall_ids
    )
    wall_top_z = max(
        float(model.geom_pos[geom_id, 2] + model.geom_size[geom_id, 2])
        for geom_id in wall_ids
    )
    wall_top = _box_corners(outer_x, outer_y, wall_top_z)

    landmark_names = ["peg_tip_center", "peg_axis_far"]
    object_kinds = ["peg", "peg"]
    support_classes = [PEG_CLASS, PEG_CLASS]
    body_names = [names.peg_body, names.peg_body]
    points = [
        np.asarray([0.0, 0.0, peg_tip_z]),
        np.asarray([0.0, 0.0, peg_far_z]),
    ]
    for prefix, rim in (("peg_tip_rim", peg_tip_rim), ("peg_far_rim", peg_far_rim)):
        for index, point in enumerate(rim):
            landmark_names.append(f"{prefix}_{index}")
            object_kinds.append("peg")
            support_classes.append(PEG_CLASS)
            body_names.append(names.peg_body)
            points.append(point)
    landmark_names.extend(["hole_center", "hole_axis_far"])
    object_kinds.extend(["socket", "socket"])
    support_classes.extend([SOCKET_CLASS, SOCKET_CLASS])
    body_names.extend([names.socket_body, names.socket_body])
    points.extend(
        [
            hole_center,
            hole_center + np.asarray([0.0, 0.0, 0.02]),
        ]
    )
    for prefix, values in (
        ("hole_rim", hole_rim),
        ("tray_base_corner", base_top),
        ("tray_wall_top_corner", wall_top),
    ):
        for index, point in enumerate(values):
            landmark_names.append(f"{prefix}_{index}")
            object_kinds.append("socket")
            support_classes.append(SOCKET_CLASS)
            body_names.append(names.socket_body)
            points.append(point)
    return CADLandmarkSet(
        names=tuple(landmark_names),
        object_kinds=tuple(object_kinds),
        support_classes=tuple(support_classes),
        body_names=tuple(body_names),
        points_body=np.stack(points),
    )


@dataclass(frozen=True)
class CADPoseEstimate:
    position_world: np.ndarray
    rotation_world: np.ndarray
    reprojection_rmse_px: float
    inlier_count: int


def solve_cad_pose_pnp(
    points_object: np.ndarray,
    points_uv: np.ndarray,
    calibration: CameraCalibration,
    *,
    valid: np.ndarray | None = None,
    minimum_points: int = 6,
    ransac_reprojection_error_px: float = 4.0,
) -> CADPoseEstimate:
    """Solve object pose from CAD correspondences and convert to world coordinates."""
    object_points = np.asarray(points_object, dtype=np.float64)
    image_points = np.asarray(points_uv, dtype=np.float64)
    if object_points.ndim != 2 or object_points.shape[1] != 3:
        raise ValueError(f"points_object must have shape (N, 3), got {object_points.shape}")
    if image_points.shape != (len(object_points), 2):
        raise ValueError(f"points_uv must have shape ({len(object_points)}, 2)")
    selected = np.ones(len(object_points), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    selected &= np.isfinite(object_points).all(axis=1) & np.isfinite(image_points).all(axis=1)
    if int(selected.sum()) < minimum_points:
        raise ValueError(f"PnP requires at least {minimum_points} valid correspondences")
    object_selected = object_points[selected]
    image_selected = image_points[selected]
    success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        object_selected,
        image_selected,
        calibration.intrinsic_matrix,
        None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=float(ransac_reprojection_error_px),
        iterationsCount=200,
        confidence=0.999,
    )
    if not success or inliers is None or len(inliers) < minimum_points:
        raise RuntimeError("CAD PnP failed to find a valid inlier pose")
    inlier_rows = inliers.reshape(-1)
    rotation_vector, translation = cv2.solvePnPRefineLM(
        object_selected[inlier_rows],
        image_selected[inlier_rows],
        calibration.intrinsic_matrix,
        None,
        rotation_vector,
        translation,
    )
    projected, _ = cv2.projectPoints(
        object_selected[inlier_rows],
        rotation_vector,
        translation,
        calibration.intrinsic_matrix,
        None,
    )
    residual = projected.reshape(-1, 2) - image_selected[inlier_rows]
    rotation_cv_object = cv2.Rodrigues(rotation_vector)[0]
    mujoco_from_cv = np.diag([1.0, -1.0, -1.0])
    rotation_world = (
        calibration.rotation_world_from_camera @ mujoco_from_cv @ rotation_cv_object
    )
    position_world = calibration.position_world + (
        calibration.rotation_world_from_camera
        @ mujoco_from_cv
        @ translation.reshape(3)
    )
    return CADPoseEstimate(
        position_world=position_world,
        rotation_world=rotation_world,
        reprojection_rmse_px=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        inlier_count=int(len(inlier_rows)),
    )


def symmetric_assembly_state5(
    peg_tip_world: np.ndarray,
    peg_axis_world: np.ndarray,
    hole_position_world: np.ndarray,
    hole_rotation_world: np.ndarray,
) -> np.ndarray:
    """Compute the five-dimensional state without using peg roll."""
    hole_rotation = np.asarray(hole_rotation_world, dtype=np.float64).reshape(3, 3)
    relative_position = hole_rotation.T @ (
        np.asarray(peg_tip_world, dtype=np.float64).reshape(3)
        - np.asarray(hole_position_world, dtype=np.float64).reshape(3)
    )
    peg_axis_hole = hole_rotation.T @ np.asarray(peg_axis_world, dtype=np.float64).reshape(3)
    peg_axis_hole /= np.linalg.norm(peg_axis_hole) + 1e-12
    target_axis = np.asarray([0.0, 0.0, 1.0])
    cross = np.cross(target_axis, peg_axis_hole)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(target_axis, peg_axis_hole), -1.0, 1.0))
    if cross_norm <= 1e-12:
        tilt_rotvec = np.zeros(3)
    else:
        tilt_rotvec = cross / cross_norm * np.arctan2(cross_norm, dot)
    return np.asarray(
        [
            relative_position[0],
            relative_position[1],
            tilt_rotvec[0],
            tilt_rotvec[1],
            -relative_position[2],
        ],
        dtype=np.float64,
    )
