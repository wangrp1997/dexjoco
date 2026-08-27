"""Public proprioception and ego-depth geometry for a coarse insertion frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


_TCP_TO_PALM = np.asarray(
    [
        [0.0, 0.0, 1.0, 0.05],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.03],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_FINGER_LINKS = (0.0164, 0.054, 0.0384)
_PAD_CENTER_Z = (0.019, 0.019, 0.019, 0.035)

# Fixed deployable robot, peg-CAD, and ego-camera calibration constants.
_PAD_RADIUS_M = 0.012
_PEG_RADIUS_M = 0.01785
_PEG_EFFECTIVE_CONTACT_RADIUS_M = _PEG_RADIUS_M + _PAD_RADIUS_M
_PEG_SHAFT_CENTER_Z_M = 0.081
_PEG_INSERT_END_Z_M = 0.0135
_PEG_CENTER_TO_INSERT_END_M = _PEG_SHAFT_CENTER_Z_M - _PEG_INSERT_END_Z_M
_EGO_CAMERA_POSITION_WORLD = np.asarray([-1.3, 0.0, 2.2], dtype=np.float64)
_EGO_CAMERA_QUATERNION_WXYZ = (
    0.6532815,
    0.2705981,
    -0.2705981,
    -0.6532815,
)
_EGO_CAMERA_FOVY_RAD = np.deg2rad(45.0)
_EGO_DEPTH_SHAPE = (640, 640)
_DEPTH_STRIDE = 4
_TRAY_ROI_RADIUS_M = 0.18
_RANSAC_TRIALS = 160
_PLANE_DISTANCE_THRESHOLD_M = 0.003
_MIN_PLANE_INLIERS = 40
_MIN_PLANE_SPAN_M = 0.04
_MAX_PLANE_SPAN_M = 0.25
_MIN_PLANE_NORMAL_ALIGNMENT = float(np.cos(np.deg2rad(60.0)))

_MIN_FINGER_SPAN_M = 0.02
_MAX_LINE_RATIO = 0.25
_MIN_OPPOSITION_M = 0.02


class UnobservableTaskFrame(ValueError):
    """Raised when encoder geometry cannot safely orient an insertion frame."""


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result.flags.writeable = False
    return result


def _axis_rotation(axis: str, angle: float) -> np.ndarray:
    vector = {
        "x": (angle, 0.0, 0.0),
        "y": (0.0, angle, 0.0),
        "z": (0.0, 0.0, angle),
    }[axis]
    return Rotation.from_rotvec(vector).as_matrix()


def _fixed_rotation(quaternion_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    return Rotation.from_quat(quaternion_wxyz, scalar_first=True).as_matrix()


def _finger_pad_in_palm(
    q4: np.ndarray,
    *,
    base_position: tuple[float, float, float],
    base_x_angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(base_position, dtype=np.float64).copy()
    rotation = _axis_rotation("x", base_x_angle) @ _axis_rotation("z", q4[0])
    for length, angle in zip(_FINGER_LINKS, q4[1:], strict=True):
        position += rotation @ np.asarray([0.0, 0.0, length])
        rotation = rotation @ _axis_rotation("y", angle)
    position += rotation @ np.asarray([0.0, 0.0, _PAD_CENTER_Z[0]])
    return position, rotation[:, 2].copy()


def _thumb_pad_in_palm(q4: np.ndarray, *, side: str) -> tuple[np.ndarray, np.ndarray]:
    if side == "right":
        position = np.asarray([-0.0182, 0.019333, -0.045987])
        rotation = _fixed_rotation((0.477714, -0.521334, -0.521334, -0.477714))
        joint0, joint1 = -q4[0], q4[1]
        proximal = np.asarray([-0.027, 0.005, 0.0399])
    else:
        position = np.asarray([-0.0182, -0.019333, -0.045987])
        rotation = _fixed_rotation((0.477714, 0.521334, -0.521334, 0.477714))
        joint0, joint1 = q4[0], -q4[1]
        proximal = np.asarray([-0.027, -0.005, 0.0399])
    rotation = rotation @ _axis_rotation("x", joint0)
    position += rotation @ proximal
    rotation = rotation @ _axis_rotation("z", joint1)
    position += rotation @ np.asarray([0.0, 0.0, 0.0177])
    rotation = rotation @ _axis_rotation("y", q4[2])
    position += rotation @ np.asarray([0.0, 0.0, 0.0514])
    rotation = rotation @ _axis_rotation("y", q4[3])
    position += rotation @ np.asarray([0.0, 0.0, _PAD_CENTER_Z[3]])
    return position, rotation[:, 2].copy()


def fingertip_pads_in_palm(hand16: np.ndarray, *, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Return pad centers and pad longitudinal axes in `[ff, mf, rf, th]` order."""
    joints = np.asarray(hand16, dtype=np.float64)
    if joints.shape != (16,) or not np.isfinite(joints).all():
        raise ValueError("hand16 must be a finite vector with shape (16,)")
    if side not in {"right", "left"}:
        raise ValueError("side must be 'right' or 'left'")
    blocks = joints.reshape(4, 4)
    if side == "left":
        blocks = blocks[[2, 1, 0, 3]]
    sign = 1.0 if side == "right" else -1.0
    bases = (
        ((0.0, sign * 0.0435, -0.001542), -sign * np.deg2rad(5.0)),
        ((0.0, 0.0, 0.0007), 0.0),
        ((0.0, -sign * 0.0435, -0.001542), sign * np.deg2rad(5.0)),
    )
    pads = [
        _finger_pad_in_palm(blocks[index], base_position=base, base_x_angle=angle)
        for index, (base, angle) in enumerate(bases)
    ]
    pads.append(_thumb_pad_in_palm(blocks[3], side=side))
    return _readonly(np.stack([item[0] for item in pads])), _readonly(
        np.stack([item[1] for item in pads])
    )


def fingertip_pads_world(state46: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return world pad centers and axes with shape `(2, 4, 3)` (right, left)."""
    state = np.asarray(state46, dtype=np.float64)
    if state.shape != (46,) or not np.isfinite(state).all():
        raise ValueError("state46 must be a finite vector with shape (46,)")
    outputs = []
    for side, pose_slice, hand_slice in (
        ("right", slice(0, 7), slice(14, 30)),
        ("left", slice(7, 14), slice(30, 46)),
    ):
        pose = state[pose_slice]
        quaternion = pose[3:7]
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12:
            raise ValueError(f"{side} TCP quaternion must be non-zero")
        tcp_rotation = Rotation.from_quat(quaternion / norm, scalar_first=True).as_matrix()
        palm_rotation = tcp_rotation @ _TCP_TO_PALM[:3, :3]
        palm_position = pose[:3] + tcp_rotation @ _TCP_TO_PALM[:3, 3]
        centers_palm, axes_palm = fingertip_pads_in_palm(state[hand_slice], side=side)
        centers_world = centers_palm @ palm_rotation.T + palm_position
        axes_world = axes_palm @ palm_rotation.T
        outputs.append((centers_world, axes_world))
    return _readonly(np.stack([item[0] for item in outputs])), _readonly(
        np.stack([item[1] for item in outputs])
    )


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name} is degenerate")
    return np.asarray(vector, dtype=np.float64) / norm


def _principal_line(points: np.ndarray, *, name: str) -> tuple[np.ndarray, float]:
    values = np.asarray(points, dtype=np.float64)
    span = float(np.linalg.norm(values[0] - values[2]))
    if span < _MIN_FINGER_SPAN_M:
        raise UnobservableTaskFrame(f"{name} finger span is too short")
    centered = values - np.mean(values, axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    if float(eigenvalues[-1]) <= 1e-12:
        raise UnobservableTaskFrame(f"{name} finger line is degenerate")
    line_ratio = float(max(eigenvalues[-2], 0.0) / eigenvalues[-1])
    if line_ratio > _MAX_LINE_RATIO:
        raise UnobservableTaskFrame(
            f"{name} three-finger topology mismatch: line_ratio={line_ratio:.3f}"
        )
    axis = eigenvectors[:, -1]
    if float(axis @ (values[0] - values[2])) < 0.0:
        axis = -axis
    return _unit(axis, name=f"{name} principal line"), line_ratio


def _perpendicular(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=np.float64) - axis * float(axis @ vector)


def ego_depth_to_world(depth_m: np.ndarray) -> np.ndarray:
    """Back-project the fixed ego metric-depth image into world coordinates."""
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.shape != _EGO_DEPTH_SHAPE:
        raise ValueError(f"ego depth must have shape {_EGO_DEPTH_SHAPE}, got {depth.shape}")
    finite = np.isfinite(depth)
    if np.isnan(depth).any() or np.isneginf(depth).any() or not finite.any():
        raise ValueError("ego depth must contain positive metric depth or +inf")
    if np.any(depth[finite] <= 0.0):
        raise ValueError("finite ego depth values must be positive")

    rows, columns = np.mgrid[0 : depth.shape[0] : _DEPTH_STRIDE, 0 : depth.shape[1] : _DEPTH_STRIDE]
    z = depth[::_DEPTH_STRIDE, ::_DEPTH_STRIDE]
    valid = np.isfinite(z)
    z = z[valid]
    rows = rows[valid]
    columns = columns[valid]
    focal = 0.5 * depth.shape[0] / np.tan(0.5 * _EGO_CAMERA_FOVY_RAD)
    center_x = 0.5 * (depth.shape[1] - 1)
    center_y = 0.5 * (depth.shape[0] - 1)
    camera_points = np.column_stack(
        [
            (columns - center_x) * z / focal,
            (center_y - rows) * z / focal,
            -z,
        ]
    )
    rotation_world = _fixed_rotation(_EGO_CAMERA_QUATERNION_WXYZ)
    points_world = camera_points @ rotation_world.T + _EGO_CAMERA_POSITION_WORLD
    return _readonly(points_world)


def _plane_tangents(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    tangent = _unit(np.cross(normal, axis), name="plane tangent")
    bitangent = _unit(np.cross(normal, tangent), name="plane bitangent")
    return tangent, bitangent


def _fit_tray_plane_ransac(
    scene_points_world: np.ndarray,
    *,
    roi_center_world: np.ndarray,
    peg_center_world: np.ndarray,
) -> tuple:
    scene = np.asarray(scene_points_world, dtype=np.float64)
    if scene.ndim != 2 or scene.shape[1:] != (3,):
        raise ValueError("scene_points_world must have shape (N, 3)")
    if not np.isfinite(scene).all():
        raise ValueError("scene_points_world must contain only finite points")
    roi_distance = np.linalg.norm(scene - roi_center_world, axis=1)
    roi = scene[roi_distance <= _TRAY_ROI_RADIUS_M]
    if roi.shape[0] < _MIN_PLANE_INLIERS:
        raise UnobservableTaskFrame(
            f"tray depth ROI has too few points: {roi.shape[0]}"
        )
    outward_guess = _unit(
        peg_center_world - roi_center_world,
        name="tray-to-peg bearing",
    )
    rng = np.random.default_rng(0)
    best_mask = None
    best_score = -np.inf

    for _ in range(_RANSAC_TRIALS):
        indices = rng.choice(roi.shape[0], size=3, replace=False)
        first, second, third = roi[indices]
        normal = np.cross(second - first, third - first)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-9:
            continue
        normal /= normal_norm
        if abs(float(normal @ outward_guess)) < _MIN_PLANE_NORMAL_ALIGNMENT:
            continue
        distances = np.abs((roi - first) @ normal)
        inlier_mask = distances <= _PLANE_DISTANCE_THRESHOLD_M
        inlier_count = int(np.count_nonzero(inlier_mask))
        if inlier_count < _MIN_PLANE_INLIERS:
            continue
        # The tray entry face must lie between the supporting hand and the peg.
        plane_offset = float(
            np.median((roi[inlier_mask] - roi_center_world) @ outward_guess)
        )
        if plane_offset <= 0.0:
            continue
        tangent, bitangent = _plane_tangents(normal)
        coordinates = roi[inlier_mask] @ np.column_stack([tangent, bitangent])
        lower, upper = np.percentile(coordinates, (5.0, 95.0), axis=0)
        spans = upper - lower
        if float(np.min(spans)) < _MIN_PLANE_SPAN_M:
            continue
        if float(np.max(spans)) > _MAX_PLANE_SPAN_M:
            continue
        score = float(inlier_count * spans[0] * spans[1])
        if score > best_score:
            best_score = score
            best_mask = inlier_mask

    if best_mask is None:
        raise UnobservableTaskFrame("no tray-like depth plane found in left-hand ROI")

    inliers = roi[best_mask]
    origin = np.mean(inliers, axis=0)
    centered = inliers - origin
    _, axes = np.linalg.eigh(centered.T @ centered)
    normal = axes[:, 0]
    if float(normal @ outward_guess) < 0.0:
        normal = -normal
    if float(normal @ outward_guess) < _MIN_PLANE_NORMAL_ALIGNMENT:
        raise UnobservableTaskFrame("tray plane normal disagrees with hand geometry")
    distances = np.abs((roi - origin) @ normal)
    inliers = roi[distances <= _PLANE_DISTANCE_THRESHOLD_M]
    if inliers.shape[0] < _MIN_PLANE_INLIERS:
        raise UnobservableTaskFrame("refined tray plane has too few inliers")

    origin = np.mean(inliers, axis=0)
    centered = inliers - origin
    _, axes = np.linalg.eigh(centered.T @ centered)
    normal = axes[:, 0]
    tangent = axes[:, 2]
    if float(normal @ outward_guess) < 0.0:
        normal = -normal
    bitangent = _unit(np.cross(normal, tangent), name="tray plane bitangent")
    plane_axes = np.column_stack([tangent, bitangent])
    coordinates = centered @ plane_axes
    lower, upper = np.percentile(coordinates, (5.0, 95.0), axis=0)
    spans = upper - lower
    if float(np.min(spans)) < _MIN_PLANE_SPAN_M:
        raise UnobservableTaskFrame("refined tray plane footprint is too small")
    center = origin + plane_axes @ (0.5 * (lower + upper))
    plane_rms = float(np.sqrt(np.mean(((inliers - center) @ normal) ** 2)))
    return (
        center,
        normal,
        tangent,
        plane_rms,
        int(inliers.shape[0]),
        int(roi.shape[0]),
        float(inliers.shape[0] / roi.shape[0]),
        float(0.5 * np.linalg.norm(spans)),
    )


@dataclass(frozen=True, slots=True)
class KinematicTaskEstimate:
    """A peg CAD fit plus ego-depth tray-plane fit for coarse first contact."""

    basis_world: np.ndarray
    approach_bearing_world: np.ndarray
    peg_axis_proxy_world: np.ndarray
    tray_normal_proxy_world: np.ndarray
    peg_grasp_center_world: np.ndarray
    tray_grasp_center_world: np.ndarray
    peg_insert_end_world: np.ndarray
    tray_entry_center_world: np.ndarray
    axis_alignment_cosine: float
    tray_alignment_cosine: float
    approach_distance_m: float
    surface_distance_m: float
    lateral_centering_distance_m: float
    maximum_advance_offset_m: float
    peg_radius_rms_m: float
    tray_plane_rms_m: float
    tray_plane_radius_m: float
    tray_plane_inlier_fraction: float
    tray_plane_inliers: int
    tray_roi_points: int
    right_linearity_ratio: float
    source: str = "runtime_ego_depth_ransac_round8_peg_fk_roi"

    def __post_init__(self) -> None:
        for name, shape in (
            ("basis_world", (3, 3)),
            ("approach_bearing_world", (3,)),
            ("peg_axis_proxy_world", (3,)),
            ("tray_normal_proxy_world", (3,)),
            ("peg_grasp_center_world", (3,)),
            ("tray_grasp_center_world", (3,)),
            ("peg_insert_end_world", (3,)),
            ("tray_entry_center_world", (3,)),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            object.__setattr__(self, name, _readonly(value))
        if not np.allclose(self.basis_world.T @ self.basis_world, np.eye(3), atol=1e-6):
            raise ValueError("basis_world must be orthonormal")
        if not np.isclose(np.linalg.det(self.basis_world), 1.0, atol=1e-6):
            raise ValueError("basis_world must be right-handed")
        scalars = (
            self.axis_alignment_cosine,
            self.tray_alignment_cosine,
            self.approach_distance_m,
            self.surface_distance_m,
            self.lateral_centering_distance_m,
            self.maximum_advance_offset_m,
            self.peg_radius_rms_m,
            self.tray_plane_rms_m,
            self.tray_plane_radius_m,
            self.tray_plane_inlier_fraction,
            self.right_linearity_ratio,
        )
        if not np.isfinite(scalars).all():
            raise ValueError("task estimate metrics must be finite")
        if (
            self.approach_distance_m <= 0.0
            or self.surface_distance_m <= 0.0
            or self.lateral_centering_distance_m < 0.0
            or self.maximum_advance_offset_m <= 0.0
        ):
            raise ValueError("task estimate distances must be positive")
        if not 0.0 < self.tray_plane_inlier_fraction <= 1.0:
            raise ValueError("tray plane inlier fraction must be within (0, 1]")
        if self.tray_plane_inliers <= 0 or self.tray_roi_points < self.tray_plane_inliers:
            raise ValueError("tray plane counts are inconsistent")

    def summary_record(self) -> dict[str, object]:
        return {
            "source": self.source,
            "basis_world": self.basis_world.tolist(),
            "approach_bearing_world": self.approach_bearing_world.tolist(),
            "peg_axis_proxy_world": self.peg_axis_proxy_world.tolist(),
            "tray_normal_proxy_world": self.tray_normal_proxy_world.tolist(),
            "peg_grasp_center_world": self.peg_grasp_center_world.tolist(),
            "tray_grasp_center_world": self.tray_grasp_center_world.tolist(),
            "peg_insert_end_world": self.peg_insert_end_world.tolist(),
            "tray_entry_center_world": self.tray_entry_center_world.tolist(),
            "axis_alignment_cosine": float(self.axis_alignment_cosine),
            "tray_alignment_cosine": float(self.tray_alignment_cosine),
            "approach_distance_m": float(self.approach_distance_m),
            "surface_distance_m": float(self.surface_distance_m),
            "lateral_centering_distance_m": float(
                self.lateral_centering_distance_m
            ),
            "maximum_advance_offset_m": float(self.maximum_advance_offset_m),
            "peg_radius_rms_m": float(self.peg_radius_rms_m),
            "tray_plane_rms_m": float(self.tray_plane_rms_m),
            "tray_plane_radius_m": float(self.tray_plane_radius_m),
            "tray_plane_inlier_fraction": float(self.tray_plane_inlier_fraction),
            "tray_plane_inliers": int(self.tray_plane_inliers),
            "tray_roi_points": int(self.tray_roi_points),
            "right_linearity_ratio": float(self.right_linearity_ratio),
            "limitations": (
                "fixed ego RGB-D calibration, proprioceptive tray ROI, and "
                "single-plane footprint center as a coarse hole proxy"
            ),
        }


def estimate_known_geometry_from_pads_and_points(
    right_pads_world: np.ndarray,
    left_pads_world: np.ndarray,
    scene_points_world: np.ndarray,
) -> KinematicTaskEstimate:
    """Fit the round8 peg and the tray's visible top plane."""
    right = np.asarray(right_pads_world, dtype=np.float64)
    left = np.asarray(left_pads_world, dtype=np.float64)
    if right.shape != (4, 3) or left.shape != (4, 3):
        raise ValueError("right and left pads must each have shape (4, 3)")
    if not np.isfinite(right).all() or not np.isfinite(left).all():
        raise ValueError("pad positions must be finite")

    peg_axis, right_linearity = _principal_line(right[:3], name="right peg")
    right_finger_center = np.mean(right[:3], axis=0)
    peg_opposition = _perpendicular(right[3] - right_finger_center, peg_axis)
    if float(np.linalg.norm(peg_opposition)) < _MIN_OPPOSITION_M:
        raise UnobservableTaskFrame("right peg opposition is too short")
    peg_center = 0.5 * (right_finger_center + right[3])
    radial_distances = np.linalg.norm(
        np.stack([_perpendicular(point - peg_center, peg_axis) for point in right]),
        axis=1,
    )
    peg_radius_rms = float(
        np.sqrt(np.mean((radial_distances - _PEG_EFFECTIVE_CONTACT_RADIUS_M) ** 2))
    )
    tray_grasp_center = np.mean(left, axis=0)
    (
        tray_entry,
        tray_surface_normal,
        tray_tangent,
        tray_plane_rms,
        tray_plane_inliers,
        tray_roi_points,
        tray_plane_inlier_fraction,
        tray_plane_radius,
    ) = _fit_tray_plane_ransac(
        scene_points_world,
        roi_center_world=tray_grasp_center,
        peg_center_world=peg_center,
    )

    if float(peg_axis @ (tray_entry - peg_center)) < 0.0:
        peg_axis = -peg_axis
    peg_insert_end = peg_center + _PEG_CENTER_TO_INSERT_END_M * peg_axis
    approach = tray_entry - peg_insert_end
    approach_distance = float(np.linalg.norm(approach))
    if approach_distance <= 1e-6:
        raise UnobservableTaskFrame(
            f"coarse approach distance is degenerate: {approach_distance:.4f}m"
        )
    approach_axis = approach / approach_distance
    tray_inward_normal = -tray_surface_normal
    tray_alignment = float(approach_axis @ tray_inward_normal)
    surface_distance = float(approach @ tray_inward_normal)
    if surface_distance <= 1e-6:
        raise UnobservableTaskFrame(
            f"peg tip is not outside the tray plane: {surface_distance:.4f}m"
        )
    lateral_centering_distance = float(
        np.linalg.norm(approach - surface_distance * tray_inward_normal)
    )
    x_axis = _unit(
        tray_tangent
        - tray_inward_normal * float(tray_inward_normal @ tray_tangent),
        name="task x axis",
    )
    y_axis = _unit(np.cross(tray_inward_normal, x_axis), name="task y axis")
    x_axis = _unit(np.cross(y_axis, tray_inward_normal), name="task x axis")
    basis = np.column_stack([x_axis, y_axis, tray_inward_normal])
    return KinematicTaskEstimate(
        basis_world=basis,
        approach_bearing_world=approach_axis,
        peg_axis_proxy_world=peg_axis,
        tray_normal_proxy_world=tray_inward_normal,
        peg_grasp_center_world=peg_center,
        tray_grasp_center_world=tray_grasp_center,
        peg_insert_end_world=peg_insert_end,
        tray_entry_center_world=tray_entry,
        axis_alignment_cosine=float(peg_axis @ tray_inward_normal),
        tray_alignment_cosine=tray_alignment,
        approach_distance_m=approach_distance,
        surface_distance_m=surface_distance,
        lateral_centering_distance_m=lateral_centering_distance,
        maximum_advance_offset_m=surface_distance + _PEG_CENTER_TO_INSERT_END_M,
        peg_radius_rms_m=peg_radius_rms,
        tray_plane_rms_m=tray_plane_rms,
        tray_plane_radius_m=tray_plane_radius,
        tray_plane_inlier_fraction=tray_plane_inlier_fraction,
        tray_plane_inliers=tray_plane_inliers,
        tray_roi_points=tray_roi_points,
        right_linearity_ratio=right_linearity,
    )


def estimate_task_frame(
    state46: np.ndarray,
    ego_depth_m: np.ndarray,
) -> KinematicTaskEstimate:
    """Estimate one frozen frame from public proprioception and ego depth."""
    pads, _ = fingertip_pads_world(state46)
    scene_points = ego_depth_to_world(ego_depth_m)
    return estimate_known_geometry_from_pads_and_points(
        pads[0],
        pads[1],
        scene_points,
    )
