"""Privileged geometry labels used to train the post-grasp cerebellum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from hybrid_insert.assembly_contacts import AssemblyContactLabeler

from .privileged import PrivilegedAssemblyPrimitiveProvider


def _matrix3(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(3, 3)


def _hole_frame_rotation(socket_xmat: np.ndarray, hole_axis_world: np.ndarray) -> np.ndarray:
    """Build a right-handed hole frame whose +Z points out of the opening."""
    socket_rotation = _matrix3(socket_xmat)
    z_axis = np.array(hole_axis_world, dtype=np.float64, copy=True)
    z_axis /= np.linalg.norm(z_axis) + 1e-8
    x_hint = socket_rotation[:, 0]
    x_axis = x_hint - z_axis * np.dot(x_hint, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        x_hint = socket_rotation[:, 1]
        x_axis = x_hint - z_axis * np.dot(x_hint, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-8
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-8
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def relative_pose(
    reference_position_world: np.ndarray,
    reference_rotation_world: np.ndarray,
    target_position_world: np.ndarray,
    target_rotation_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target position and rotvec in the reference coordinate frame."""
    reference_position = np.asarray(reference_position_world, dtype=np.float64)
    reference_rotation = _matrix3(reference_rotation_world)
    target_position = np.asarray(target_position_world, dtype=np.float64)
    target_rotation = _matrix3(target_rotation_world)
    position = reference_rotation.T @ (target_position - reference_position)
    rotation = reference_rotation.T @ target_rotation
    return position, Rotation.from_matrix(rotation).as_rotvec()


@dataclass(frozen=True)
class GeometryPriorFrame:
    family_id: str
    peg_tip_world: np.ndarray
    peg_axis_world: np.ndarray
    hole_entry_world: np.ndarray
    hole_axis_world: np.ndarray
    peg_tip_in_hole_position: np.ndarray
    peg_in_hole_rotvec: np.ndarray
    peg_in_right_palm_position: np.ndarray
    peg_in_right_palm_rotvec: np.ndarray
    tray_in_left_palm_position: np.ndarray
    tray_in_left_palm_rotvec: np.ndarray
    lateral_error_m: float
    axis_error_rad: float
    approach_height_m: float
    insertion_depth_m: float
    target_depth_m: float
    nominal_peg_size_m: float
    peg_ok: bool
    tray_ok: bool
    insert_ok: bool
    peg_contact_count: int
    tray_contact_count: int


class PrivilegedGeometryLabeler:
    """Compute explicit assembly and hand-object geometry after each replay step."""

    _RIGHT_PALM = "allegro_palm_right"
    _LEFT_PALM = "allegro_palm_left"

    def __init__(self, raw_env) -> None:
        self._provider = PrivilegedAssemblyPrimitiveProvider(raw_env)
        self._contacts = AssemblyContactLabeler(raw_env)
        model = raw_env._model
        names = self._provider.names
        self._peg_body_id = int(model.body(names.peg_body).id)
        self._tray_body_id = int(model.body(names.socket_body).id)
        self._socket_site_id = int(model.site(names.socket_site).id)
        self._right_palm_id = int(model.body(self._RIGHT_PALM).id)
        self._left_palm_id = int(model.body(self._LEFT_PALM).id)

    def reset_reference(self, raw_env) -> None:
        self._contacts.reset_reference(raw_env)

    def compute(self, raw_env) -> GeometryPriorFrame:
        data = raw_env._data
        primitives = self._provider.snapshot(raw_env)
        outcome = self._contacts.compute(raw_env)
        peg_rotation = _matrix3(data.xmat[self._peg_body_id])
        tray_rotation = _matrix3(data.xmat[self._tray_body_id])
        hole_rotation = _hole_frame_rotation(
            data.site_xmat[self._socket_site_id],
            primitives.hole_axis_world,
        )
        peg_tip_in_hole, peg_in_hole_rotvec = relative_pose(
            primitives.hole_entry_world,
            hole_rotation,
            primitives.peg_tip_world,
            peg_rotation,
        )
        peg_in_right_palm, peg_in_right_palm_rotvec = relative_pose(
            data.xpos[self._right_palm_id],
            data.xmat[self._right_palm_id],
            data.xpos[self._peg_body_id],
            peg_rotation,
        )
        tray_in_left_palm, tray_in_left_palm_rotvec = relative_pose(
            data.xpos[self._left_palm_id],
            data.xmat[self._left_palm_id],
            data.xpos[self._tray_body_id],
            tray_rotation,
        )
        return GeometryPriorFrame(
            family_id=primitives.family_id,
            peg_tip_world=primitives.peg_tip_world.copy(),
            peg_axis_world=primitives.peg_axis_world.copy(),
            hole_entry_world=primitives.hole_entry_world.copy(),
            hole_axis_world=primitives.hole_axis_world.copy(),
            peg_tip_in_hole_position=peg_tip_in_hole,
            peg_in_hole_rotvec=peg_in_hole_rotvec,
            peg_in_right_palm_position=peg_in_right_palm,
            peg_in_right_palm_rotvec=peg_in_right_palm_rotvec,
            tray_in_left_palm_position=tray_in_left_palm,
            tray_in_left_palm_rotvec=tray_in_left_palm_rotvec,
            lateral_error_m=primitives.lateral_error_m,
            axis_error_rad=primitives.axis_error_rad,
            approach_height_m=primitives.approach_height_m,
            insertion_depth_m=primitives.insertion_depth_m,
            target_depth_m=primitives.target_depth_m,
            nominal_peg_size_m=primitives.nominal_peg_size_m,
            peg_ok=outcome.peg_ok,
            tray_ok=outcome.tray_ok,
            insert_ok=outcome.insert_ok,
            peg_contact_count=outcome.peg_contact_count,
            tray_contact_count=outcome.tray_contact_count,
        )
