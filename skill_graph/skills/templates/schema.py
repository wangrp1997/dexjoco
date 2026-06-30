"""Grasp template schema (object-frame)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from skill_graph.constants import ObjectName, Side
from skill_graph.math.se3 import rotate_mocap_about_object_z, rotate_mocap_stack_about_object_z


@dataclass
class GraspTemplate:
    episode_index: int
    side: Side
    object_name: ObjectName
    demo_obj_pos: np.ndarray
    demo_obj_quat: np.ndarray
    grasp_mocap_pos_obj: np.ndarray
    grasp_mocap_quat_obj: np.ndarray
    grasp_hand: np.ndarray
    squeeze_mocap_pos_obj: np.ndarray
    squeeze_mocap_quat_obj: np.ndarray
    squeeze_hand: np.ndarray
    approach_mocap_pos_obj: np.ndarray
    approach_mocap_quat_obj: np.ndarray
    approach_hand: np.ndarray
    contact_pos_obj: np.ndarray
    contact_normal_obj: np.ndarray
    contact_force_obj: np.ndarray
    contact_hand_bodies: tuple[str, ...]
    contact_object_bodies: tuple[str, ...]
    export_contact_count: int
    zarr_path: str = ""

    @property
    def template_id(self) -> str:
        return f"ep{self.episode_index:03d}_{self.object_name}"

    def with_object_yaw(self, yaw_rad: float) -> "GraspTemplate":
        """Cylinder symmetry: rotate mocap poses around peg local Z."""
        if abs(float(yaw_rad)) < 1e-9:
            return self
        g_pos, g_quat = rotate_mocap_about_object_z(
            self.grasp_mocap_pos_obj, self.grasp_mocap_quat_obj, yaw_rad
        )
        s_pos, s_quat = rotate_mocap_about_object_z(
            self.squeeze_mocap_pos_obj, self.squeeze_mocap_quat_obj, yaw_rad
        )
        a_pos, a_quat = rotate_mocap_stack_about_object_z(
            self.approach_mocap_pos_obj, self.approach_mocap_quat_obj, yaw_rad
        )
        return replace(
            self,
            grasp_mocap_pos_obj=g_pos,
            grasp_mocap_quat_obj=g_quat,
            squeeze_mocap_pos_obj=s_pos,
            squeeze_mocap_quat_obj=s_quat,
            approach_mocap_pos_obj=a_pos,
            approach_mocap_quat_obj=a_quat,
        )

    def save_npz(self, path) -> None:
        np.savez_compressed(
            path,
            episode_index=self.episode_index,
            side=self.side,
            object_name=self.object_name,
            demo_obj_pos=self.demo_obj_pos,
            demo_obj_quat=self.demo_obj_quat,
            grasp_mocap_pos_obj=self.grasp_mocap_pos_obj,
            grasp_mocap_quat_obj=self.grasp_mocap_quat_obj,
            grasp_hand=self.grasp_hand,
            squeeze_mocap_pos_obj=self.squeeze_mocap_pos_obj,
            squeeze_mocap_quat_obj=self.squeeze_mocap_quat_obj,
            squeeze_hand=self.squeeze_hand,
            approach_mocap_pos_obj=self.approach_mocap_pos_obj,
            approach_mocap_quat_obj=self.approach_mocap_quat_obj,
            approach_hand=self.approach_hand,
            contact_pos_obj=self.contact_pos_obj,
            contact_normal_obj=self.contact_normal_obj,
            contact_force_obj=self.contact_force_obj,
            contact_hand_bodies=np.asarray(self.contact_hand_bodies, dtype=object),
            contact_object_bodies=np.asarray(self.contact_object_bodies, dtype=object),
            export_contact_count=self.export_contact_count,
            zarr_path=self.zarr_path,
        )

    @classmethod
    def load_npz(cls, path) -> "GraspTemplate":
        d = np.load(path, allow_pickle=True)
        return cls(
            episode_index=int(d["episode_index"]),
            side=str(d["side"]),
            object_name=str(d["object_name"]),
            demo_obj_pos=np.asarray(d["demo_obj_pos"], dtype=np.float64),
            demo_obj_quat=np.asarray(d["demo_obj_quat"], dtype=np.float64),
            grasp_mocap_pos_obj=np.asarray(d["grasp_mocap_pos_obj"], dtype=np.float64),
            grasp_mocap_quat_obj=np.asarray(d["grasp_mocap_quat_obj"], dtype=np.float64),
            grasp_hand=np.asarray(d["grasp_hand"], dtype=np.float64),
            squeeze_mocap_pos_obj=np.asarray(d["squeeze_mocap_pos_obj"], dtype=np.float64),
            squeeze_mocap_quat_obj=np.asarray(d["squeeze_mocap_quat_obj"], dtype=np.float64),
            squeeze_hand=np.asarray(d["squeeze_hand"], dtype=np.float64),
            approach_mocap_pos_obj=np.asarray(d["approach_mocap_pos_obj"], dtype=np.float64),
            approach_mocap_quat_obj=np.asarray(d["approach_mocap_quat_obj"], dtype=np.float64),
            approach_hand=np.asarray(d["approach_hand"], dtype=np.float64),
            contact_pos_obj=np.asarray(d["contact_pos_obj"], dtype=np.float64),
            contact_normal_obj=np.asarray(d["contact_normal_obj"], dtype=np.float64),
            contact_force_obj=np.asarray(d["contact_force_obj"], dtype=np.float64),
            contact_hand_bodies=tuple(str(x) for x in d["contact_hand_bodies"].tolist()),
            contact_object_bodies=tuple(str(x) for x in d["contact_object_bodies"].tolist()),
            export_contact_count=int(d["export_contact_count"]),
            zarr_path=str(d.get("zarr_path", "")),
        )
