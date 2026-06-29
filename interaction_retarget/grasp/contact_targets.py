"""MuJoCo hand–object contact targets in object frame (DexGraspBench / Dexonomy refs).

Refs:
  - interaction_retarget/tpsr/grasp_filter.py ``hand_object_contacts`` (Dexonomy)
  - refs/DexGraspBench/src/util/hand_util.py ``get_contact_info``
  - refs/GenHand/optimisation/icp.py (linear assignment on contact anchors)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from interaction_retarget.constants import PEG_BODY, TRAY_BODY
from interaction_retarget.transforms import quat_wxyz_to_matrix, world_to_object
from interaction_retarget.tpsr.grasp_filter import hand_object_contacts

Side = Literal["left", "right"]
ObjectName = Literal["tray", "peg"]


def _object_body(object_name: ObjectName) -> str:
    return TRAY_BODY if object_name == "tray" else PEG_BODY


def object_pose_world(raw_env, object_name: ObjectName) -> tuple[np.ndarray, np.ndarray]:
    bid = int(raw_env._model.body(_object_body(object_name)).id)
    data = raw_env._data
    return (
        np.asarray(data.xpos[bid], dtype=np.float64).copy(),
        np.asarray(data.xquat[bid], dtype=np.float64).copy(),
    )


def rotate_world_to_object(vectors_world: np.ndarray, obj_quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotate world-frame directions into object body frame."""
    vecs = np.asarray(vectors_world, dtype=np.float64).reshape(-1, 3)
    rot = quat_wxyz_to_matrix(obj_quat_wxyz)
    return vecs @ rot


@dataclass(frozen=True)
class ContactTargetSet:
    """Privileged demo contacts stored in object frame."""

    hand_bodies: tuple[str, ...]
    object_bodies: tuple[str, ...]
    pos_obj: np.ndarray  # (N, 3)
    normal_obj: np.ndarray  # (N, 3)
    source_episode_index: int = -1

    @property
    def count(self) -> int:
        return int(self.pos_obj.shape[0])

    def as_dict(self) -> dict[str, np.ndarray | list[str] | int]:
        return {
            "hand_bodies": list(self.hand_bodies),
            "object_bodies": list(self.object_bodies),
            "contact_pos_obj": np.asarray(self.pos_obj, dtype=np.float64),
            "contact_normal_obj": np.asarray(self.normal_obj, dtype=np.float64),
            "source_episode_index": int(self.source_episode_index),
        }


def record_contact_targets_obj(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    contact_dist_thre_m: float = 0.002,
    source_episode_index: int = -1,
) -> ContactTargetSet:
    """Record current MuJoCo HO contacts in object frame."""
    ho = hand_object_contacts(
        raw_env,
        side=side,
        object_name=object_name,
        contact_dist_thre_m=contact_dist_thre_m,
    )
    if ho["pos"].shape[0] == 0:
        return ContactTargetSet((), (), np.zeros((0, 3)), np.zeros((0, 3)), source_episode_index)

    obj_pos, obj_quat = object_pose_world(raw_env, object_name)
    pos_obj = world_to_object(ho["pos"], obj_pos, obj_quat)
    normal_obj = rotate_world_to_object(ho["normal"], obj_quat)
    norms = np.linalg.norm(normal_obj, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normal_obj = normal_obj / norms

    return ContactTargetSet(
        hand_bodies=tuple(str(x) for x in ho["bn1"]),
        object_bodies=tuple(str(x) for x in ho["bn2"]),
        pos_obj=pos_obj,
        normal_obj=normal_obj,
        source_episode_index=source_episode_index,
    )


def load_contact_targets_from_npz(data: dict) -> ContactTargetSet | None:
    pos = data.get("contact_pos_obj")
    if pos is None or np.asarray(pos).size == 0:
        return None
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    normal = np.asarray(data.get("contact_normal_obj", np.zeros_like(pos)), dtype=np.float64).reshape(-1, 3)
    hb = tuple(str(x) for x in data.get("contact_hand_bodies", data.get("hand_bodies", [])))
    ob = tuple(str(x) for x in data.get("contact_object_bodies", data.get("object_bodies", [])))
    if not hb:
        hb = tuple("" for _ in range(pos.shape[0]))
    if not ob:
        ob = tuple("" for _ in range(pos.shape[0]))
    src = int(data.get("source_episode_index", data.get("representative_episode_index", -1)))
    return ContactTargetSet(hb, ob, pos, normal, src)


def _current_contacts_obj(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    contact_dist_thre_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    ho = hand_object_contacts(
        raw_env,
        side=side,
        object_name=object_name,
        contact_dist_thre_m=contact_dist_thre_m,
    )
    if ho["pos"].shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    obj_pos, obj_quat = object_pose_world(raw_env, object_name)
    pos_obj = world_to_object(ho["pos"], obj_pos, obj_quat)
    normal_obj = rotate_world_to_object(ho["normal"], obj_quat)
    norms = np.linalg.norm(normal_obj, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return pos_obj, normal_obj / norms


def contact_match_rmse_m(
    raw_env,
    targets: ContactTargetSet,
    *,
    side: Side,
    object_name: ObjectName,
    contact_dist_thre_m: float = 0.002,
    w_normal: float = 0.003,
) -> tuple[float, int, int]:
    """Hungarian match current contacts to demo targets; return RMSE (m), n_cur, n_tgt."""
    cur_pos, cur_n = _current_contacts_obj(
        raw_env, side=side, object_name=object_name, contact_dist_thre_m=contact_dist_thre_m
    )
    tgt_pos = np.asarray(targets.pos_obj, dtype=np.float64).reshape(-1, 3)
    tgt_n = np.asarray(targets.normal_obj, dtype=np.float64).reshape(-1, 3)
    n_cur, n_tgt = int(cur_pos.shape[0]), int(tgt_pos.shape[0])
    if n_tgt == 0:
        return 0.0, n_cur, 0
    if n_cur == 0:
        return float("inf"), 0, n_tgt

    cost = np.zeros((n_cur, n_tgt), dtype=np.float64)
    for i in range(n_cur):
        dpos = np.linalg.norm(cur_pos[i : i + 1] - tgt_pos, axis=1)
        ndot = np.sum(cur_n[i : i + 1] * tgt_n, axis=1)
        ang = 1.0 - np.clip(ndot, -1.0, 1.0)
        cost[i] = dpos + float(w_normal) * ang

    row, col = linear_sum_assignment(cost)
    matched = cost[row, col]
    rmse = float(np.sqrt(np.mean(matched**2)))
    return rmse, n_cur, n_tgt


def contact_count_shortfall(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    min_contacts: int,
    contact_dist_thre_m: float = 0.002,
) -> float:
    ho = hand_object_contacts(
        raw_env,
        side=side,
        object_name=object_name,
        contact_dist_thre_m=contact_dist_thre_m,
    )
    n = int(ho["pos"].shape[0])
    return max(0, int(min_contacts) - n) * 0.01
