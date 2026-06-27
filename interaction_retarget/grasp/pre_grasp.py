"""Pre-grasp pose derived from grasp configuration (GenHand simulation/robot_base.py).

GenHand defines a gripper-frame retreat offset per robot, e.g. Allegro
``pre_grasp = [-0.2, 0.0, -0.2]``, and uses it in ``simulation/trajectory.py``
before closing to grasp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation as R

from interaction_retarget.sim.settle import vec_to_arm_action

Side = Literal["left", "right"]

# refs/GenHand/simulation/robot_base.py :: allegro.pre_grasp (PyBullet hand frame, meters)
GENHAND_ALLEGRO_PRE_GRASP_OFFSET = np.asarray([-0.2, 0.0, -0.2], dtype=np.float64)

_ALLEGRO_OPEN = np.asarray(
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0), dtype=np.float64
)


@dataclass
class PreGraspAction:
    side: Side
    action23: np.ndarray
    offset_m: np.ndarray


def derive_pre_grasp_from_grasp(
    grasp_action23: np.ndarray,
    *,
    side: Side,
    offset_hand_frame: np.ndarray | None = None,
    open_hand: np.ndarray | None = None,
    offset_scale: float = 1.0,
) -> PreGraspAction:
    """Retreat mocap along GenHand Allegro offset; open fingers for approach."""
    grasp_action23 = vec_to_arm_action(grasp_action23)
    offset = (
        GENHAND_ALLEGRO_PRE_GRASP_OFFSET.copy()
        if offset_hand_frame is None
        else np.asarray(offset_hand_frame, dtype=np.float64).reshape(3)
    )
    offset = offset * float(offset_scale)

    pos = grasp_action23[0:3]
    quat_wxyz = grasp_action23[3:7]
    rot = R.from_quat(quat_wxyz[[1, 2, 3, 0]])
    pos_pre = pos + rot.apply(offset)

    hand_open = _ALLEGRO_OPEN.copy() if open_hand is None else np.asarray(open_hand, dtype=np.float64).reshape(16)
    action23 = np.concatenate([pos_pre, quat_wxyz, hand_open], axis=0)
    return PreGraspAction(side=side, action23=action23, offset_m=offset)
