# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

from embodichain.agents.agentchord.monitor_utils import (
    _as_pose_matrix,
    _get_object_pose,
    get_gripper_distance,
)
import numpy as np
import torch

__all__ = [
    "monitor_object_held",
    "monitor_object_moved",
]


def monitor_object_moved(
    env,
    obj_name: str,
    last_frame_pose: torch.Tensor | np.ndarray | list | tuple | dict = None,
    threshold: float = 0.01,
    **kwargs,
) -> bool:
    """Trigger when an object moved from the last frame beyond a threshold.

    Args:
        env: The current agent environment.
        obj_name: Target rigid object name.
        last_frame_pose: Previous-frame pose or a state dict returned by
            :func:`capture_object_state`.
        threshold: Maximum allowed translation change in meters.

    Returns:
        ``True`` if the monitored failure occurs, i.e. the object moved more than
        the threshold.
    """
    if last_frame_pose is None:
        last_frame_pose = env.obj_info.get(obj_name).get("pose")
    current_pose = _get_object_pose(env, obj_name)
    previous_pose = _as_pose_matrix(last_frame_pose, device=current_pose.device)
    movement = torch.norm(current_pose[:3, 3] - previous_pose[:3, 3]).item()
    return movement > threshold


def monitor_object_held(
    env,
    robot_name: str,
    obj_name: str | None = None,
    threshold: float = 0.01,
    **kwargs,
) -> bool:
    """Trigger when the selected gripper appears to have lost its held object.

    The monitor uses the current gripper opening distance instead of object-arm
    distance. It returns ``True`` when the opening is below ``threshold``, which
    indicates the gripper has closed too far and likely no object remains between
    the fingers.

    Args:
        env: The current agent environment.
        robot_name: Arm identifier containing ``"left"`` or ``"right"``.
        obj_name: Optional object name kept for recovery graph compatibility.
        threshold: Minimum expected gripper opening distance while holding.

    Returns:
        ``True`` if the monitored failure occurs, i.e. hold loss is detected.
    """
    return get_gripper_distance(env, robot_name) < threshold
