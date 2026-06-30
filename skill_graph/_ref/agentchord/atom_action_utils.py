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

import numpy as np
import torch
import ast
from functools import partial
from typing import Any, List

from embodichain.utils.logger import log_error, log_warning
from embodichain.lab.gym.utils.misc import mul_linear_expand
from embodichain.lab.sim.planners import (
    MoveType,
    PlanState,
    MotionGenerator,
    MotionGenCfg,
    MotionGenOptions,
    ToppraPlanOptions,
    ToppraPlannerCfg,
)

__all__ = [
    "apply_offset_to_pose",
    "convert_action_from_real_to_sim",
    "convert_action_from_sim_to_real",
    "draw_axis",
    "extract_drive_calls",
    "finalize_actions",
    "find_nearest_valid_pose",
    "get_function_name",
    "get_arm_states",
    "get_qpos",
    "is_real_backend",
    "plan_gripper_trajectory",
    "plan_trajectory",
    "resolve_action",
    "resolve_object_pose",
    "set_qpos",
    "sim_action_dim",
    "sync_agent_state_from_sim_action",
    "sync_agent_state_from_robot",
]


SIM_LEFT_ARM = [0, 2, 4, 6, 8, 10]
SIM_RIGHT_ARM = [1, 3, 5, 7, 9, 11]
SIM_LEFT_GRIPPER = 12
SIM_LEFT_GRIPPER_MIMIC = 13
SIM_RIGHT_GRIPPER = 14
SIM_RIGHT_GRIPPER_MIMIC = 15

REAL_LEFT_ARM = [0, 1, 2, 3, 4, 5]
REAL_LEFT_GRIPPER = 6
REAL_RIGHT_ARM = [7, 8, 9, 10, 11, 12]
REAL_RIGHT_GRIPPER = 13


def is_real_backend(env=None, kwargs: dict[str, Any] | None = None) -> bool:
    kwargs = kwargs or {}
    if kwargs.get("real_world") or kwargs.get("action_backend") == "real":
        return True
    backend = getattr(env, "backend", None)
    return bool(
        backend == "real"
        or getattr(env, "is_real", False)
        or getattr(env, "real_world", False)
    )


def resolve_object_pose(env, obj_name: str, robot_name: str, kwargs, reference=None):
    """Resolve an object pose from simulator state or real perception."""
    if kwargs.get("target_obj_pose") is not None:
        return kwargs["target_obj_pose"]

    if (
        is_real_backend(env, kwargs)
        or kwargs.get("use_perception")
        or kwargs.get("use_perception_pose")
    ):
        from embodichain.deploy.perception.pose_estimation import (
            get_obj_pose_from_perception,
        )

        pose = get_obj_pose_from_perception(env, obj_name, robot_name, kwargs)
        return pose

    obj_uids = env.sim.get_rigid_object_uid_list()
    if obj_name in obj_uids:
        target_obj = env.sim.get_rigid_object(obj_name)
    else:
        log_error(f"No matched object {obj_uids}.")
    return target_obj.get_local_pose(to_matrix=True).squeeze(0)


def convert_action_from_sim_to_real(sim_qpos: np.ndarray) -> np.ndarray:
    """Map AgentChord's 16D sim action layout to CobotMagic's 14D real layout."""
    sim_qpos = np.asarray(sim_qpos, dtype=np.float32).reshape(-1)
    if sim_qpos.shape[0] != 16:
        raise ValueError(f"Expected sim_qpos shape (16,), got {sim_qpos.shape}.")
    real_qpos = np.zeros(14, dtype=sim_qpos.dtype)
    real_qpos[REAL_LEFT_ARM] = sim_qpos[SIM_LEFT_ARM]
    real_qpos[REAL_LEFT_GRIPPER] = sim_qpos[SIM_LEFT_GRIPPER]
    real_qpos[REAL_RIGHT_ARM] = sim_qpos[SIM_RIGHT_ARM]
    real_qpos[REAL_RIGHT_GRIPPER] = sim_qpos[SIM_RIGHT_GRIPPER]
    return real_qpos


def convert_action_from_real_to_sim(real_qpos: np.ndarray) -> np.ndarray:
    """Map CobotMagic's 14D real layout back to AgentChord's 16D sim layout."""
    real_qpos = np.asarray(real_qpos, dtype=np.float32).reshape(-1)
    if real_qpos.shape[0] != 14:
        raise ValueError(f"Expected real_qpos shape (14,), got {real_qpos.shape}.")
    sim_qpos = np.zeros(16, dtype=real_qpos.dtype)
    sim_qpos[SIM_LEFT_ARM] = real_qpos[REAL_LEFT_ARM]
    sim_qpos[SIM_LEFT_GRIPPER] = real_qpos[REAL_LEFT_GRIPPER]
    sim_qpos[SIM_LEFT_GRIPPER_MIMIC] = real_qpos[REAL_LEFT_GRIPPER]
    sim_qpos[SIM_RIGHT_ARM] = real_qpos[REAL_RIGHT_ARM]
    sim_qpos[SIM_RIGHT_GRIPPER] = real_qpos[REAL_RIGHT_GRIPPER]
    sim_qpos[SIM_RIGHT_GRIPPER_MIMIC] = real_qpos[REAL_RIGHT_GRIPPER]
    return sim_qpos


def set_qpos(env, qpos: np.ndarray, wait: bool = True, interp_num: int = 0) -> None:
    """Send one normalized 14D real action to the physical controller."""
    controller = getattr(env, "controller", env)
    cmd = np.asarray(qpos, dtype=np.float32).copy()
    left_range = getattr(controller, "left_gripper_joint_limits", (0.0, 1.0))
    right_range = getattr(controller, "right_gripper_joint_limits", (0.0, 1.0))
    cmd[REAL_LEFT_GRIPPER] = (
        left_range[0] + (left_range[1] - left_range[0]) * cmd[REAL_LEFT_GRIPPER]
    )
    cmd[REAL_RIGHT_GRIPPER] = (
        right_range[0] + (right_range[1] - right_range[0]) * cmd[REAL_RIGHT_GRIPPER]
    )
    controller.set_current_qpos(cmd, interp_num=interp_num, wait=wait)


def sync_agent_state_from_sim_action(env, sim_action: np.ndarray) -> None:
    env.left_arm_current_qpos = sim_action[env.left_arm_joints]
    env.right_arm_current_qpos = sim_action[env.right_arm_joints]
    env.left_arm_current_gripper_state = np.array(
        [sim_action[env.left_eef_joints][0]],
        dtype=np.float32,
    )
    env.right_arm_current_gripper_state = np.array(
        [sim_action[env.right_eef_joints][0]],
        dtype=np.float32,
    )
    if hasattr(env, "get_arm_fk"):
        env.left_arm_current_xpos = env.get_arm_fk(
            qpos=env.left_arm_current_qpos,
            is_left=True,
        )
        env.right_arm_current_xpos = env.get_arm_fk(
            qpos=env.right_arm_current_qpos,
            is_left=False,
        )


def get_function_name(function) -> str:
    if isinstance(function, partial):
        return getattr(function.func, "__name__", function.__class__.__name__)
    return getattr(function, "__name__", function.__class__.__name__)


def sim_action_dim(env) -> int:
    if hasattr(env, "init_qpos"):
        return len(env.init_qpos)
    robot = getattr(env, "robot", None)
    if robot is not None and hasattr(robot, "get_qpos"):
        return len(robot.get_qpos().squeeze(0))
    return 16


def draw_axis(env, pose):
    """Draw an axis marker in the simulation for debugging/visualization.

    Args:
        env: The simulation environment.
        pose: The pose (4x4 matrix) where to draw the axis.
    """
    from embodichain.lab.sim.cfg import MarkerCfg

    marker_cfg = MarkerCfg(
        name="test",
        marker_type="axis",
        axis_xpos=pose,
        axis_size=0.01,
        axis_len=0.2,
        arena_index=-1,  # All arenas
    )
    env.sim.draw_marker(cfg=marker_cfg)
    env.sim.update()


def get_arm_states(env, robot_name):
    """Get the current state of the specified robot arm.

    Args:
        env: The simulation environment.
        robot_name: Name of the robot arm (should contain "left" or "right").

    Returns:
        Tuple of (is_left, select_arm, current_qpos, current_pose, current_gripper_state):
            - is_left: bool, whether this is the left arm
            - select_arm: str, arm identifier ("left_arm" or "right_arm")
            - current_qpos: Current joint positions
            - current_pose: Current end-effector pose (4x4 matrix)
            - current_gripper_state: Current gripper state
    """
    left_arm_current_qpos, right_arm_current_qpos = env.get_current_qpos_agent()
    left_arm_current_pose, right_arm_current_pose = env.get_current_xpos_agent()
    (
        left_arm_current_gripper_state,
        right_arm_current_gripper_state,
    ) = env.get_current_gripper_state_agent()

    side = "right" if "right" in robot_name else "left"
    is_left = True if side == "left" else False
    select_arm = "left_arm" if is_left else "right_arm"

    arms = {
        "left": (
            left_arm_current_qpos,
            left_arm_current_pose,
            left_arm_current_gripper_state,
        ),
        "right": (
            right_arm_current_qpos,
            right_arm_current_pose,
            right_arm_current_gripper_state,
        ),
    }
    (
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = arms[side]

    return (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    )


def find_nearest_valid_pose(env, select_arm, pose, xpos_resolution=0.1):
    """Find the nearest valid pose using reachability validation.

    Args:
        env: The simulation environment.
        select_arm: Arm identifier ("left_arm" or "right_arm").
        pose: Target pose (4x4 matrix).
        xpos_resolution: Resolution for reachability checking.

    Returns:
        torch.Tensor: The nearest valid pose (4x4 matrix).
    """
    # use the validator to choose the nearest valid pose
    # delete the cache every time
    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()
    ret, _ = env.robot.compute_xpos_reachability(
        select_arm,
        pose,
        xpos_resolution=xpos_resolution,
        qpos_resolution=np.radians(60),
        cache_mode="disk",
        use_cached=False,
        visualize=False,
    )
    ret = np.stack(ret, axis=0)
    # find the nearest valid pose
    xyz = pose[:3, 3]
    ts = np.stack([M[:3, 3] for M in ret], axis=0)  # shape (N,3)
    dists = np.linalg.norm(ts - xyz[None, :], axis=1)
    best_idx = np.argmin(dists)
    nearest_valid_pose = ret[best_idx]
    return torch.from_numpy(nearest_valid_pose)


def get_qpos(env, is_left, select_arm, pose, qpos_seed, force_valid=False, name=""):
    """Solve inverse kinematics to get joint positions for a given pose.

    Args:
        env: The simulation environment.
        is_left: bool, whether this is the left arm.
        select_arm: Arm identifier ("left_arm" or "right_arm").
        pose: Target end-effector pose (4x4 matrix).
        qpos_seed: Seed joint positions for IK solving.
        force_valid: If True, use nearest valid pose if IK fails.
        name: Name for logging purposes.

    Returns:
        tuple[torch.Tensor | np.ndarray, torch.Tensor]:
            The actual pose used for IK solving and its corresponding joint positions.
    """
    solved_pose = pose

    if force_valid:
        try:
            ret, qpos = env.get_arm_ik(
                solved_pose, is_left=is_left, qpos_seed=qpos_seed
            )
            if not ret:
                log_error(f"Generate {name} qpos failed.\n")
        except Exception as e:
            log_warning(
                f"Original {name} pose invalid, using nearest valid pose. ({e})\n"
            )
            solved_pose = find_nearest_valid_pose(env, select_arm, solved_pose)

            ret, qpos = env.get_arm_ik(
                solved_pose, is_left=is_left, qpos_seed=qpos_seed
            )
    else:
        ret, qpos = env.get_arm_ik(solved_pose, is_left=is_left, qpos_seed=qpos_seed)
        if not ret:
            log_error(f"Generate {name} qpos failed.\n")

    return solved_pose, qpos


def plan_trajectory(
    env,
    select_arm,
    qpos_list,
    sample_num,
    select_arm_current_gripper_state,
    select_qpos_traj,
    ee_state_list_select,
):
    """Plan a trajectory between joint positions and append to trajectory lists.

    Args:
        env: The simulation environment.
        select_arm: Arm identifier ("left_arm" or "right_arm").
        qpos_list: List of joint positions to plan between.
        sample_num: Number of samples for trajectory interpolation.
        select_arm_current_gripper_state: Current gripper state.
        select_qpos_traj: List to append planned joint positions to (modified in-place).
        ee_state_list_select: List to append gripper states to (modified in-place).
    """
    motion_generator = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=env.robot.uid))
    )

    plan_state = [
        PlanState(qpos=torch.as_tensor(qpos), move_type=MoveType.JOINT_MOVE)
        for qpos in qpos_list
    ]

    ret = motion_generator.generate(
        target_states=plan_state,
        options=MotionGenOptions(
            control_part=select_arm,
            plan_opts=ToppraPlanOptions(
                sample_interval=sample_num,
            ),
        ),
    )

    select_qpos_traj.extend(ret.positions.numpy())
    ee_state_list_select.extend([select_arm_current_gripper_state] * len(ret.positions))


def plan_gripper_trajectory(
    env,
    is_left,
    sample_num,
    execute_open,
    select_arm_current_qpos,
    select_qpos_traj,
    ee_state_list_select,
):
    """Plan a gripper trajectory (opening or closing) and append to trajectory lists.

    Args:
        env: The simulation environment.
        is_left: bool, whether this is the left arm.
        sample_num: Number of samples for gripper motion.
        execute_open: bool, True for opening, False for closing.
        select_arm_current_qpos: Current joint positions.
        select_qpos_traj: List to append joint positions to (modified in-place).
        ee_state_list_select: List to append gripper states to (modified in-place).
    """
    open_state = env.open_state
    close_state = env.close_state

    if execute_open:
        ee_state_expand_select = np.array([close_state, open_state])
        env.set_current_gripper_state_agent(open_state, is_left=is_left)
    else:
        ee_state_expand_select = np.array([open_state, close_state])
        env.set_current_gripper_state_agent(close_state, is_left=is_left)

    ee_state_expand_select = mul_linear_expand(ee_state_expand_select, [sample_num])

    select_qpos_traj.extend([select_arm_current_qpos] * sample_num)
    ee_state_list_select.extend(ee_state_expand_select)


def finalize_actions(select_qpos_traj, ee_state_list_select):
    """Format trajectory data into action format.

    Args:
        select_qpos_traj: List of joint positions.
        ee_state_list_select: List of gripper states.

    Returns:
        np.ndarray: Formatted actions array with joint positions and gripper states.
    """
    # mimic eef state
    actions = np.concatenate(
        [
            np.array(select_qpos_traj),
            np.array(ee_state_list_select),
            np.array(ee_state_list_select),
        ],
        axis=-1,
    )
    return actions


def extract_drive_calls(code_str: str) -> List[str]:
    """Extract all drive() function calls from a code string.

    Args:
        code_str: Python code string to parse.

    Returns:
        List of code blocks containing drive() calls.
    """
    tree = ast.parse(code_str)
    lines = code_str.splitlines()

    drive_blocks = []

    for node in tree.body:
        # Match: drive(...)
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "drive"
        ):
            # AST line numbers are 1-based
            start = node.lineno - 1
            end = node.end_lineno
            block = "\n".join(lines[start:end])
            drive_blocks.append(block)

    return drive_blocks


def apply_offset_to_pose(pose, offset: list):
    pose[0, 3] += offset[0]
    pose[1, 3] += offset[1]
    pose[2, 3] += offset[2]
    return pose


def resolve_action(action, env, kwargs):
    if callable(action):
        return action(env=env, **kwargs)
    return action


def sync_agent_state_from_robot(env) -> None:
    """Synchronize cached agent arm states from the physical robot state."""
    action = env.robot.get_qpos().squeeze(0)
    env.left_arm_current_qpos = action[env.left_arm_joints]
    env.left_arm_current_xpos = env.robot.compute_fk(
        qpos=env.left_arm_current_qpos,
        name="left_arm",
        to_matrix=True,
    ).squeeze(0)
    env.left_arm_current_gripper_state = action[env.left_eef_joints][0].unsqueeze(0)

    env.right_arm_current_qpos = action[env.right_arm_joints]
    env.right_arm_current_xpos = env.robot.compute_fk(
        qpos=env.right_arm_current_qpos,
        name="right_arm",
        to_matrix=True,
    ).squeeze(0)
    env.right_arm_current_gripper_state = action[env.right_eef_joints][0].unsqueeze(0)
