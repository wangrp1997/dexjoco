"""Replay zarr demos and record per-step kinematics + contacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

from interaction_retarget.constants import (
    LEFT_HAND_BODIES,
    NUM_HAND_KEYPOINTS,
    PEG_BODY,
    RIGHT_HAND_BODIES,
    TRAY_BODY,
)
from interaction_retarget.sim.contact import AssemblyContactDetector, FrameContact
from interaction_retarget.sim.hand_geom import hand_keypoints_world


def rotvec_dual_arm_to_policy(action44: np.ndarray) -> np.ndarray:
    action44 = np.asarray(action44, dtype=np.float64).reshape(-1)
    if action44.shape[0] != 44:
        raise ValueError(f"Expected 44-d rotvec action, got {action44.shape}")
    r_xyz, r_rot, r_hand = action44[0:3], action44[3:6], action44[6:22]
    l_xyz, l_rot, l_hand = action44[22:25], action44[25:28], action44[28:44]
    r_quat = R.from_rotvec(r_rot).as_quat(scalar_first=True)
    l_quat = R.from_rotvec(l_rot).as_quat(scalar_first=True)
    return np.concatenate([r_xyz, r_quat, l_xyz, l_quat, r_hand, l_hand], axis=0)


def policy_dual_arm_to_raw(action46: np.ndarray) -> dict[str, np.ndarray]:
    action46 = np.asarray(action46, dtype=np.float64).reshape(-1)
    if action46.shape[0] != 46:
        raise ValueError(f"Expected 46-d policy action, got {action46.shape}")
    right_pose, left_pose = action46[0:7], action46[7:14]
    right_hand, left_hand = action46[14:30], action46[30:46]
    return {
        "right": np.concatenate([right_pose, right_hand], axis=0),
        "left": np.concatenate([left_pose, left_hand], axis=0),
    }


def raw_flat_to_dict(action_flat: np.ndarray) -> dict[str, np.ndarray]:
    """Convert stored zarr action to raw bimanual env dict."""
    action_flat = np.asarray(action_flat, dtype=np.float64).reshape(-1)
    if action_flat.shape[0] == 46:
        # record_demos_zarr stores [right(23), left(23)] per arm pose+hand.
        return {
            "right": action_flat[:23],
            "left": action_flat[23:46],
        }
    if action_flat.shape[0] == 44:
        return policy_dual_arm_to_raw(rotvec_dual_arm_to_policy(action_flat))
    raise ValueError(f"Unsupported action dim {action_flat.shape[0]}")


def make_assembly_env(*, seed: int, randomize: bool = False, render_mode: str = "rgb_array"):
    config = CONFIG_MAPPING["bimanual_assembly"]()
    return config.get_environment(
        policy_mode=True,
        render_mode=render_mode,
        randomize=randomize,
        randomize_dynamics=False,
        seed=seed,
    )


@dataclass
class ReplayStep:
    contact: FrameContact
    tray_pos: np.ndarray
    tray_quat: np.ndarray
    peg_pos: np.ndarray
    peg_quat: np.ndarray
    left_hand_world: np.ndarray
    right_hand_world: np.ndarray
    left_gripper_speed: float
    right_gripper_speed: float
    tray_z: float
    peg_z: float


@dataclass
class ReplayTrace:
    steps: list[ReplayStep] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


def _body_pos_quat(data, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    return data.xpos[body_id].copy(), data.xquat[body_id].copy()


def _gripper_speed(data, qpos_adrs: np.ndarray, qvel_adrs: np.ndarray) -> float:
    qvel = data.qvel[qvel_adrs]
    return float(np.linalg.norm(qvel))


def replay_episode(
    actions: np.ndarray,
    *,
    seed: int,
    initial_state: np.ndarray | None = None,
    randomize: bool = False,
) -> ReplayTrace:
    env = make_assembly_env(seed=seed, randomize=randomize)
    raw_env = env.unwrapped
    model = raw_env._model
    detector = AssemblyContactDetector(raw_env)
    config = CONFIG_MAPPING["bimanual_assembly"]()

    left_qpos_adr = raw_env._allegro_dof_left_ids
    right_qpos_adr = raw_env._allegro_dof_right_ids
    left_qvel_adr = np.asarray(
        [int(model.joint(n).dofadr) for n in raw_env._allegro_joint_left_names], dtype=int
    )
    right_qvel_adr = np.asarray(
        [int(model.joint(n).dofadr) for n in raw_env._allegro_joint_right_names], dtype=int
    )
    tray_id = model.body(TRAY_BODY).id
    peg_id = model.body(PEG_BODY).id

    trace = ReplayTrace()
    try:
        env.reset()
        if initial_state is not None and has_restorer("bimanual_assembly"):
            restore_initial_state(env, "bimanual_assembly", config, initial_state)
        detector.reset_reference(raw_env)

        for action in actions:
            raw_env.step(raw_flat_to_dict(action))
            data = raw_env._data
            contact = detector.compute(raw_env)
            tray_pos, tray_quat = _body_pos_quat(data, tray_id)
            peg_pos, peg_quat = _body_pos_quat(data, peg_id)
            trace.steps.append(
                ReplayStep(
                    contact=contact,
                    tray_pos=tray_pos,
                    tray_quat=tray_quat,
                    peg_pos=peg_pos,
                    peg_quat=peg_quat,
                    left_hand_world=hand_keypoints_world(model, data, LEFT_HAND_BODIES),
                    right_hand_world=hand_keypoints_world(model, data, RIGHT_HAND_BODIES),
                    left_gripper_speed=_gripper_speed(data, left_qpos_adr, left_qvel_adr),
                    right_gripper_speed=_gripper_speed(data, right_qpos_adr, right_qvel_adr),
                    tray_z=float(tray_pos[2]),
                    peg_z=float(peg_pos[2]),
                )
            )

        trace.info = {
            "num_steps": len(trace.steps),
            "seed": int(seed),
            "used_initial_state": initial_state is not None,
        }
        return trace
    finally:
        env.close()
