"""Laplacian grasp IK: canonical δ* → arm action (MuJoCo + scipy).

Object-frame Laplacian follows holosoma ``interaction_mesh_retargeter`` (obj_frame=True).
Local/global hand terms follow pyroki ``examples/09_hand_retargeting.py``.
Demo alignment: palm seed = T_world_obj × hand_points_obj[0], then L-BFGS-B on
mocap pos/rot + Allegro joints to match δ* in the object frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from interaction_retarget.constants import (
    FINGERTIP_KEYPOINT_INDICES,
    LEFT_HAND_BODIES,
    MIN_GRASP_CONTACT_COUNT,
    PEG_BODY,
    RIGHT_HAND_BODIES,
    TRAY_BODY,
)
from interaction_retarget.grasp.distill import load_canonical_grasp
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.laplacian import laplacian_coordinates
from interaction_retarget.sim.hand_geom import hand_keypoints_world
from interaction_retarget.sim.replay import raw_flat_to_dict
from interaction_retarget.sim.settle import (
    read_arm_action,
    settle_bimanual_actions,
    settle_side_actions,
    vec_to_arm_action,
)
from interaction_retarget.sim.state import restore_sim, snapshot_sim
from interaction_retarget.transforms import (
    mocap_world_from_object_frame,
    object_to_world,
    world_to_object,
)

ObjectName = Literal["tray", "peg"]
Side = Literal["left", "right"]

_ALLEGRO_HOME = np.asarray(
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0), dtype=np.float64
)
_ALLEGRO_OPEN = np.zeros(16, dtype=np.float64)


@dataclass
class IkWeights:
    laplacian: float = 2.0
    global_hand: float = 120.0
    local_hand: float = 8.0
    site_tracking: float = 60.0
    joint_regularization: float = 0.01
    inactive_arm_hold: float = 0.05
    contact: float = 0.0
    contact_site: float = 0.0


def _contact_site_rmse_m(hand_obj: np.ndarray, contact_sites_obj: np.ndarray | None) -> float:
    """SPIDER mjwp contact_pos: demo sites vs fingertip keypoints in object frame."""
    sites = np.asarray(contact_sites_obj, dtype=np.float64).reshape(-1, 3) if contact_sites_obj is not None else None
    if sites is None or sites.size == 0:
        return 0.0
    tips = np.asarray(hand_obj, dtype=np.float64)[list(FINGERTIP_KEYPOINT_INDICES)]
    dists = [float(np.min(np.linalg.norm(tips - s, axis=1))) for s in sites]
    return float(np.sqrt(np.mean(np.square(dists))))


@dataclass
class GraspIkResult:
    object_name: str
    active_side: str
    action_right: np.ndarray
    action_left: np.ndarray
    cost: float
    laplacian_rmse_m: float
    hand_rmse_m: float
    contact_count: int
    contact_site_rmse_m: float
    success: bool


def _object_body(object_name: ObjectName) -> str:
    return TRAY_BODY if object_name == "tray" else PEG_BODY


def _active_side(object_name: ObjectName) -> Side:
    return "left" if object_name == "tray" else "right"


def _hand_bodies(side: Side) -> tuple[str, ...]:
    return LEFT_HAND_BODIES if side == "left" else RIGHT_HAND_BODIES


def _object_pose(raw_env, obj_body: str) -> tuple[np.ndarray, np.ndarray]:
    obj_id = raw_env._model.body(obj_body).id
    data = raw_env._data
    return (
        np.asarray(data.xpos[obj_id], dtype=np.float64),
        np.asarray(data.xquat[obj_id], dtype=np.float64),
    )


def mocap_world_from_canonical(
    raw_env, canonical: dict, *, object_name: ObjectName
) -> tuple[np.ndarray, np.ndarray]:
    """Grasp wrist target: T_world_obj × (mocap_pos_obj, mocap_quat_obj) from δ*."""
    obj_body = _object_body(object_name)
    obj_pos, obj_quat = _object_pose(raw_env, obj_body)
    mocap_pos_obj = canonical.get("mocap_pos_obj")
    mocap_quat_obj = canonical.get("mocap_quat_obj")
    if mocap_pos_obj is not None and mocap_quat_obj is not None:
        return mocap_world_from_object_frame(
            np.asarray(mocap_pos_obj, dtype=np.float64),
            np.asarray(mocap_quat_obj, dtype=np.float64),
            obj_pos,
            obj_quat,
        )
    palm_obj = np.asarray(canonical["hand_points_obj"], dtype=np.float64)[0:1]
    palm_world = object_to_world(palm_obj, obj_pos, obj_quat)[0]
    home = read_arm_action(raw_env, _active_side(object_name))
    return palm_world, np.asarray(home[3:7], dtype=np.float64)


def palm_world_from_canonical(raw_env, canonical: dict, *, object_name: ObjectName) -> np.ndarray:
    """Palm keypoint in world: T_world_obj × hand_points_obj[0] (TopoRetarget seed)."""
    obj_body = _object_body(object_name)
    obj_pos, obj_quat = _object_pose(raw_env, obj_body)
    palm_obj = np.asarray(canonical["hand_points_obj"], dtype=np.float64)[0]
    return object_to_world(palm_obj.reshape(1, 3), obj_pos, obj_quat)[0]


def _action23(pos: np.ndarray, quat_wxyz: np.ndarray, hand: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(pos, dtype=np.float64).reshape(3),
            np.asarray(quat_wxyz, dtype=np.float64).reshape(4),
            np.asarray(hand, dtype=np.float64).reshape(16),
        ],
        axis=0,
    )


def _split_action23(action23: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action23 = np.asarray(action23, dtype=np.float64).reshape(23)
    return action23[0:3], action23[3:7], action23[7:23]


def _x22_to_action23(x22: np.ndarray) -> np.ndarray:
    x22 = np.asarray(x22, dtype=np.float64).reshape(22)
    pos, rotvec, hand = x22[0:3], x22[3:6], x22[6:22]
    quat = R.from_rotvec(rotvec).as_quat(scalar_first=True)
    return _action23(pos, quat, hand)


def _action23_to_x22(action23: np.ndarray) -> np.ndarray:
    pos, quat, hand = _split_action23(action23)
    rotvec = R.from_quat(quat[[1, 2, 3, 0]]).as_rotvec()
    return np.concatenate([pos, rotvec, hand], axis=0)


def _merge_bimanual(
    active_side: Side,
    x22: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    active23 = _x22_to_action23(x22)
    if active_side == "right":
        return active23, hold_left
    return hold_right, active23


def _merge_achieved_actions(
    side: Side,
    achieved23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sim-achieved active arm qpos/mocap (post rollout), not commanded x."""
    achieved23 = vec_to_arm_action(achieved23)
    if side == "right":
        return achieved23, vec_to_arm_action(hold_left)
    return vec_to_arm_action(hold_right), achieved23


RolloutOpt = Literal["settle", "physics"]


def _ik_success(
    metrics: dict[str, float],
    *,
    success_hand_rmse_m: float,
    success_laplacian_rmse_m: float,
    success_min_contact: int = 0,
) -> bool:
    """Interaction mesh δ* matched (Laplacian primary, hand guard)."""
    ok = (
        metrics["laplacian_rmse_m"] <= success_laplacian_rmse_m
        and metrics["hand_rmse_m"] <= success_hand_rmse_m
    )
    if success_min_contact > 0:
        ok = ok and metrics.get("contact_count", 0) >= success_min_contact
    return ok


def _pairwise_local_cost(current: np.ndarray, target: np.ndarray) -> float:
    """pyroki 09-style relative position + angle matching."""
    n = current.shape[0]
    if n < 2:
        return 0.0
    delta_c = current[:, None, :] - current[None, :, :]
    delta_t = target[:, None, :] - target[None, :, :]
    mask = 1.0 - np.eye(n)
    pos_err = ((delta_c - delta_t) * mask[..., None]) ** 2
    norm_c = delta_c / (np.linalg.norm(delta_c, axis=-1, keepdims=True) + 1e-6)
    norm_t = delta_t / (np.linalg.norm(delta_t, axis=-1, keepdims=True) + 1e-6)
    ang_err = (1.0 - (norm_c * norm_t).sum(axis=-1)) * mask
    return float(pos_err.sum() + ang_err.sum())


def interaction_metrics_obj_frame(
    model,
    data,
    *,
    side: Side,
    obj_body: str,
    target_hand_obj: np.ndarray,
    target_obj_samples_obj: np.ndarray,
    target_laplacian: np.ndarray,
    adjacency: list[list[int]],
) -> tuple[float, dict[str, float]]:
    """Laplacian + hand error in object body frame (holosoma obj_frame=True)."""
    obj_id = model.body(obj_body).id
    obj_pos = np.asarray(data.xpos[obj_id], dtype=np.float64)
    obj_quat = np.asarray(data.xquat[obj_id], dtype=np.float64)

    hand_world = hand_keypoints_world(model, data, _hand_bodies(side))
    hand_obj = world_to_object(hand_world, obj_pos, obj_quat)
    target_hand = np.asarray(target_hand_obj, dtype=np.float64)
    obj_samples = np.asarray(target_obj_samples_obj, dtype=np.float64)

    vertices = np.concatenate([hand_obj, obj_samples], axis=0)
    lap_current = laplacian_coordinates(vertices, adjacency)
    lap_target = np.asarray(target_laplacian, dtype=np.float64)

    lap_err = lap_current - lap_target
    hand_err = hand_obj - target_hand
    metrics = {
        "laplacian_rmse_m": float(np.sqrt(np.mean(lap_err**2))),
        "hand_rmse_m": float(np.sqrt(np.mean(hand_err**2))),
    }
    local = _pairwise_local_cost(hand_obj, target_hand)
    cost = (
        metrics["laplacian_rmse_m"] ** 2 * len(lap_err.flatten())
        + metrics["hand_rmse_m"] ** 2 * hand_obj.size
        + local
    )
    return cost, metrics


def _hand_joint_bounds(model, side: Side) -> tuple[np.ndarray, np.ndarray]:
    names = (
        [
            "ffj0_right", "ffj1_right", "ffj2_right", "ffj3_right",
            "mfj0_right", "mfj1_right", "mfj2_right", "mfj3_right",
            "rfj0_right", "rfj1_right", "rfj2_right", "rfj3_right",
            "thj0_right", "thj1_right", "thj2_right", "thj3_right",
        ]
        if side == "right"
        else [
            "rfj0_left", "rfj1_left", "rfj2_left", "rfj3_left",
            "mfj0_left", "mfj1_left", "mfj2_left", "mfj3_left",
            "ffj0_left", "ffj1_left", "ffj2_left", "ffj3_left",
            "thj0_left", "thj1_left", "thj2_left", "thj3_left",
        ]
    )
    lo, hi = [], []
    for name in names:
        jnt = model.joint(name)
        lo.append(float(model.jnt_range[jnt.id, 0]))
        hi.append(float(model.jnt_range[jnt.id, 1]))
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def initial_active_from_canonical(
    raw_env,
    canonical: dict,
    *,
    object_name: ObjectName,
) -> np.ndarray:
    """IK seed: palm at T×hand[0] + mocap–palm offset; quat/fingers from rep-frame δ*."""
    side = _active_side(object_name)
    home = read_arm_action(raw_env, side)
    palm_target = palm_world_from_canonical(raw_env, canonical, object_name=object_name)
    palm_now = hand_keypoints_world(raw_env._model, raw_env._data, _hand_bodies(side))[0]
    mocap_pos = palm_target + (home[0:3] - palm_now)

    mocap_quat_obj = canonical.get("mocap_quat_obj")
    if mocap_quat_obj is not None and float(np.linalg.norm(mocap_quat_obj)) > 0.5:
        _, mocap_quat = mocap_world_from_canonical(raw_env, canonical, object_name=object_name)
    else:
        mocap_quat = np.asarray(home[3:7], dtype=np.float64)

    hand_joint = canonical.get("hand_joint_median")
    if hand_joint is not None:
        hand = np.asarray(hand_joint, dtype=np.float64).reshape(16)
    else:
        hand = _ALLEGRO_OPEN.copy()
    return _action23(mocap_pos, mocap_quat, hand)


def compose_grasp_action_from_canonical(
    raw_env,
    canonical: dict,
    *,
    object_name: ObjectName,
) -> np.ndarray:
    """Alias: T_world_obj × mocap + rep-frame finger joints from δ*."""
    return initial_active_from_canonical(raw_env, canonical, object_name=object_name)


def warm_start_from_demo(
    env,
    *,
    zarr_path: Path,
    grasp_frame: int,
    initial_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay demo to grasp frame; return settled arm actions."""
    from dexjoco.tasks import CONFIG_MAPPING
    from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

    raw_env = env.unwrapped
    actions, _, init = load_zarr_episode(Path(zarr_path))
    if initial_state is None:
        initial_state = init
    frame = int(np.clip(grasp_frame, 0, len(actions) - 1))
    config = CONFIG_MAPPING["bimanual_assembly"]()
    env.reset()
    if initial_state is not None and has_restorer("bimanual_assembly"):
        restore_initial_state(env, "bimanual_assembly", config, initial_state)
    for action in actions[: frame + 1]:
        raw_env.step(raw_flat_to_dict(action))
    return read_arm_action(raw_env, "right"), read_arm_action(raw_env, "left")


def _site_tracking_rmse_m(raw_env, side: Side, target_mocap_pos: np.ndarray) -> float:
    site_id = int(raw_env._site_left_id if side == "left" else raw_env._site_right_id)
    site_pos = np.asarray(raw_env._data.site_xpos[site_id], dtype=np.float64)
    return float(np.linalg.norm(site_pos - np.asarray(target_mocap_pos, dtype=np.float64).reshape(3)))


def _fast_settle_active(
    raw_env,
    *,
    side: Side,
    x22: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    settle_steps: int,
) -> None:
    """Cheap IK inner loop: opspace ramp (no env.step task overhead)."""
    active23 = _x22_to_action23(x22)
    settle_side_actions(
        raw_env,
        side=side,
        active23=active23,
        hold_right=hold_right,
        hold_left=hold_left,
        n_substeps=settle_steps,
    )


def _physics_hold_active(
    raw_env,
    *,
    side: Side,
    x22: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    settle_steps: int,
    hold_steps: int,
    detector,
) -> int:
    """Arm to grasp mocap with open hand, then close fingers (GenHand trajectory.py)."""
    active23 = _x22_to_action23(x22)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    settle_side_actions(
        raw_env,
        side=side,
        active23=active23,
        hold_right=hold_right,
        hold_left=hold_left,
        n_substeps=max(int(settle_steps), 1),
    )
    arm_cmd = active23[0:7]
    target_hand = active23[7:23]
    warmup = max(int(hold_steps) // 2, 0)
    final_contact = 0
    n_hold = max(int(hold_steps), 1)
    for i in range(n_hold):
        t = (i + 1) / n_hold
        hand = (1.0 - t) * _ALLEGRO_OPEN + t * target_hand
        cmd = np.concatenate([arm_cmd, hand], axis=0)
        if side == "left":
            settle_bimanual_actions(
                raw_env,
                right23=hold_right,
                left23=cmd,
                n_substeps=1,
            )
        else:
            settle_bimanual_actions(
                raw_env,
                right23=cmd,
                left23=hold_left,
                n_substeps=1,
            )
        if detector is not None and i >= warmup:
            c = detector.compute(raw_env)
            final_contact = int(
                c.tray_contact_count if side == "left" else c.peg_contact_count
            )
    return final_contact


def _reach_active(
    raw_env,
    *,
    side: Side,
    x22: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    reach_steps: int,
) -> None:
    from interaction_retarget.grasp.approach import reach_side_via_env

    active23 = _x22_to_action23(x22)
    reach_side_via_env(
        raw_env,
        side=side,
        target23=active23,
        hold_right=hold_right,
        hold_left=hold_left,
        n_steps=reach_steps,
    )


def solve_grasp_ik(
    raw_env,
    canonical: dict,
    *,
    object_name: ObjectName,
    hold_right: np.ndarray | None = None,
    hold_left: np.ndarray | None = None,
    initial_active: np.ndarray | None = None,
    weights: IkWeights | None = None,
    reach_steps: int = 90,
    reach_steps_opt: int | None = None,
    settle_steps_opt: int = 15,
    maxiter: int = 10,
    n_outer_iters: int = 2,
    maxfun: int = 24,
    optimize: bool = False,
    prealign_maxiter: int = 0,
    finger_refine_maxiter: int = 0,
    pos_bounds_m: float = 0.12,
    success_laplacian_rmse_m: float = 0.045,
    success_hand_rmse_m: float = 0.055,
    success_min_contact: int = 0,
    restore_env: bool = True,
    rollout_opt: RolloutOpt = "settle",
    physics_hold_steps: int = 12,
    detector=None,
) -> GraspIkResult:
    """Object-frame point-alignment IK; does not leave sim mutated if restore_env."""
    _ = (prealign_maxiter, finger_refine_maxiter)
    weights = weights or IkWeights()
    reach_steps_opt = int(settle_steps_opt if reach_steps_opt is None else reach_steps_opt)
    reach_steps = int(reach_steps)
    settle_steps_opt = int(settle_steps_opt)
    side = _active_side(object_name)
    obj_body = _object_body(object_name)

    if hold_right is None:
        hold_right = read_arm_action(raw_env, "right")
    if hold_left is None:
        hold_left = read_arm_action(raw_env, "left")
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)

    sim_home = snapshot_sim(raw_env)

    if initial_active is None:
        initial_active = initial_active_from_canonical(raw_env, canonical, object_name=object_name)
    initial_active = vec_to_arm_action(initial_active)

    target_hand = np.asarray(canonical["hand_points_obj"], dtype=np.float64)
    target_obj = np.asarray(canonical["object_samples_obj"], dtype=np.float64)
    target_lap = np.asarray(canonical["laplacian_coords"], dtype=np.float64)
    adjacency = canonical["adjacency"]
    contact_sites_obj = canonical.get("contact_sites_obj")

    model = raw_env._model
    hand_lo, hand_hi = _hand_joint_bounds(model, side)

    init_n = max(reach_steps, settle_steps_opt)
    restore_sim(raw_env, sim_home)
    _reach_active(
        raw_env,
        side=side,
        x22=_action23_to_x22(initial_active),
        hold_right=hold_right,
        hold_left=hold_left,
        reach_steps=init_n,
    )
    x = _action23_to_x22(read_arm_action(raw_env, side))
    sim_work = snapshot_sim(raw_env)

    mocap_center = np.asarray(_x22_to_action23(x)[0:3], dtype=np.float64)
    bound = float(pos_bounds_m)
    bounds_lo = np.concatenate([mocap_center - bound, np.full(3, -np.pi), hand_lo], axis=0)
    bounds_hi = np.concatenate([mocap_center + bound, np.full(3, np.pi), hand_hi], axis=0)

    def _metrics_at_current() -> dict[str, float]:
        achieved = read_arm_action(raw_env, side)
        _, metrics = interaction_metrics_obj_frame(
            model,
            raw_env._data,
            side=side,
            obj_body=obj_body,
            target_hand_obj=target_hand,
            target_obj_samples_obj=target_obj,
            target_laplacian=target_lap,
            adjacency=adjacency,
        )
        metrics["site_tracking_rmse_m"] = _site_tracking_rmse_m(raw_env, side, achieved[0:3])
        hand_world = hand_keypoints_world(model, raw_env._data, _hand_bodies(side))
        hand_obj = world_to_object(hand_world, *_object_pose(raw_env, obj_body))
        metrics["contact_site_rmse_m"] = _contact_site_rmse_m(hand_obj, contact_sites_obj)
        if detector is not None:
            c = detector.compute(raw_env)
            metrics["contact_count"] = float(
                c.tray_contact_count if object_name == "tray" else c.peg_contact_count
            )
        else:
            metrics["contact_count"] = 0.0
        return metrics

    def _metrics_after_x22(
        x_vec: np.ndarray,
        *,
        rollout: RolloutOpt | None = None,
        n_steps: int | None = None,
    ) -> tuple[dict[str, float], float]:
        rollout = rollout_opt if rollout is None else rollout
        n_steps = settle_steps_opt if n_steps is None else int(n_steps)
        restore_sim(raw_env, sim_work)
        min_contact = 0
        if rollout == "physics":
            min_contact = _physics_hold_active(
                raw_env,
                side=side,
                x22=x_vec,
                hold_right=hold_right,
                hold_left=hold_left,
                settle_steps=n_steps,
                hold_steps=physics_hold_steps,
                detector=detector,
            )
        elif rollout == "settle":
            _fast_settle_active(
                raw_env,
                side=side,
                x22=x_vec,
                hold_right=hold_right,
                hold_left=hold_left,
                settle_steps=n_steps,
            )
        else:
            _reach_active(
                raw_env,
                side=side,
                x22=x_vec,
                hold_right=hold_right,
                hold_left=hold_left,
                reach_steps=n_steps,
            )
        _, metrics = interaction_metrics_obj_frame(
            model,
            raw_env._data,
            side=side,
            obj_body=obj_body,
            target_hand_obj=target_hand,
            target_obj_samples_obj=target_obj,
            target_laplacian=target_lap,
            adjacency=adjacency,
        )
        site_rmse = _site_tracking_rmse_m(raw_env, side, _x22_to_action23(x_vec)[0:3])
        metrics["site_tracking_rmse_m"] = site_rmse
        if rollout == "physics" and detector is not None:
            metrics["contact_count"] = float(min_contact)
        elif detector is not None:
            c = detector.compute(raw_env)
            metrics["contact_count"] = float(
                c.tray_contact_count if object_name == "tray" else c.peg_contact_count
            )
        else:
            metrics["contact_count"] = 0.0
        hand_world = hand_keypoints_world(model, raw_env._data, _hand_bodies(side))
        hand_obj = world_to_object(hand_world, *_object_pose(raw_env, obj_body))
        metrics["contact_site_rmse_m"] = _contact_site_rmse_m(hand_obj, contact_sites_obj)
        return metrics, site_rmse

    def _result_from_current(metrics: dict[str, float], *, cost: float) -> GraspIkResult:
        achieved = read_arm_action(raw_env, side)
        right23, left23 = _merge_achieved_actions(side, achieved, hold_right, hold_left)
        if restore_env:
            restore_sim(raw_env, sim_home)
        return GraspIkResult(
            object_name=object_name,
            active_side=side,
            action_right=right23,
            action_left=left23,
            cost=cost,
            laplacian_rmse_m=metrics["laplacian_rmse_m"],
            hand_rmse_m=metrics["hand_rmse_m"],
            contact_count=int(metrics.get("contact_count", 0)),
            contact_site_rmse_m=float(metrics.get("contact_site_rmse_m", 0.0)),
            success=_ik_success(
                metrics,
                success_hand_rmse_m=success_hand_rmse_m,
                success_laplacian_rmse_m=success_laplacian_rmse_m,
                success_min_contact=success_min_contact,
            ),
        )

    def _finalize_result(x_vec: np.ndarray, *, cost: float) -> GraspIkResult:
        if rollout_opt == "physics" and detector is not None:
            metrics, _ = _metrics_after_x22(x_vec, rollout="physics", n_steps=settle_steps_opt)
        else:
            fin_steps = max(12, settle_steps_opt // 3)
            metrics, _ = _metrics_after_x22(x_vec, n_steps=fin_steps)
        return _result_from_current(metrics, cost=cost)

    restore_sim(raw_env, sim_work)
    if rollout_opt == "physics" and detector is not None:
        init_metrics, _ = _metrics_after_x22(x, rollout="physics", n_steps=settle_steps_opt)
    else:
        init_metrics = _metrics_at_current()
    if _ik_success(
        init_metrics,
        success_hand_rmse_m=success_hand_rmse_m,
        success_laplacian_rmse_m=success_laplacian_rmse_m,
        success_min_contact=success_min_contact,
    ):
        return _result_from_current(init_metrics, cost=0.0)

    if not optimize or maxiter <= 0 or n_outer_iters <= 0:
        return _result_from_current(init_metrics, cost=0.0)

    def objective(x_vec: np.ndarray) -> float:
        metrics, site_rmse = _metrics_after_x22(x_vec)
        hand_obj = world_to_object(
            hand_keypoints_world(model, raw_env._data, _hand_bodies(side)),
            *_object_pose(raw_env, obj_body),
        )
        lap_term = metrics["laplacian_rmse_m"] ** 2
        global_term = metrics["hand_rmse_m"] ** 2
        local_term = _pairwise_local_cost(hand_obj, target_hand)
        site_term = site_rmse**2
        reg = float(np.sum((x_vec[6:22] - _ALLEGRO_HOME) ** 2))
        contact_gap = max(0.0, float(MIN_GRASP_CONTACT_COUNT) - metrics.get("contact_count", 0.0))
        contact_term = contact_gap**2
        contact_site_term = float(metrics.get("contact_site_rmse_m", 0.0)) ** 2
        return (
            weights.laplacian * lap_term
            + weights.global_hand * global_term
            + weights.local_hand * local_term
            + weights.site_tracking * site_term
            + weights.joint_regularization * reg
            + weights.contact * contact_term
            + weights.contact_site * contact_site_term
        )

    last_cost = np.inf
    res_fun = 0.0
    for _ in range(int(n_outer_iters)):
        restore_sim(raw_env, sim_work)
        res = minimize(
            objective,
            x,
            method="L-BFGS-B",
            bounds=list(zip(bounds_lo, bounds_hi, strict=True)),
            options={"maxiter": int(maxiter), "maxfun": int(maxfun), "disp": False},
        )
        x = np.asarray(res.x, dtype=np.float64)
        res_fun = float(res.fun)
        sim_work = snapshot_sim(raw_env)
        if np.isclose(res_fun, last_cost, rtol=1e-4, atol=1e-6):
            break
        last_cost = res_fun

    restore_sim(raw_env, sim_work)
    return _finalize_result(x, cost=res_fun)


def solve_from_canonical_npz(
    raw_env,
    npz_path: Path,
    *,
    object_name: ObjectName | None = None,
    **kwargs,
) -> GraspIkResult:
    canonical = load_canonical_grasp(Path(npz_path))
    obj = object_name or str(canonical["object_name"])  # type: ignore[assignment]
    if obj not in ("tray", "peg"):
        raise ValueError(obj)
    return solve_grasp_ik(raw_env, canonical, object_name=obj, **kwargs)  # type: ignore[arg-type]
