"""ContactOpt ``optimize_pose`` loop (numpy/scipy) + Topo δ* Laplacian guard + GraspTTA.

Source: refs/contactopt/contactopt/optimize_pose.py L17–98
Adapted: Allegro action23 (mocap trans + fingers), MuJoCo FK, holosoma Laplacian penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize

from interaction_retarget.constants import FINGERTIP_KEYPOINT_INDICES, PEG_BODY, TRAY_BODY
from interaction_retarget.contact_refine.config import ContactCapsConfig, DEFAULT_CAPS
from interaction_retarget.contact_refine.targets import (
    demo_contact_targets_from_canonical,
    object_normals_at_points,
    predicted_contact_maps_obj_frame,
)
from interaction_retarget.contactopt.contactopt.contact_map_loss import contact_map_match_loss
from interaction_retarget.contactopt.contactopt.penetration import penetration_cost_along_normal
from interaction_retarget.grasp.ik import _contact_site_rmse_m, interaction_metrics_obj_frame
from interaction_retarget.grasp.repair import _hand_joint_bounds
from interaction_retarget.GraspTTA.utils.loss import contact_loss_object_cmap
from interaction_retarget.mesh.sampling import load_object_mesh
from interaction_retarget.constants import INDUSTREAL_MESH_SCALE, PEG_MESH_PATH, TRAY_MESH_PATH
from interaction_retarget.sim.hand_geom import hand_keypoints_world
from interaction_retarget.sim.settle import read_arm_action, settle_bimanual_actions, vec_to_arm_action
from interaction_retarget.sim.state import restore_sim, snapshot_sim
from interaction_retarget.transforms import world_to_object

Side = Literal["left", "right"]
ObjectName = Literal["tray", "peg"]


@dataclass(frozen=True)
class ContactOptPoseConfig:
    """Mirrors ContactOpt optimize_pose hyper-params + dexjoco Topo guards."""

    n_iter: int = 12
    maxfun: int = 40
    w_cont_hand: float = 2.0
    w_cont_obj: float = 1.0
    w_cont_asym: float = 2.0
    w_opt_trans: float = 0.3
    w_opt_pose: float = 1.0
    w_pen_cost: float = 600.0
    pen_it: int = 0
    w_grasptta: float = 1.0
    w_laplacian: float = 120.0
    w_hand: float = 40.0
    w_site: float = 60.0
    max_laplacian_drift_m: float = 0.048
    max_hand_drift_m: float = 0.090
    max_mocap_trans_m: float = 0.010
    settle_substeps: int = 2


DEFAULT_POSE_CFG = ContactOptPoseConfig()


def _object_body(object_name: ObjectName) -> str:
    return TRAY_BODY if object_name == "tray" else PEG_BODY


def _hand_bodies(side: Side):
    from interaction_retarget.constants import LEFT_HAND_BODIES, RIGHT_HAND_BODIES

    return LEFT_HAND_BODIES if side == "left" else RIGHT_HAND_BODIES


def _mesh_path(object_name: ObjectName):
    return TRAY_MESH_PATH if object_name == "tray" else PEG_MESH_PATH


def _hand_normals_obj(hand_obj: np.ndarray, object_name: ObjectName) -> np.ndarray:
    mesh = load_object_mesh(_mesh_path(object_name), scale=INDUSTREAL_MESH_SCALE)
    return object_normals_at_points(mesh, hand_obj)


def _demo_hand_contact_target(
    targets: dict[str, np.ndarray],
    *,
    caps: ContactCapsConfig,
) -> np.ndarray:
    hand = targets["hand_points_obj"]
    hn = targets["hand_normals_proxy_obj"]
    _, hand_c = predicted_contact_maps_obj_frame(hand, hn, targets, caps=caps)
    return np.asarray(hand_c, dtype=np.float64)


def _apply_delta(base23: np.ndarray, delta: np.ndarray, cfg: ContactOptPoseConfig) -> np.ndarray:
    """ContactOpt-style: small trans + finger delta on top of base pose."""
    base23 = vec_to_arm_action(base23)
    d = np.asarray(delta, dtype=np.float64).reshape(-1)
    out = base23.copy()
    out[0:3] = base23[0:3] + d[0:3] * float(cfg.w_opt_trans)
    out[7:23] = base23[7:23] + d[3:19] * float(cfg.w_opt_pose)
    return out


def _loss_at_action(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    canonical: dict,
    action23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    targets: dict[str, np.ndarray],
    target_hand_contact: np.ndarray,
    caps: ContactCapsConfig,
    cfg: ContactOptPoseConfig,
    iteration: int,
) -> tuple[float, dict[str, float]]:
    action23 = vec_to_arm_action(action23)
    if side == "left":
        settle_bimanual_actions(raw_env, right23=hold_right, left23=action23, n_substeps=cfg.settle_substeps)
    else:
        settle_bimanual_actions(raw_env, right23=action23, left23=hold_left, n_substeps=cfg.settle_substeps)

    obj_id = int(raw_env._model.body(_object_body(object_name)).id)
    obj_pos = np.asarray(raw_env._data.xpos[obj_id], dtype=np.float64)
    obj_quat = np.asarray(raw_env._data.xquat[obj_id], dtype=np.float64)
    hand_world = hand_keypoints_world(raw_env._model, raw_env._data, _hand_bodies(side))
    hand_obj = world_to_object(hand_world, obj_pos, obj_quat)
    hand_normals = _hand_normals_obj(hand_obj, object_name)

    pred_obj, pred_hand = predicted_contact_maps_obj_frame(hand_obj, hand_normals, targets, caps=caps)
    loss_contact_obj = contact_map_match_loss(
        targets["target_obj_contact"], pred_obj, w_cont_asym=cfg.w_cont_asym
    )
    loss_contact_hand = contact_map_match_loss(
        target_hand_contact, pred_hand, w_cont_asym=cfg.w_cont_asym
    )
    gt_loss = contact_loss_object_cmap(
        targets["object_samples_obj"], hand_obj, targets["object_cmap"]
    )
    pen = penetration_cost_along_normal(
        hand_obj[list(FINGERTIP_KEYPOINT_INDICES), :],
        hand_normals[list(FINGERTIP_KEYPOINT_INDICES), :],
        targets["object_samples_obj"],
        targets["object_normals_obj"],
    )
    pen_mean = float(np.mean(pen)) if pen.size else 0.0

    _, topo = interaction_metrics_obj_frame(
        raw_env._model,
        raw_env._data,
        side=side,
        obj_body=_object_body(object_name),
        target_hand_obj=canonical["hand_points_obj"],
        target_obj_samples_obj=canonical["object_samples_obj"],
        target_laplacian=canonical["laplacian_coords"],
        adjacency=canonical["adjacency"],
    )
    site_rmse = _contact_site_rmse_m(hand_obj, canonical.get("contact_sites_obj"))

    loss = (
        float(cfg.w_cont_obj) * loss_contact_obj
        + float(cfg.w_cont_hand) * loss_contact_hand
        + float(cfg.w_grasptta) * gt_loss
    )
    if iteration >= int(cfg.pen_it):
        loss += float(cfg.w_pen_cost) * pen_mean
    loss += (
        float(cfg.w_laplacian) * topo["laplacian_rmse_m"] ** 2
        + float(cfg.w_hand) * topo["hand_rmse_m"] ** 2
        + float(cfg.w_site) * site_rmse**2
    )
    metrics = {
        "loss_contact_obj": loss_contact_obj,
        "loss_contact_hand": loss_contact_hand,
        "grasptta_loss": gt_loss,
        "penetration_mean": pen_mean,
        "laplacian_rmse_m": topo["laplacian_rmse_m"],
        "hand_rmse_m": topo["hand_rmse_m"],
        "contact_site_rmse_m": site_rmse,
        "total_loss": loss,
    }
    return loss, metrics


def optimize_contact_pose(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    canonical: dict,
    base_action23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    caps: ContactCapsConfig | None = None,
    cfg: ContactOptPoseConfig | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """ContactOpt Adam loop → L-BFGS-B on trans+finger deltas with Laplacian guard."""
    caps = caps or DEFAULT_CAPS
    cfg = cfg or DEFAULT_POSE_CFG
    base23 = vec_to_arm_action(base_action23)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    lo, hi = _hand_joint_bounds(raw_env._model, side)
    targets = demo_contact_targets_from_canonical(canonical, object_name=object_name)
    target_hand_contact = _demo_hand_contact_target(targets, caps=caps)

    sim0 = snapshot_sim(raw_env)
    n_var = 19  # trans(3) + fingers(16)
    x0 = np.zeros(n_var, dtype=np.float64)
    iter_count = [0]

    trans_bound = float(cfg.max_mocap_trans_m) / max(float(cfg.w_opt_trans), 1e-6)
    bounds: list[tuple[float, float]] = [(-trans_bound, trans_bound)] * 3
    for i in range(16):
        lo_i = (float(lo[i]) - float(base23[7 + i])) / max(float(cfg.w_opt_pose), 1e-6)
        hi_i = (float(hi[i]) - float(base23[7 + i])) / max(float(cfg.w_opt_pose), 1e-6)
        bounds.append((min(lo_i, hi_i), max(lo_i, hi_i)))

    def objective(x: np.ndarray) -> float:
        restore_sim(raw_env, sim0)
        action = _apply_delta(base23, x, cfg)
        loss, _ = _loss_at_action(
            raw_env,
            side=side,
            object_name=object_name,
            canonical=canonical,
            action23=action,
            hold_right=hold_right,
            hold_left=hold_left,
            targets=targets,
            target_hand_contact=target_hand_contact,
            caps=caps,
            cfg=cfg,
            iteration=iter_count[0],
        )
        iter_count[0] += 1
        return loss

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(cfg.n_iter), "maxfun": int(cfg.maxfun), "ftol": 1e-5},
    )

    restore_sim(raw_env, sim0)
    best = _apply_delta(base23, result.x, cfg)
    _, metrics = _loss_at_action(
        raw_env,
        side=side,
        object_name=object_name,
        canonical=canonical,
        action23=best,
        hold_right=hold_right,
        hold_left=hold_left,
        targets=targets,
        target_hand_contact=target_hand_contact,
        caps=caps,
        cfg=cfg,
        iteration=max(int(cfg.pen_it), 0),
    )

    if (
        metrics["laplacian_rmse_m"] > float(cfg.max_laplacian_drift_m)
        or metrics["hand_rmse_m"] > float(cfg.max_hand_drift_m)
    ):
        return base23, {**metrics, "rejected_topo": 1.0}

    if side == "left":
        settle_bimanual_actions(raw_env, right23=hold_right, left23=best, n_substeps=4)
        best = vec_to_arm_action(read_arm_action(raw_env, "left"))
    else:
        settle_bimanual_actions(raw_env, right23=best, left23=hold_left, n_substeps=4)
        best = vec_to_arm_action(read_arm_action(raw_env, "right"))
    metrics["optimizer_success"] = float(result.success)
    metrics["rejected_topo"] = 0.0
    return best, metrics
