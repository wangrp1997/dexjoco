"""Contact refine: ContactOpt optimize_pose + sim contact + GraspFilter FC gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from interaction_retarget.constants import MIN_GRASP_CONTACT_COUNT, PEG_BODY, TRAY_BODY
from interaction_retarget.contact_refine.config import ContactCapsConfig, DEFAULT_CAPS
from interaction_retarget.contact_refine.optimize_pose import (
    ContactOptPoseConfig,
    DEFAULT_POSE_CFG,
    optimize_contact_pose,
)
from interaction_retarget.contact_refine.targets import demo_contact_targets_from_canonical
from interaction_retarget.grasp.repair import side_contact_count
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.settle import read_arm_action, settle_bimanual_actions, vec_to_arm_action
from interaction_retarget.sim.state import restore_sim, snapshot_sim
from interaction_retarget.tpsr.grasp_filter import GraspFilter, GraspFilterConfig

Side = Literal["left", "right"]
ObjectName = Literal["tray", "peg"]


@dataclass
class ContactRefineReport:
    contact_count: int
    contact_site_rmse_m: float
    contactopt_loss: float
    grasptta_loss: float
    penetration_mean: float
    laplacian_rmse_m: float
    hand_rmse_m: float
    total_score: float
    qp_ok: bool
    qp_max_error: float
    improved: bool


def refine_demo_contact(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    canonical: dict,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    detector: AssemblyContactDetector,
    max_iters: int = 80,
    caps: ContactCapsConfig | None = None,
    pose_cfg: ContactOptPoseConfig | None = None,
    grasp_filter_cfg: GraspFilterConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, ContactRefineReport]:
    """ContactOpt optimize_pose (Topo-guarded) → apply if sim contact ≥ start → GraspFilter FC."""
    caps = caps or DEFAULT_CAPS
    pose_cfg = pose_cfg or DEFAULT_POSE_CFG
    if max_iters != int(pose_cfg.n_iter):
        pose_cfg = ContactOptPoseConfig(**{**pose_cfg.__dict__, "n_iter": int(max_iters)})

    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    _ = demo_contact_targets_from_canonical(canonical, object_name=object_name)

    sim0 = snapshot_sim(raw_env)
    base23 = vec_to_arm_action(read_arm_action(raw_env, side))
    start_contact = int(side_contact_count(detector, raw_env, object_name=object_name))

    optimized, metrics = optimize_contact_pose(
        raw_env,
        side=side,
        object_name=object_name,
        canonical=canonical,
        base_action23=base23,
        hold_right=hold_right,
        hold_left=hold_left,
        caps=caps,
        cfg=pose_cfg,
    )

    if side == "left":
        settle_bimanual_actions(raw_env, right23=hold_right, left23=optimized, n_substeps=4)
        trial_contact = int(side_contact_count(detector, raw_env, object_name=object_name))
        if trial_contact >= start_contact:
            hold_left = vec_to_arm_action(read_arm_action(raw_env, "left"))
        else:
            restore_sim(raw_env, sim0)
            hold_left = base23
    else:
        settle_bimanual_actions(raw_env, right23=optimized, left23=hold_left, n_substeps=4)
        trial_contact = int(side_contact_count(detector, raw_env, object_name=object_name))
        if trial_contact >= start_contact:
            hold_right = vec_to_arm_action(read_arm_action(raw_env, "right"))
        else:
            restore_sim(raw_env, sim0)
            hold_right = base23

    final_contact = int(side_contact_count(detector, raw_env, object_name=object_name))

    gf_cfg = grasp_filter_cfg or GraspFilterConfig(min_contacts=MIN_GRASP_CONTACT_COUNT)
    gf = GraspFilter(gf_cfg).forward(raw_env, side=side, object_name=object_name)

    co_loss = float(metrics.get("loss_contact_obj", 0.0) + metrics.get("loss_contact_hand", 0.0))
    report = ContactRefineReport(
        contact_count=final_contact,
        contact_site_rmse_m=float(metrics.get("contact_site_rmse_m", 0.0)),
        contactopt_loss=co_loss,
        grasptta_loss=float(metrics.get("grasptta_loss", 0.0)),
        penetration_mean=float(metrics.get("penetration_mean", 0.0)),
        laplacian_rmse_m=float(metrics.get("laplacian_rmse_m", 0.0)),
        hand_rmse_m=float(metrics.get("hand_rmse_m", 0.0)),
        total_score=float(metrics.get("total_loss", 0.0)),
        qp_ok=bool(gf.ok),
        qp_max_error=float(gf.max_qp_error),
        improved=bool(final_contact > start_contact or (final_contact >= MIN_GRASP_CONTACT_COUNT and co_loss < 1.0)),
    )
    return hold_right, hold_left, report
