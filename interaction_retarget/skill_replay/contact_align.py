"""Skill-replay IK weights + wrapper around ``contact_refine``."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from interaction_retarget.contact_refine import ContactRefineReport, refine_demo_contact
from interaction_retarget.grasp.ik import IkWeights
from interaction_retarget.grasp.ik_config import GraspSideConfig, side_config

ObjectName = Literal["tray", "peg"]

# Re-export for deploy
ContactAlignReport = ContactRefineReport


def skill_side_config(object_name: ObjectName, *, fast: bool = False) -> GraspSideConfig:
    """Stronger demo contact-site IK weights for one-demo replay."""
    base = side_config(object_name, fast=fast)
    w = base.weights
    boosted = IkWeights(
        laplacian=w.laplacian,
        global_hand=w.global_hand,
        local_hand=w.local_hand,
        site_tracking=w.site_tracking,
        joint_regularization=w.joint_regularization,
        inactive_arm_hold=w.inactive_arm_hold,
        contact=max(w.contact, 800.0),
        contact_site=max(w.contact_site, 600.0),
    )
    return replace(
        base,
        weights=boosted,
        n_outer_iters=min(base.n_outer_iters + 1, 4),
        maxiter=min(base.maxiter + 4, 24),
    )


def refine_demo_contact_sites(raw_env, *, side, object_name, canonical, hold_right, hold_left, detector, max_iters=24):
    """Backward-compatible name → ``contact_refine.refine_demo_contact``."""
    return refine_demo_contact(
        raw_env,
        side=side,
        object_name=object_name,
        canonical=canonical,
        hold_right=hold_right,
        hold_left=hold_left,
        detector=detector,
        max_iters=max_iters,
    )
