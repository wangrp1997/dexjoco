"""Contact-consistent grasp refine (ContactOpt + GraspTTA + sidecar targets)."""

from interaction_retarget.contact_refine.optimize_pose import ContactOptPoseConfig, optimize_contact_pose
from interaction_retarget.contact_refine.refine import ContactRefineReport, refine_demo_contact
from interaction_retarget.contact_refine.targets import demo_contact_targets_from_canonical

__all__ = [
    "ContactOptPoseConfig",
    "ContactRefineReport",
    "optimize_contact_pose",
    "refine_demo_contact",
    "demo_contact_targets_from_canonical",
]
