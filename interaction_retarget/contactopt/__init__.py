"""ContactOpt-derived contact map math (numpy port, mirror path)."""

from interaction_retarget.contactopt.contactopt.contact_map_loss import contact_map_match_loss
from interaction_retarget.contactopt.contactopt.diffcontact import (
    calculate_contact_capsule,
    capsule_contact_on_object,
    sdf_to_contact,
)
from interaction_retarget.contactopt.contactopt.penetration import penetration_cost_along_normal

__all__ = [
    "calculate_contact_capsule",
    "capsule_contact_on_object",
    "sdf_to_contact",
    "contact_map_match_loss",
    "penetration_cost_along_normal",
]
