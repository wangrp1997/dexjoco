"""Score grasp templates for online selection."""

from __future__ import annotations

import numpy as np

from skill_graph.adapters.assembly import AssemblySim
from skill_graph.adapters.control import read_arm23
from skill_graph.constants import MIN_GRASP_CONTACTS, ObjectName, Side
from skill_graph.math.se3 import arm23_from_object_frame
from skill_graph.skills.templates.schema import GraspTemplate


def score_template(
    sim: AssemblySim,
    template: GraspTemplate,
    *,
    side: Side,
    object_name: ObjectName,
    prefer_episode: int | None = None,
) -> float:
    """Higher is better. Uses reach cost + exported grasp quality."""
    obj_pos, obj_quat = sim.object_pose(object_name)
    cur = read_arm23(sim, side)
    if template.approach_mocap_pos_obj.shape[0] == 0:
        tgt_pos = template.grasp_mocap_pos_obj
        tgt_quat = template.grasp_mocap_quat_obj
        hand = template.grasp_hand
    else:
        tgt_pos = template.approach_mocap_pos_obj[0]
        tgt_quat = template.approach_mocap_quat_obj[0]
        hand = template.approach_hand[0]
    first = arm23_from_object_frame(tgt_pos, tgt_quat, hand, live_obj_pos=obj_pos, live_obj_quat=obj_quat)
    reach_cost = float(np.linalg.norm(cur[:3] - first[:3]))
    contact_bonus = min(template.export_contact_count, 8) * 0.05
    quality = 0.0 if template.export_contact_count < MIN_GRASP_CONTACTS else 0.2
    ep_bonus = 0.15 if prefer_episode is not None and template.episode_index == prefer_episode else 0.0
    return contact_bonus + quality + ep_bonus - reach_cost
