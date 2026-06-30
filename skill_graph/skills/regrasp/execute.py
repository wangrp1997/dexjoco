"""Execute regrasp sub-skill."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skill_graph.adapters.assembly import AssemblySim
from skill_graph.adapters.contacts import hand_object_contacts
from skill_graph.adapters.control import read_arm23
from skill_graph.constants import MIN_GRASP_CONTACTS, ObjectName, Side
from skill_graph.skills.approach.base import ApproachBackendName, get_approach_backend
from skill_graph.skills.regrasp.select import rank_templates
from skill_graph.skills.templates.schema import GraspTemplate


@dataclass
class RegraspReport:
    success: bool
    template_id: str
    contact_count: int
    attempts: int
    steps: int
    reason: str = ""


def execute_regrasp(
    sim: AssemblySim,
    templates: list[GraspTemplate],
    *,
    side: Side,
    object_name: ObjectName,
    hold_right: np.ndarray | None = None,
    hold_left: np.ndarray | None = None,
    approach: ApproachBackendName = "demo_warp",
    max_attempts: int = 5,
    prefer_episode: int | None = None,
) -> RegraspReport:
    hold_right = np.asarray(hold_right if hold_right is not None else read_arm23(sim, "right"), dtype=np.float64)
    hold_left = np.asarray(hold_left if hold_left is not None else read_arm23(sim, "left"), dtype=np.float64)
    backend = get_approach_backend(approach)
    ranked = rank_templates(
        sim, templates, side=side, object_name=object_name, prefer_episode=prefer_episode
    )

    total_steps = 0
    for attempt, (template, _score) in enumerate(ranked[:max_attempts], start=1):
        result = backend.execute(
            sim,
            template,
            side=side,
            object_name=object_name,
            hold_right=hold_right,
            hold_left=hold_left,
        )
        total_steps += result.steps
        contacts = hand_object_contacts(sim, side=side, object_name=object_name)
        n = len(contacts)
        if n >= MIN_GRASP_CONTACTS:
            return RegraspReport(
                success=True,
                template_id=template.template_id,
                contact_count=n,
                attempts=attempt,
                steps=total_steps,
            )

    return RegraspReport(
        success=False,
        template_id=ranked[0][0].template_id if ranked else "",
        contact_count=len(hand_object_contacts(sim, side=side, object_name=object_name)),
        attempts=min(max_attempts, len(ranked)),
        steps=total_steps,
        reason="insufficient_contacts",
    )
