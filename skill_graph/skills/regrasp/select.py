"""Select best grasp template from bank."""

from __future__ import annotations

from skill_graph.adapters.assembly import AssemblySim
from skill_graph.constants import PEG_YAW_CANDIDATES_RAD, ObjectName, Side
from skill_graph.skills.regrasp.score import score_template
from skill_graph.skills.templates.schema import GraspTemplate


def _yaw_candidates(object_name: ObjectName) -> tuple[float, ...]:
    if object_name == "peg":
        return PEG_YAW_CANDIDATES_RAD
    return (0.0,)


def rank_templates(
    sim: AssemblySim,
    templates: list[GraspTemplate],
    *,
    side: Side,
    object_name: ObjectName,
    prefer_episode: int | None = None,
) -> list[tuple[GraspTemplate, float]]:
    scored: list[tuple[GraspTemplate, float]] = []
    for t in templates:
        if t.object_name != object_name or t.side != side:
            continue
        for yaw in _yaw_candidates(object_name):
            variant = t.with_object_yaw(yaw)
            scored.append(
                (
                    variant,
                    score_template(
                        sim, variant, side=side, object_name=object_name, prefer_episode=prefer_episode
                    ),
                )
            )
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def select_best_template(
    sim: AssemblySim,
    templates: list[GraspTemplate],
    *,
    side: Side,
    object_name: ObjectName,
    top_k: int = 5,
    prefer_episode: int | None = None,
) -> list[GraspTemplate]:
    ranked = rank_templates(
        sim, templates, side=side, object_name=object_name, prefer_episode=prefer_episode
    )
    return [t for t, _ in ranked[:top_k]]
