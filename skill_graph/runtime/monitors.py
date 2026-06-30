"""Sim-privileged monitors for graph edges."""

from __future__ import annotations

from skill_graph.adapters.assembly import AssemblySim
from skill_graph.adapters.contacts import hand_object_contacts
from skill_graph.constants import MIN_GRASP_CONTACTS, ObjectName, Side


def _side_for_object(object_name: ObjectName) -> Side:
    return "left" if object_name == "tray" else "right"


def check_monitor(sim: AssemblySim, spec: dict) -> bool:
    """Return True if failure condition triggered."""
    mtype = str(spec.get("type", ""))
    if mtype == "hold_lost":
        objects = spec.get("objects") or [spec.get("obj_name", "peg")]
        for obj in objects:
            object_name = str(obj)
            side = _side_for_object(object_name)  # type: ignore[arg-type]
            n = len(hand_object_contacts(sim, side=side, object_name=object_name))  # type: ignore[arg-type]
            if n < MIN_GRASP_CONTACTS:
                return True
        return False
    if "fn" in spec:
        return False
    return False


def first_triggered_monitor(sim: AssemblySim, monitors: list[dict]) -> int | None:
    for i, spec in enumerate(monitors):
        if check_monitor(sim, spec):
            return i
    return None
