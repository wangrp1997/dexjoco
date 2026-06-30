"""Incremental regrasp session (one warp step per eval loop frame)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from skill_graph.adapters.assembly import AssemblySim
from skill_graph.adapters.contacts import hand_object_contacts
from skill_graph.adapters.control import read_arm23, step_side
from skill_graph.constants import MIN_GRASP_CONTACTS, ObjectName, Side
from skill_graph.skills.approach.demo_warp_stepper import DemoWarpStepper, WarpStep
from skill_graph.skills.regrasp.execute import RegraspReport
from skill_graph.skills.regrasp.select import rank_templates
from skill_graph.skills.templates.schema import GraspTemplate


@dataclass
class RegraspSession:
    side: Side
    object_name: ObjectName
    prefer_episode: int | None
    max_attempts: int = 5
    compact_approach: bool = False
    _ranked: list[tuple[GraspTemplate, float]] = field(default_factory=list)
    _attempt_idx: int = 0
    _stepper: DemoWarpStepper | None = None
    _hold_right: np.ndarray | None = None
    _hold_left: np.ndarray | None = None
    _steps_done: int = 0
    _chunk_total: int = 0
    _chunk_done: int = 0
    report: RegraspReport | None = None

    @classmethod
    def begin(
        cls,
        sim: AssemblySim,
        templates: list[GraspTemplate],
        *,
        side: Side,
        object_name: ObjectName,
        prefer_episode: int | None = None,
        max_attempts: int = 5,
        compact_approach: bool = False,
    ) -> RegraspSession:
        session = cls(
            side=side,
            object_name=object_name,
            prefer_episode=prefer_episode,
            max_attempts=max_attempts,
            compact_approach=compact_approach,
        )
        session._ranked = rank_templates(
            sim, templates, side=side, object_name=object_name, prefer_episode=prefer_episode
        )
        session._hold_right = read_arm23(sim, "right")
        session._hold_left = read_arm23(sim, "left")
        if not session._load_next_stepper(sim):
            session.report = RegraspReport(
                success=False,
                template_id="",
                contact_count=0,
                attempts=0,
                steps=0,
                reason="no_templates",
            )
        return session

    @property
    def busy(self) -> bool:
        return self.report is None

    @property
    def planned_steps(self) -> int:
        """Steps remaining in the current template chunk."""
        if self._stepper is None:
            return 0
        return self._stepper.remaining + self._chunk_done

    @property
    def progress_text(self) -> str:
        return (
            f"total_step={self._steps_done} "
            f"attempt={self._attempt_idx}/{min(self.max_attempts, len(self._ranked))} "
            f"chunk={self._chunk_done}/{max(self._chunk_total, 1)}"
        )

    def _load_next_stepper(self, sim: AssemblySim) -> bool:
        self._stepper = None
        while self._attempt_idx < min(self.max_attempts, len(self._ranked)):
            template, _ = self._ranked[self._attempt_idx]
            self._attempt_idx += 1
            assert self._hold_right is not None and self._hold_left is not None
            stepper = DemoWarpStepper.from_template(
                sim,
                template,
                side=self.side,
                object_name=self.object_name,
                hold_right=self._hold_right,
                hold_left=self._hold_left,
                compact_approach=self.compact_approach,
            )
            if stepper.remaining > 0:
                self._stepper = stepper
                self._chunk_total = stepper.remaining
                self._chunk_done = 0
                return True
        return False

    def step_once(self, sim: AssemblySim) -> bool:
        """Advance one warp step. Return True when session finished."""
        if not self.busy:
            return True
        if self._stepper is None:
            if not self._load_next_stepper(sim):
                self._fail(sim, reason="insufficient_contacts")
                return True

        assert self._stepper is not None
        warp = self._stepper.pop_next()
        if warp is None:
            if self._finalize_attempt(sim):
                return True
            if not self._load_next_stepper(sim):
                self._fail(sim, reason="insufficient_contacts")
            return not self.busy

        self._apply(sim, warp)
        self._steps_done += 1
        self._chunk_done += 1
        if self._stepper.done:
            if self._finalize_attempt(sim):
                return True
            if not self._load_next_stepper(sim):
                self._fail(sim, reason="insufficient_contacts")
        return not self.busy

    def _apply(self, sim: AssemblySim, warp: WarpStep) -> None:
        step_side(
            sim,
            side=warp.side,
            active23=warp.active23,
            hold_right=warp.hold_right,
            hold_left=warp.hold_left,
        )
        if warp.side == "left":
            self._hold_left = read_arm23(sim, "left")
        else:
            self._hold_right = read_arm23(sim, "right")

    def _finalize_attempt(self, sim: AssemblySim) -> bool:
        if self._attempt_idx <= 0 or not self._ranked:
            return False
        template = self._ranked[self._attempt_idx - 1][0]
        n = len(hand_object_contacts(sim, side=self.side, object_name=self.object_name))
        if n >= MIN_GRASP_CONTACTS:
            self.report = RegraspReport(
                success=True,
                template_id=template.template_id,
                contact_count=n,
                attempts=self._attempt_idx,
                steps=self._steps_done,
            )
            return True
        return False

    def _fail(self, sim: AssemblySim, *, reason: str) -> None:
        self.report = RegraspReport(
            success=False,
            template_id=self._ranked[-1][0].template_id if self._ranked else "",
            contact_count=len(hand_object_contacts(sim, side=self.side, object_name=self.object_name)),
            attempts=self._attempt_idx,
            steps=self._steps_done,
            reason=reason,
        )
