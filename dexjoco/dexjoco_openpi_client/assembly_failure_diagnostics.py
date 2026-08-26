"""Structured failure diagnostics for full-task bimanual assembly evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from dexjoco.sim.envs.assembly_geometry import names_from_raw
from hybrid_insert.assembly_contacts import AssemblyContactLabeler
from hybrid_insert.geometry import peg_insert_end_pos, tip_socket_distance


@dataclass
class AssemblyFailureSummary:
    success: bool
    failure_reason: str
    highest_stage: str
    observed_steps: int
    ever_tray_ok: bool
    ever_peg_ok: bool
    ever_both_ok: bool
    ever_near_socket: bool
    ever_insert_contact: bool
    tray_lost_after_both: bool
    peg_lost_after_both: bool
    max_insert_contact_streak: int
    min_tip_socket_distance_m: float
    final_tray_ok: bool
    final_peg_ok: bool
    final_insert_ok: bool


class AssemblyFailureTracker:
    """Track the highest achieved task stage and classify terminal failures."""

    def __init__(self, raw_env, *, near_socket_m: float = 0.12, lost_streak_steps: int = 10) -> None:
        self.raw_env = raw_env
        self.near_socket_m = float(near_socket_m)
        self.lost_streak_steps = int(lost_streak_steps)
        self.labeler = AssemblyContactLabeler(raw_env)
        self.labeler.reset_reference(raw_env)
        names = names_from_raw(raw_env)
        self._peg_body_id = int(raw_env._model.body(names.peg_body).id)
        self._socket_site_id = int(raw_env._model.site(names.socket_site).id)
        self.observed_steps = 0
        self.ever_tray_ok = False
        self.ever_peg_ok = False
        self.ever_both_ok = False
        self.ever_near_socket = False
        self.ever_insert_contact = False
        self.tray_lost_after_both = False
        self.peg_lost_after_both = False
        self._tray_lost_streak = 0
        self._peg_lost_streak = 0
        self.insert_contact_streak = 0
        self.max_insert_contact_streak = 0
        self.min_tip_socket_distance_m = float("inf")
        self._final_outcome = self.labeler.compute(raw_env)
        self.observe()

    def _tip_socket_distance(self) -> float:
        data = self.raw_env._data
        tip = peg_insert_end_pos(data.xpos[self._peg_body_id], data.xmat[self._peg_body_id])
        socket = np.asarray(data.site_xpos[self._socket_site_id], dtype=np.float64)
        return tip_socket_distance(tip, socket)

    def observe(self) -> None:
        outcome = self.labeler.compute(self.raw_env)
        distance = self._tip_socket_distance()
        self.observed_steps += 1
        self.ever_tray_ok = self.ever_tray_ok or bool(outcome.tray_ok)
        self.ever_peg_ok = self.ever_peg_ok or bool(outcome.peg_ok)
        both_ok = bool(outcome.tray_ok and outcome.peg_ok)
        self.ever_both_ok = self.ever_both_ok or both_ok
        if self.ever_both_ok:
            self._tray_lost_streak = 0 if outcome.tray_ok else self._tray_lost_streak + 1
            self._peg_lost_streak = 0 if outcome.peg_ok else self._peg_lost_streak + 1
            self.tray_lost_after_both = self.tray_lost_after_both or (
                self._tray_lost_streak >= self.lost_streak_steps
            )
            self.peg_lost_after_both = self.peg_lost_after_both or (
                self._peg_lost_streak >= self.lost_streak_steps
            )
        self.min_tip_socket_distance_m = min(self.min_tip_socket_distance_m, distance)
        self.ever_near_socket = self.ever_near_socket or (both_ok and distance <= self.near_socket_m)
        if outcome.insert_ok:
            self.insert_contact_streak += 1
            self.max_insert_contact_streak = max(
                self.max_insert_contact_streak,
                self.insert_contact_streak,
            )
            self.ever_insert_contact = True
        else:
            self.insert_contact_streak = 0
        self._final_outcome = outcome

    def _highest_stage(self, success: bool) -> str:
        if success:
            return "success"
        if self.ever_insert_contact:
            return "insert_contact"
        if self.ever_near_socket:
            return "near_socket"
        if self.ever_both_ok:
            return "bimanual_grasp_lift"
        if self.ever_peg_ok:
            return "peg_grasp_lift"
        if self.ever_tray_ok:
            return "tray_grasp_lift"
        return "initial"

    def _failure_reason(self, success: bool) -> str:
        if success:
            return "success"
        if not self.ever_tray_ok:
            return "tray_grasp_or_lift_failed"
        if not self.ever_peg_ok:
            return "peg_grasp_or_lift_failed"
        if not self.ever_both_ok:
            return "bimanual_grasp_coordination_failed"
        if self.peg_lost_after_both:
            return "peg_lost_after_grasp"
        if self.tray_lost_after_both:
            return "tray_lost_after_grasp"
        if not self.ever_near_socket:
            return "transport_to_socket_failed"
        if not self.ever_insert_contact:
            return "alignment_or_hole_entry_failed"
        if self.max_insert_contact_streak < 30:
            return "unstable_or_incomplete_insertion"
        return "success_latch_mismatch"

    def finalize(self, *, success: bool) -> dict:
        summary = AssemblyFailureSummary(
            success=bool(success),
            failure_reason=self._failure_reason(bool(success)),
            highest_stage=self._highest_stage(bool(success)),
            observed_steps=self.observed_steps,
            ever_tray_ok=self.ever_tray_ok,
            ever_peg_ok=self.ever_peg_ok,
            ever_both_ok=self.ever_both_ok,
            ever_near_socket=self.ever_near_socket,
            ever_insert_contact=self.ever_insert_contact,
            tray_lost_after_both=self.tray_lost_after_both,
            peg_lost_after_both=self.peg_lost_after_both,
            max_insert_contact_streak=self.max_insert_contact_streak,
            min_tip_socket_distance_m=float(self.min_tip_socket_distance_m),
            final_tray_ok=bool(self._final_outcome.tray_ok),
            final_peg_ok=bool(self._final_outcome.peg_ok),
            final_insert_ok=bool(self._final_outcome.insert_ok),
        )
        return asdict(summary)
