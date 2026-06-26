"""Detect grasp-stable frames and lift onset from replay traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interaction_retarget.constants import (
    CONTACT_WINDOW,
    GRIPPER_VEL_EPS,
    LIFT_HEIGHT_M,
    MIN_GRASP_CONTACT_COUNT,
    ON_TABLE_MARGIN_M,
)


@dataclass
class GraspTiming:
    left_grasp_frame: int | None
    right_grasp_frame: int | None
    tray_lift_start: int | None
    peg_lift_start: int | None
    left_grasp_fallback: bool = False
    right_grasp_fallback: bool = False


def timing_warnings(timing: GraspTiming) -> list[str]:
    """Human-readable flags for manifest QA."""
    warnings: list[str] = []
    if timing.tray_lift_start is not None and timing.left_grasp_frame is None:
        warnings.append("tray_lift_without_grasp")
    if timing.peg_lift_start is not None and timing.right_grasp_frame is None:
        warnings.append("peg_lift_without_grasp")
    if timing.left_grasp_frame is not None and timing.tray_lift_start is not None:
        gap = timing.tray_lift_start - timing.left_grasp_frame
        if gap < 0:
            warnings.append("tray_grasp_after_lift")
        elif gap > 40:
            warnings.append(f"tray_grasp_lift_gap_{gap}")
    if timing.right_grasp_frame is not None and timing.peg_lift_start is not None:
        gap = timing.peg_lift_start - timing.right_grasp_frame
        if gap < 0:
            warnings.append("peg_grasp_after_lift")
        elif gap > 40:
            warnings.append(f"peg_grasp_lift_gap_{gap}")
    if timing.left_grasp_fallback:
        warnings.append("tray_grasp_used_fallback")
    if timing.right_grasp_fallback:
        warnings.append("peg_grasp_used_fallback")
    return warnings


def _first_true(mask: np.ndarray) -> int | None:
    if not mask.any():
        return None
    idx = int(np.argmax(mask))
    return idx if mask[idx] else None


def _peak_grasp_before_lift(
    contact_counts: np.ndarray,
    on_table: np.ndarray,
    lift_start: int | None,
    *,
    min_contact_count: int,
    settle_window: int = CONTACT_WINDOW,
) -> int | None:
    """Pick the last on-table frame before lift with strong sustained contact."""
    end = int(lift_start) if lift_start is not None else len(contact_counts)
    best_t: int | None = None
    best_c = 0
    for t in range(end):
        if not on_table[t]:
            continue
        count = int(contact_counts[t])
        if count < min_contact_count:
            continue
        w0 = max(0, t - settle_window + 1)
        window = contact_counts[w0 : t + 1]
        if window.size < settle_window or (window >= min_contact_count).sum() < settle_window - 2:
            continue
        if count > best_c or (count == best_c and (best_t is None or t > best_t)):
            best_c = count
            best_t = t
    return best_t


def _fallback_grasp_near_lift(
    contact_counts: np.ndarray,
    lift_start: int | None,
    *,
    lookback: int = 25,
    min_contact_count: int = 2,
) -> int | None:
    """When sustained on-table grasp is missed, take peak contact just before lift."""
    end = int(lift_start) if lift_start is not None else len(contact_counts)
    start = max(0, end - lookback)
    best_t: int | None = None
    best_c = 0
    for t in range(start, end):
        count = int(contact_counts[t])
        if count < min_contact_count:
            continue
        if count > best_c or (count == best_c and (best_t is None or t > best_t)):
            best_c = count
            best_t = t
    return best_t


def detect_grasp_timing(
    *,
    tray_contact: np.ndarray,
    peg_contact: np.ndarray,
    tray_contact_count: np.ndarray,
    peg_contact_count: np.ndarray,
    left_gripper_speed: np.ndarray,
    right_gripper_speed: np.ndarray,
    tray_z: np.ndarray,
    peg_z: np.ndarray,
    tray_rest_z: float,
    peg_rest_z: float,
    gripper_vel_eps: float = GRIPPER_VEL_EPS,
    lift_height_m: float = LIFT_HEIGHT_M,
    on_table_margin_m: float = ON_TABLE_MARGIN_M,
    min_grasp_contact_count: int = MIN_GRASP_CONTACT_COUNT,
) -> GraspTiming:
    """Detect grasp (peak contact before lift) and lift onset relative to rest height."""
    tray_contact = tray_contact.astype(bool)
    peg_contact = peg_contact.astype(bool)
    tray_on_table = (tray_z - tray_rest_z) <= on_table_margin_m
    peg_on_table = (peg_z - peg_rest_z) <= on_table_margin_m

    tray_lifted = (tray_z - tray_rest_z) >= lift_height_m
    peg_lifted = (peg_z - peg_rest_z) >= lift_height_m
    tray_lift = _first_true(tray_contact & tray_lifted)
    peg_lift = _first_true(peg_contact & peg_lifted)

    left_grasp = _peak_grasp_before_lift(
        tray_contact_count,
        tray_on_table,
        tray_lift,
        min_contact_count=min_grasp_contact_count,
    )
    right_grasp = _peak_grasp_before_lift(
        peg_contact_count,
        peg_on_table,
        peg_lift,
        min_contact_count=min_grasp_contact_count,
    )
    left_fallback = False
    right_fallback = False
    if left_grasp is None and tray_lift is not None:
        left_grasp = _fallback_grasp_near_lift(
            tray_contact_count,
            tray_lift,
            min_contact_count=max(2, min_grasp_contact_count - 1),
        )
        left_fallback = left_grasp is not None
    if right_grasp is None and peg_lift is not None:
        right_grasp = _fallback_grasp_near_lift(
            peg_contact_count,
            peg_lift,
            min_contact_count=max(2, min_grasp_contact_count - 1),
        )
        right_fallback = right_grasp is not None

    return GraspTiming(
        left_grasp_frame=left_grasp,
        right_grasp_frame=right_grasp,
        tray_lift_start=tray_lift,
        peg_lift_start=peg_lift,
        left_grasp_fallback=left_fallback,
        right_grasp_fallback=right_fallback,
    )
