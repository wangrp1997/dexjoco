"""Detect insert-phase frame ranges from privileged sim replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interaction_retarget.grasp.lift_reference import detect_lift_end_frames
from interaction_retarget.sim.grasp_timing import GraspTiming
from interaction_retarget.sim.replay import ReplayTrace


@dataclass(frozen=True)
class InsertSegment:
    start_frame: int
    end_frame: int
    peg_lift_end_frame: int
    first_insert_ok_frame: int | None
    first_approach_frame: int | None
    min_tip_dist_frame: int
    min_tip_dist_m: float
    num_frames: int


def detect_insert_segment(
    trace: ReplayTrace,
    timing: GraspTiming,
    *,
    insert_ok: np.ndarray | None = None,
    approach_ready: np.ndarray | None = None,
    tip_socket_dist_m: np.ndarray | None = None,
    min_segment_frames: int = 8,
) -> InsertSegment:
    """Pick [start, end] for PoseInsert export from a full demo replay trace."""
    lift_frames = detect_lift_end_frames(trace, timing)
    peg_lift_end = int(lift_frames.peg_lift_end_frame)
    n = len(trace.steps)

    if insert_ok is None:
        insert_ok = np.zeros(n, dtype=bool)
    else:
        insert_ok = np.asarray(insert_ok, dtype=bool).reshape(-1)
        if insert_ok.shape[0] != n:
            raise ValueError(f"insert_ok length {insert_ok.shape[0]} != trace steps {n}")

    if tip_socket_dist_m is None:
        tip_socket_dist_m = np.full(n, np.inf, dtype=np.float64)
    else:
        tip_socket_dist_m = np.asarray(tip_socket_dist_m, dtype=np.float64).reshape(-1)
        if tip_socket_dist_m.shape[0] != n:
            raise ValueError(f"tip_socket_dist_m length {tip_socket_dist_m.shape[0]} != trace steps {n}")

    search_start = int(np.clip(peg_lift_end, 0, max(0, n - 1)))
    search_end = n - 1
    min_tip_frame = search_start + int(np.argmin(tip_socket_dist_m[search_start:]))
    min_tip_dist = float(tip_socket_dist_m[min_tip_frame])

    first_insert: int | None
    if insert_ok.any():
        first_insert = int(np.argmax(insert_ok))
        end = int(first_insert)
        for t in range(first_insert, n):
            if insert_ok[t]:
                end = t
    else:
        first_insert = None
        end = int(min_tip_frame)

    first_approach: int | None = None
    start = peg_lift_end
    if approach_ready is not None:
        approach_ready = np.asarray(approach_ready, dtype=bool).reshape(-1)
        for t in range(peg_lift_end, min(end + 1, n)):
            if approach_ready[t]:
                first_approach = t
                start = t
                break

    start = int(np.clip(start, 0, max(0, n - 1)))
    end = int(np.clip(end, start, n - 1))
    if end - start + 1 < min_segment_frames:
        end = int(np.clip(start + min_segment_frames - 1, start, n - 1))

    return InsertSegment(
        start_frame=start,
        end_frame=end,
        peg_lift_end_frame=peg_lift_end,
        first_insert_ok_frame=first_insert,
        first_approach_frame=first_approach,
        min_tip_dist_frame=min_tip_frame,
        min_tip_dist_m=min_tip_dist,
        num_frames=end - start + 1,
    )
