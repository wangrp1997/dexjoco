"""Lock passive arm during active-side grasp (DexGraspBench: hold other arm fixed)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from interaction_retarget.sim.settle import read_arm_action, settle_bimanual_actions, vec_to_arm_action

Side = Literal["left", "right"]


def snapshot_arm_action(raw_env, side: Side) -> np.ndarray:
    return vec_to_arm_action(read_arm_action(raw_env, side))


def apply_bimanual_with_locked_passive(
    raw_env,
    *,
    active_side: Side,
    active23: np.ndarray,
    locked_left: np.ndarray,
    locked_right: np.ndarray,
    n_substeps: int = 1,
) -> None:
    """Active arm moves; passive arm replay locked command (no drift)."""
    locked_left = vec_to_arm_action(locked_left)
    locked_right = vec_to_arm_action(locked_right)
    active23 = vec_to_arm_action(active23)
    if active_side == "left":
        settle_bimanual_actions(raw_env, right23=locked_right, left23=active23, n_substeps=n_substeps)
    else:
        settle_bimanual_actions(raw_env, right23=active23, left23=locked_left, n_substeps=n_substeps)


def enforce_locked_passive(
    raw_env,
    *,
    locked_left: np.ndarray,
    locked_right: np.ndarray,
    n_substeps: int = 2,
) -> None:
    """Re-apply both commands; use after right-side refine so left does not drift."""
    settle_bimanual_actions(
        raw_env,
        right23=vec_to_arm_action(locked_right),
        left23=vec_to_arm_action(locked_left),
        n_substeps=n_substeps,
    )
