"""Per-arm defaults (demo grasp-frame counts from sidecar timing)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from interaction_retarget.constants import MIN_GRASP_CONTACT_COUNT
from interaction_retarget.grasp.ik import IkWeights, RolloutOpt

ObjectName = Literal["tray", "peg"]
_DEFAULT_MIN_CONTACT = int(MIN_GRASP_CONTACT_COUNT)


@dataclass(frozen=True)
class GraspSideConfig:
    weights: IkWeights
    settle_steps_opt: int
    reach_steps: int
    pos_bounds_m: float
    success_hand_rmse_m: float
    success_laplacian_rmse_m: float
    maxiter: int
    n_outer_iters: int
    maxfun: int
    optimize: bool
    # GenHand approach: demo mean left_grasp_frame≈145, right≈230 (@30Hz)
    approach_pre_steps: int
    approach_grasp_steps: int
    max_repair_iters: int
    repair_hold_steps: int = 5
    rollout_opt: RolloutOpt = "settle"
    physics_hold_steps: int = 12
    success_min_contact: int = 0


_TRAY_WEIGHTS = IkWeights(
    laplacian=80.0,
    global_hand=12.0,
    local_hand=6.0,
    site_tracking=1.0,
    joint_regularization=0.008,
    contact=600.0,
    contact_site=200.0,
)
_PEG_WEIGHTS = IkWeights(
    laplacian=100.0,
    global_hand=15.0,
    local_hand=8.0,
    site_tracking=8.0,
    joint_regularization=0.005,
    contact=600.0,
    contact_site=200.0,
)

# pre≈45% of demo grasp frame (GenHand trajectory pre-grasp phase)
TRAY_SIDE = GraspSideConfig(
    weights=_TRAY_WEIGHTS,
    settle_steps_opt=25,
    reach_steps=50,
    pos_bounds_m=0.20,
    success_hand_rmse_m=0.090,
    success_laplacian_rmse_m=0.030,
    maxiter=12,
    n_outer_iters=2,
    maxfun=40,
    optimize=True,
    approach_pre_steps=65,
    approach_grasp_steps=80,
    max_repair_iters=8,
    rollout_opt="physics",
    physics_hold_steps=20,
    success_min_contact=_DEFAULT_MIN_CONTACT,
)

PEG_SIDE = GraspSideConfig(
    weights=_PEG_WEIGHTS,
    settle_steps_opt=20,
    reach_steps=55,
    pos_bounds_m=0.24,
    success_hand_rmse_m=0.095,
    success_laplacian_rmse_m=0.035,
    maxiter=18,
    n_outer_iters=3,
    maxfun=55,
    optimize=True,
    approach_pre_steps=104,
    approach_grasp_steps=126,
    max_repair_iters=8,
    rollout_opt="physics",
    physics_hold_steps=22,
    success_min_contact=_DEFAULT_MIN_CONTACT,
)


# Inference: one-shot δ* + short ramps (no scipy loop; target ~3-5s).
FAST_TRAY_SIDE = GraspSideConfig(
    weights=_TRAY_WEIGHTS,
    settle_steps_opt=12,
    reach_steps=24,
    pos_bounds_m=0.18,
    success_hand_rmse_m=0.095,
    success_laplacian_rmse_m=0.040,
    maxiter=10,
    n_outer_iters=2,
    maxfun=25,
    optimize=True,
    approach_pre_steps=16,
    approach_grasp_steps=20,
    max_repair_iters=6,
    repair_hold_steps=2,
    rollout_opt="physics",
    physics_hold_steps=14,
    success_min_contact=_DEFAULT_MIN_CONTACT,
)

FAST_PEG_SIDE = GraspSideConfig(
    weights=_PEG_WEIGHTS,
    settle_steps_opt=12,
    reach_steps=24,
    pos_bounds_m=0.20,
    success_hand_rmse_m=0.095,
    success_laplacian_rmse_m=0.040,
    maxiter=10,
    n_outer_iters=2,
    maxfun=25,
    optimize=True,
    approach_pre_steps=16,
    approach_grasp_steps=20,
    max_repair_iters=6,
    repair_hold_steps=2,
    rollout_opt="physics",
    physics_hold_steps=14,
    success_min_contact=_DEFAULT_MIN_CONTACT,
)


def side_config(object_name: ObjectName, *, fast: bool = False) -> GraspSideConfig:
    if object_name == "tray":
        return FAST_TRAY_SIDE if fast else TRAY_SIDE
    if object_name == "peg":
        return FAST_PEG_SIDE if fast else PEG_SIDE
    raise ValueError(object_name)
