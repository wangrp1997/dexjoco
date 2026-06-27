"""Per-arm defaults (demo grasp-frame counts from sidecar timing)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from interaction_retarget.grasp.ik import IkWeights, RolloutOpt

ObjectName = Literal["tray", "peg"]


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
)
_PEG_WEIGHTS = IkWeights(
    laplacian=100.0,
    global_hand=15.0,
    local_hand=8.0,
    site_tracking=8.0,
    joint_regularization=0.005,
    contact=400.0,
)

# pre≈45% of demo grasp frame (GenHand trajectory pre-grasp phase)
TRAY_SIDE = GraspSideConfig(
    weights=_TRAY_WEIGHTS,
    settle_steps_opt=25,
    reach_steps=50,
    pos_bounds_m=0.18,
    success_hand_rmse_m=0.085,
    success_laplacian_rmse_m=0.030,
    maxiter=12,
    n_outer_iters=2,
    maxfun=40,
    optimize=True,
    approach_pre_steps=65,
    approach_grasp_steps=80,
    max_repair_iters=24,
)

PEG_SIDE = GraspSideConfig(
    weights=_PEG_WEIGHTS,
    settle_steps_opt=20,
    reach_steps=55,
    pos_bounds_m=0.20,
    success_hand_rmse_m=0.085,
    success_laplacian_rmse_m=0.032,
    maxiter=14,
    n_outer_iters=2,
    maxfun=45,
    optimize=True,
    approach_pre_steps=104,
    approach_grasp_steps=126,
    max_repair_iters=24,
    rollout_opt="physics",
    physics_hold_steps=16,
)


def side_config(object_name: ObjectName) -> GraspSideConfig:
    if object_name == "tray":
        return TRAY_SIDE
    if object_name == "peg":
        return PEG_SIDE
    raise ValueError(object_name)
