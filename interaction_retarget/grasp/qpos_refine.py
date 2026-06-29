"""Refine action23 to match demo MuJoCo contacts + DexGraspBench FC (scipy, no Lap).

Refs:
  - refs/contactopt/contactopt/optimize_pose.py (L-BFGS-B loop)
  - refs/GenHand/optimisation/Optimize.py (contact anchor + FC)
  - interaction_retarget/tpsr/grasp_filter.py (GraspFilter / GraspQP)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize

from interaction_retarget.grasp.contact_targets import (
    ContactTargetSet,
    contact_count_shortfall,
    contact_match_rmse_m,
)
from interaction_retarget.grasp.repair import _hand_joint_bounds, _step_side
from interaction_retarget.sim.settle import read_arm_action, vec_to_arm_action
from interaction_retarget.sim.state import restore_sim, snapshot_sim
from interaction_retarget.tpsr.grasp_filter import GraspFilter, GraspFilterConfig, grasp_filter_cfg_from_tpsr
from interaction_retarget.tpsr.config import TpsrConfig

Side = Literal["left", "right"]
ObjectName = Literal["tray", "peg"]


@dataclass(frozen=True)
class QposRefineConfig:
    maxiter: int = 48
    w_contact_rmse: float = 800.0
    w_contact_count: float = 120.0
    w_qp: float = 80.0
    w_reg_trans: float = 40.0
    w_reg_finger: float = 8.0
    max_mocap_trans_m: float = 0.012
    max_finger_delta: float = 0.12
    contact_dist_thre_m: float = 0.002
    min_contacts: int = 3
    settle_substeps: int = 2


@dataclass
class QposRefineResult:
    action23: np.ndarray
    contact_rmse_m: float
    contact_count: int
    qp_error: float
    fc_ok: bool
    success: bool
    nit: int
    message: str


def _side_for_object(object_name: ObjectName) -> Side:
    return "left" if object_name == "tray" else "right"


def _pack_delta(base23: np.ndarray, delta: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    base23 = vec_to_arm_action(base23)
    d = np.asarray(delta, dtype=np.float64).reshape(19)
    out = base23.copy()
    out[0:3] = base23[0:3] + np.clip(d[0:3], -0.02, 0.02)
    out[7:23] = np.clip(base23[7:23] + d[3:19], lo, hi)
    return out


def refine_side_qpos_contacts(
    raw_env,
    *,
    base23: np.ndarray,
    targets: ContactTargetSet,
    object_name: ObjectName,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    cfg: QposRefineConfig | None = None,
    tpsr_cfg: TpsrConfig | None = None,
) -> QposRefineResult:
    """L-BFGS-B: mocap trans + fingers to match demo contacts; FC via GraspFilter."""
    cfg = cfg or QposRefineConfig()
    tpsr_cfg = tpsr_cfg or TpsrConfig()
    side = _side_for_object(object_name)
    base23 = vec_to_arm_action(base23)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)

    gf_cfg = grasp_filter_cfg_from_tpsr(tpsr_cfg, ho_collision_thre_m=-0.006)
    lo, hi = _hand_joint_bounds(raw_env._model, side)
    snap = snapshot_sim(raw_env)

    trans_bound = float(cfg.max_mocap_trans_m)
    finger_bound = float(cfg.max_finger_delta)
    bounds = [(-trans_bound, trans_bound)] * 3 + [(-finger_bound, finger_bound)] * 16

    best = {
        "x": np.zeros(19, dtype=np.float64),
        "loss": float("inf"),
        "rmse": float("inf"),
        "cc": 0,
        "qp": 1.0,
        "fc_ok": False,
    }

    def objective(x: np.ndarray) -> float:
        restore_sim(raw_env, snap)
        cand = _pack_delta(base23, x, lo, hi)
        _step_side(
            raw_env,
            side=side,
            active23=cand,
            hold_right=hold_right,
            hold_left=hold_left,
        )
        rmse, n_cur, _ = contact_match_rmse_m(
            raw_env,
            targets,
            side=side,
            object_name=object_name,
            contact_dist_thre_m=cfg.contact_dist_thre_m,
        )
        if not np.isfinite(rmse):
            return 1e6
        shortfall = contact_count_shortfall(
            raw_env,
            side=side,
            object_name=object_name,
            min_contacts=cfg.min_contacts,
            contact_dist_thre_m=cfg.contact_dist_thre_m,
        )
        gf = GraspFilter(gf_cfg).forward(raw_env, side=side, object_name=object_name)
        qp_err = float(gf.max_qp_error)
        reg_t = float(np.linalg.norm(x[0:3]))
        reg_f = float(np.linalg.norm(x[3:19]))
        loss = (
            cfg.w_contact_rmse * rmse
            + cfg.w_contact_count * shortfall
            + cfg.w_qp * qp_err
            + cfg.w_reg_trans * reg_t
            + cfg.w_reg_finger * reg_f
        )
        if loss < best["loss"]:
            best.update(loss=loss, rmse=rmse, cc=n_cur, qp=qp_err, fc_ok=bool(gf.ok), x=x.copy())
        return loss

    x0 = np.zeros(19, dtype=np.float64)
    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(cfg.maxiter), "ftol": 1e-9},
    )

    restore_sim(raw_env, snap)
    final23 = _pack_delta(base23, best["x"], lo, hi)
    _step_side(
        raw_env,
        side=side,
        active23=final23,
        hold_right=hold_right,
        hold_left=hold_left,
    )
    for _ in range(max(int(cfg.settle_substeps), 1)):
        _step_side(
            raw_env,
            side=side,
            active23=final23,
            hold_right=hold_right,
            hold_left=hold_left,
        )

    rmse, n_cur, _ = contact_match_rmse_m(
        raw_env,
        targets,
        side=side,
        object_name=object_name,
        contact_dist_thre_m=cfg.contact_dist_thre_m,
    )
    gf_final = GraspFilter(gf_cfg).forward(raw_env, side=side, object_name=object_name)
    fc_ok = bool(gf_final.ok)
    success = fc_ok and n_cur >= cfg.min_contacts and np.isfinite(rmse) and rmse < 0.004

    return QposRefineResult(
        action23=vec_to_arm_action(read_arm_action(raw_env, side)),
        contact_rmse_m=float(rmse),
        contact_count=int(n_cur),
        qp_error=float(gf_final.max_qp_error),
        fc_ok=fc_ok,
        success=bool(success),
        nit=int(getattr(res, "nit", 0)),
        message=str(getattr(res, "message", "")),
    )
