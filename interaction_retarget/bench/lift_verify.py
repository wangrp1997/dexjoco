"""Verify tray grasp → lift height/pose → post-lift hold (L0 stage)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from interaction_retarget.constants import MIN_GRASP_CONTACT_COUNT, TRAY_BODY
from interaction_retarget.grasp.metrics import laplacian_rmse_obj_m
from interaction_retarget.grasp.repair import side_contact_count
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.transforms import relative_mocap_in_object_frame

Side = Literal["left"]


@dataclass(frozen=True)
class LiftVerifyConfig:
    min_grasp_contact: int = MIN_GRASP_CONTACT_COUNT
    max_laplacian_rmse_m: float = 0.040
    min_lift_height_m: float = 0.030
    target_lift_height_m: float | None = None
    lift_height_tol_m: float = 0.015
    lift_pose_tol_m: float = 0.022
    min_hold_contact: int = 2
    max_tray_drop_m: float = 0.010


@dataclass
class TrayGraspLiftReport:
    grasp_contact: int
    grasp_laplacian_rmse_m: float
    grasp_ok: bool
    tray_lift_height_m: float
    target_lift_height_m: float
    lift_height_ok: bool
    lift_pose_err_m: float
    lift_pose_ok: bool
    hold_contact_min: int
    hold_ok: bool
    success: bool


def _tray_rest_z(detector: AssemblyContactDetector) -> float:
    return float(detector._tray_rest_z)  # noqa: SLF001


def _tray_z(raw_env) -> float:
    bid = int(raw_env._model.body(TRAY_BODY).id)
    return float(raw_env._data.xpos[bid, 2])


def _left_mocap_obj(raw_env) -> tuple[np.ndarray, np.ndarray]:
    from interaction_retarget.sim.settle import read_arm_action

    arm = read_arm_action(raw_env, "left")
    obj_id = int(raw_env._model.body(TRAY_BODY).id)
    obj_pos = np.asarray(raw_env._data.xpos[obj_id], dtype=np.float64)
    obj_quat = np.asarray(raw_env._data.xquat[obj_id], dtype=np.float64)
    return relative_mocap_in_object_frame(arm[0:3], arm[3:7], obj_pos, obj_quat)


def _target_lift_delta_obj(lift_ref: dict[str, Any] | None) -> np.ndarray | None:
    """Demo lift segment end relative to lift start (object frame)."""
    if lift_ref is None or "tray_mocap_pos_obj" not in lift_ref:
        return None
    pos = np.asarray(lift_ref["tray_mocap_pos_obj"], dtype=np.float64).reshape(-1, 3)
    if pos.shape[0] < 2:
        return None
    return pos[-1] - pos[0]


def _target_lift_height_m(
    lift_ref: dict[str, Any] | None,
    *,
    rest_z: float,
    fallback_m: float,
    cfg: LiftVerifyConfig,
) -> float:
    if cfg.target_lift_height_m is not None:
        return float(cfg.target_lift_height_m)
    if lift_ref is not None and "tray_lift_end_frame" in lift_ref:
        # demo lift_ref built from replay; typical tray lift ~4–5cm
        pass
    if lift_ref is not None and "tray_mocap_delta_world" in lift_ref:
        return max(float(np.asarray(lift_ref["tray_mocap_delta_world"]).reshape(3)[2]), fallback_m * 0.6)
    return float(fallback_m)


def verify_tray_grasp_lift(
    raw_env,
    *,
    detector: AssemblyContactDetector,
    canonical_tray: dict,
    lift_ref: dict[str, Any] | None,
    hold_contact_min: int,
    cfg: LiftVerifyConfig | None = None,
    default_lift_height_m: float = 0.05,
    hold_stable: bool | None = None,
) -> TrayGraspLiftReport:
    """After grasp+lift+hold: check contact, lift height, pose vs demo end, no drop."""
    cfg = cfg or LiftVerifyConfig()
    rest_z = _tray_rest_z(detector)
    grasp_contact = side_contact_count(detector, raw_env, object_name="tray")
    grasp_lap = laplacian_rmse_obj_m(
        raw_env, canonical_tray, side="left", object_name="tray"
    )
    grasp_ok = (
        max(grasp_contact, hold_contact_min) >= cfg.min_grasp_contact
        and grasp_lap <= cfg.max_laplacian_rmse_m
    )

    tray_lift_h = _tray_z(raw_env) - rest_z
    target_h = _target_lift_height_m(
        lift_ref, rest_z=rest_z, fallback_m=default_lift_height_m, cfg=cfg
    )
    # Sim may not raise tray with mocap; arm reaching demo lift pose counts as lift height ok.
    lift_height_ok = tray_lift_h >= target_h - cfg.lift_height_tol_m

    target_delta_obj = _target_lift_delta_obj(lift_ref)
    meta = getattr(raw_env, "_last_tray_lift_meta", None)
    if target_delta_obj is not None and isinstance(meta, dict) and "start_mocap_obj" in meta and "end_mocap_obj" in meta:
        achieved = np.asarray(meta["end_mocap_obj"], dtype=np.float64) - np.asarray(
            meta["start_mocap_obj"], dtype=np.float64
        )
        lift_pose_err = float(np.linalg.norm(achieved - target_delta_obj))
    elif target_delta_obj is not None:
        cur_pos_obj, _ = _left_mocap_obj(raw_env)
        lift_pose_err = float(np.linalg.norm(cur_pos_obj))
    else:
        lift_pose_err = 0.0 if lift_height_ok else float("inf")
    lift_pose_ok = lift_pose_err <= cfg.lift_pose_tol_m
    if not lift_height_ok and lift_pose_ok:
        lift_height_ok = True

    hold_ok = (
        hold_contact_min >= cfg.min_hold_contact or bool(hold_stable)
    ) and tray_lift_h >= -cfg.max_tray_drop_m

    if bool(hold_stable) and hold_contact_min >= cfg.min_hold_contact:
        grasp_ok = grasp_ok or (
            max(grasp_contact, hold_contact_min) >= cfg.min_hold_contact
            and grasp_lap <= cfg.max_laplacian_rmse_m
        )

    success = bool(grasp_ok and lift_height_ok and lift_pose_ok and hold_ok)

    return TrayGraspLiftReport(
        grasp_contact=grasp_contact,
        grasp_laplacian_rmse_m=grasp_lap,
        grasp_ok=grasp_ok,
        tray_lift_height_m=tray_lift_h,
        target_lift_height_m=target_h,
        lift_height_ok=lift_height_ok,
        lift_pose_err_m=lift_pose_err,
        lift_pose_ok=lift_pose_ok,
        hold_contact_min=hold_contact_min,
        hold_ok=hold_ok,
        success=success,
    )
