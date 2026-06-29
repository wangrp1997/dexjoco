"""TPSR metric helpers (shared by refine + sim_refine)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from interaction_retarget.grasp.metrics import hand_rmse_obj_m, laplacian_rmse_obj_m
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.tpsr.config import TpsrConfig
from interaction_retarget.tpsr.constraints import hole_clearance_violation_m

ObjectName = Literal["tray", "peg"]
Side = Literal["left", "right"]


@dataclass
class TpsrMetrics:
    laplacian_rmse_m: float
    hand_rmse_m: float
    hole_violation_m: float
    contact_count: int


def hole_params(cfg: TpsrConfig, object_name: ObjectName) -> tuple[float, float]:
    if object_name == "peg":
        return cfg.peg_insert_clearance_radius_m, cfg.peg_insert_guard_length_m
    return cfg.tray_socket_clearance_radius_m, cfg.tray_socket_guard_depth_m


def tpsr_metrics(
    raw_env,
    canonical: dict,
    *,
    object_name: ObjectName,
    side: Side,
    detector: AssemblyContactDetector,
    cfg: TpsrConfig | None = None,
) -> TpsrMetrics:
    cfg = cfg or TpsrConfig()
    contact = detector.compute(raw_env)
    count = int(
        contact.tray_contact_count if object_name == "tray" else contact.peg_contact_count
    )
    radius_m, length_m = hole_params(cfg, object_name)
    return TpsrMetrics(
        laplacian_rmse_m=laplacian_rmse_obj_m(
            raw_env, canonical, side=side, object_name=object_name
        ),
        hand_rmse_m=hand_rmse_obj_m(raw_env, canonical, side=side, object_name=object_name),
        hole_violation_m=hole_clearance_violation_m(
            raw_env,
            object_name=object_name,
            side=side,
            cfg_radius_m=radius_m,
            cfg_length_m=length_m,
        ),
        contact_count=count,
    )
