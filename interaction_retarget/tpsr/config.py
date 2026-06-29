"""TPSR hyper-parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TpsrConfig:
    max_iters: int = 28
    hold_steps: int = 2
    sim_close_steps: int = 16
    sim_settle_steps: int = 12
    sim_search_iters: int = 24
    bootstrap_lap_drift_scale: float = 1.45
    max_laplacian_drift_m: float = 0.055
    max_hand_drift_m: float = 0.100
    min_contact_count: int = 3
    finger_only: bool = False
    require_on_table: bool = False
    # Peg: fingertips must stay outside insert-end cone (m).
    peg_insert_clearance_radius_m: float = 0.010
    peg_insert_guard_length_m: float = 0.025
    # Tray: keep tips away from socket opening cylinder (m).
    tray_socket_clearance_radius_m: float = 0.012
    tray_socket_guard_depth_m: float = 0.020
    # Dexonomy grasp.yaml filter; optional DexGraspBench calcu_qp_metric
    require_qp_fc: bool = True
    qp_error_thre: float = 0.001
    qp_miu_coef: tuple[float, float] = (0.6, 0.02)
    qp_contact_dist_thre_m: float = 0.002
    qp_robust_metric: bool = False
    qp_robust_metric_thre: float = 0.001
    # DexGraspBench fc_mocap fallback: QP feasible but above filter thre → sim hold.
    qp_hold_soft_thre: float = 0.35
    squeeze_steps: int = 12
