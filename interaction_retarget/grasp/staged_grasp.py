"""DexGraspBench staged close: pre → grasp → squeeze (separate finger stage).

Refs:
  - refs/DexGraspBench/src/task/eval_func/tabletop_mocap.py (L22–28)
  - refs/Dexonomy/dexonomy/sim/mujoco_env.py get_squeeze_qpos (via squeeze_qpos.py)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from interaction_retarget.grasp.approach import execute_side_approach
from interaction_retarget.grasp.repair import _step_side
from interaction_retarget.sim.settle import read_arm_action, vec_to_arm_action
from interaction_retarget.tpsr.grasp_filter import hand_object_contacts, object_gravity_center_m

Side = Literal["left", "right"]


def derive_squeeze23(
    grasp23: np.ndarray,
    canonical: dict,
    *,
    raw_env=None,
    side: Side | None = None,
    object_name: str | None = None,
    wrench_scale: float = 10.0,
    use_dex_qp: bool = True,
) -> np.ndarray:
    """Squeeze qpos: Dexonomy get_squeeze_qpos when sim contacts exist, else demo median fallback."""
    grasp23 = vec_to_arm_action(grasp23)
    squeeze = grasp23.copy()

    if use_dex_qp and raw_env is not None and side is not None and object_name is not None:
        from interaction_retarget.Dexonomy.dexonomy.sim.squeeze_qpos import squeeze23_from_contacts

        ho = hand_object_contacts(raw_env, side=side, object_name=object_name)  # type: ignore[arg-type]
        ext = np.zeros(6, dtype=np.float64)
        center = object_gravity_center_m(raw_env, object_name)  # type: ignore[arg-type]
        dex_squeeze = squeeze23_from_contacts(
            raw_env,
            side=side,
            grasp23=grasp23,
            ho_contacts=ho,
            ext_wrench=ext,
            ext_center=center,
            wrench_scale=wrench_scale,
        )
        if dex_squeeze is not None:
            return vec_to_arm_action(dex_squeeze)

    target = np.asarray(
        canonical.get("hand_joint_median", squeeze[7:23]), dtype=np.float64
    ).reshape(16)
    squeeze[7:23] = np.maximum(squeeze[7:23], target)
    return squeeze


def re_squeeze_fc(
    raw_env,
    *,
    side: Side,
    object_name: str,
    canonical: dict,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    tpsr_cfg,
    max_rounds: int = 3,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """DexGraspBench squeeze → Dexonomy GraspFilter FC (refs tabletop_mocap + gen_grasp)."""
    from interaction_retarget.tpsr.config import TpsrConfig
    from interaction_retarget.tpsr.grasp_filter import GraspFilter, grasp_filter_cfg_from_tpsr

    cfg: TpsrConfig = tpsr_cfg or TpsrConfig()
    gf_cfg = grasp_filter_cfg_from_tpsr(cfg, ho_collision_thre_m=-0.006)
    base_steps = int(cfg.squeeze_steps)
    squeeze_steps = base_steps + (16 if object_name == "tray" else 20)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    qp_err = 1.0

    for rnd in range(max(int(max_rounds), 1)):
        grasp23 = vec_to_arm_action(read_arm_action(raw_env, side))
        res_pre = GraspFilter(gf_cfg).forward(
            raw_env, side=side, object_name=object_name  # type: ignore[arg-type]
        )
        squeeze23 = derive_squeeze23(
            grasp23,
            canonical,
            raw_env=raw_env,
            side=side,
            object_name=object_name,
            wrench_scale=10.0 if res_pre.ok else 1.0,
            use_dex_qp=bool(res_pre.ok),
        )
        execute_grasp_to_squeeze(
            raw_env,
            side=side,
            grasp23=grasp23,
            squeeze23=squeeze23,
            hold_right=hold_right,
            hold_left=hold_left,
            squeeze_steps=squeeze_steps + rnd * 8,
        )
        res = GraspFilter(gf_cfg).forward(
            raw_env, side=side, object_name=object_name  # type: ignore[arg-type]
        )
        qp_err = float(res.max_qp_error)
        if res.ok or not cfg.require_qp_fc:
            break
        # DexGraspBench tabletop: separate grasp→squeeze; iterate until FC or rounds exhausted.
        grasp23 = read_arm_action(raw_env, side)

    if side == "left":
        hold_left = vec_to_arm_action(read_arm_action(raw_env, "left"))
    else:
        hold_right = vec_to_arm_action(read_arm_action(raw_env, "right"))
    ok = bool(GraspFilter(gf_cfg).forward(raw_env, side=side, object_name=object_name).ok)  # type: ignore[arg-type]
    return hold_right, hold_left, ok, qp_err


def prepare_lift_squeeze(
    raw_env,
    *,
    side: Side,
    object_name: str,
    canonical: dict,
    grasp_arm: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    tpsr_cfg,
    settle_steps: int = 12,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Settle; Dexonomy squeeze only when static QP still fails badly."""
    from interaction_retarget.tpsr.grasp_filter import GraspFilter, grasp_filter_cfg_from_tpsr

    grasp_arm = vec_to_arm_action(grasp_arm)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    active = grasp_arm if side == "left" else hold_right
    passive = hold_right if side == "left" else hold_left
    for _ in range(max(int(settle_steps), 1)):
        if side == "left":
            from interaction_retarget.sim.settle import settle_bimanual_actions

            settle_bimanual_actions(raw_env, right23=passive, left23=active, n_substeps=2)
        else:
            from interaction_retarget.sim.settle import settle_bimanual_actions

            settle_bimanual_actions(raw_env, right23=active, left23=passive, n_substeps=2)
    if side == "left":
        hold_left = active
    else:
        hold_right = active

    gf_cfg = grasp_filter_cfg_from_tpsr(tpsr_cfg, ho_collision_thre_m=-0.006)
    res = GraspFilter(gf_cfg).forward(raw_env, side=side, object_name=object_name)  # type: ignore[arg-type]
    if res.ok or float(res.max_qp_error) < float(getattr(tpsr_cfg, "qp_hold_soft_thre", 0.35)):
        return hold_right, hold_left, bool(res.ok), float(res.max_qp_error)
    return re_squeeze_fc(
        raw_env,
        side=side,
        object_name=object_name,
        canonical=canonical,
        hold_right=hold_right,
        hold_left=hold_left,
        tpsr_cfg=tpsr_cfg,
    )


def execute_grasp_to_squeeze(
    raw_env,
    *,
    side: Side,
    grasp23: np.ndarray,
    squeeze23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    squeeze_steps: int = 12,
) -> None:
    """DexGraspBench tabletop: grasp_qpos → squeeze_qpos (fingers only, not merged)."""
    from interaction_retarget.DexGraspBench.src.util.rot_util import interplote_qpos

    grasp23 = vec_to_arm_action(grasp23)
    squeeze23 = vec_to_arm_action(squeeze23)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    n = max(int(squeeze_steps), 1)
    for active in interplote_qpos(grasp23, squeeze23, n):
        _step_side(
            raw_env,
            side=side,
            active23=np.asarray(active, dtype=np.float64),
            hold_right=hold_right,
            hold_left=hold_left,
        )


def execute_pre_grasp_grasp_squeeze(
    raw_env,
    *,
    side: Side,
    home23: np.ndarray,
    pre23: np.ndarray,
    grasp23: np.ndarray,
    squeeze23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    pre_steps: int = 18,
    grasp_steps: int = 12,
    squeeze_steps: int = 12,
) -> None:
    """Full DexGraspBench open-loop: pre → grasp → squeeze (three separate stages)."""
    execute_side_approach(
        raw_env,
        side=side,
        home=home23,
        pre_grasp=pre23,
        grasp=grasp23,
        hold_right=hold_right,
        hold_left=hold_left,
        pre_steps=pre_steps,
        grasp_steps=grasp_steps,
    )
    execute_grasp_to_squeeze(
        raw_env,
        side=side,
        grasp23=grasp23,
        squeeze23=squeeze23,
        hold_right=hold_right,
        hold_left=hold_left,
        squeeze_steps=squeeze_steps,
    )


def execute_direct_reach_grasp_squeeze(
    raw_env,
    *,
    side: Side,
    grasp23: np.ndarray,
    squeeze23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    reach_steps: int = 60,
    close_steps: int = 16,
    squeeze_steps: int = 12,
    start23: np.ndarray | None = None,
) -> None:
    """Direct reach with open hand, close to grasp, then separate squeeze stage."""
    from interaction_retarget.grasp.approach import reach_arm_then_close

    grasp23 = vec_to_arm_action(grasp23)
    squeeze23 = vec_to_arm_action(squeeze23)
    reach_arm_then_close(
        raw_env,
        side=side,
        target23=grasp23,
        hold_right=hold_right,
        hold_left=hold_left,
        reach_steps=reach_steps,
        close_steps=close_steps,
        start23=start23,
    )
    execute_grasp_to_squeeze(
        raw_env,
        side=side,
        grasp23=grasp23,
        squeeze23=squeeze23,
        hold_right=hold_right,
        hold_left=hold_left,
        squeeze_steps=squeeze_steps,
    )
