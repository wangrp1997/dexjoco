"""Deploy: retrieve demo n → median δ* grasp → contact/FC → per-demo lift blend → insert."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Literal

import numpy as np

from interaction_retarget.bench.config import BenchConfig
from interaction_retarget.bench.lift_verify import LiftVerifyConfig, verify_tray_grasp_lift
from interaction_retarget.bench.verify import BenchHoldReport, verify_side_hold
from interaction_retarget.constants import CONTACT_WINDOW, MIN_GRASP_CONTACT_COUNT, PEG_BODY, TRAY_BODY
from interaction_retarget.grasp.agent_tpsr import make_peg_agent_tpsr, make_tray_agent_tpsr
from interaction_retarget.grasp.ik import GraspIkResult, mocap_world_from_canonical
from interaction_retarget.grasp.lift import (
    DEFAULT_TRAY_LIFT_M,
    execute_peg_lift,
    execute_tray_lift,
    hold_tray_before_peg,
    verify_lift_fc,
)
from interaction_retarget.grasp.locked_hold import enforce_locked_passive
from interaction_retarget.grasp.dual_async import run_l1_dual_async
from interaction_retarget.grasp.plan_side_grasp import plan_side_grasp
from interaction_retarget.grasp.pipeline import _ik_grasp_ready
from interaction_retarget.grasp.pipeline_tpsr import _bimanual_fail_reason, _canonical_arm_target, _skipped_peg_ik
from interaction_retarget.grasp.repair import (
    GraspRepairResult,
    SideContactMetrics,
    laplacian_rmse,
    prepare_lift_grasp,
    repair_side_grasp,
    side_contact_count,
    verify_grasp_hold,
)
from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.replay import make_assembly_env
from interaction_retarget.sim.settle import read_arm_action, settle_bimanual_actions, vec_to_arm_action
from interaction_retarget.sim.state import restore_sim, snapshot_sim
from interaction_retarget.skill_replay.trajectory_audit import (
    MIN_PEG_LIFT_M,
    MIN_TRAY_LIFT_M,
    TrajectoryAudit,
    snapshot_phase,
    validate_delivery,
)
from interaction_retarget.sim.video import DexEnvVideoRecorder, exec_recording, reset_sim_step, sim_step_count
from interaction_retarget.grasp.staged_grasp import derive_squeeze23, execute_pre_grasp_grasp_squeeze, re_squeeze_fc
from interaction_retarget.grasp.pre_grasp import derive_pre_grasp_from_grasp
from interaction_retarget.skill_replay.grasp_finalize import GraspFinalizeReport, finalize_side_grasp
from interaction_retarget.skill_replay.demo_canonical import enrich_canonical_on_env
from interaction_retarget.skill_replay.demo_grasp import (
    apply_demo_grasp_frame,
    demo_grasp_arm23,
    demo_lift_world_dz,
    demo_video_phase_frames,
    replay_demo_privileged_video,
    replay_demo_segment,
)
from interaction_retarget.skill_replay.insert import InsertReport, run_hybrid_insert
from interaction_retarget.skill_replay.library import DemoSkill, SkillLibrary
from interaction_retarget.skill_replay.retrieval import ObjectPose, ScenePose, nearest_demo_index
from interaction_retarget.tpsr.config import TpsrConfig
from interaction_retarget.tpsr.metrics import tpsr_metrics


def _ik_action_target(ik: GraspIkResult, side: Literal["left", "right"]) -> np.ndarray:
    return vec_to_arm_action(ik.action_left if side == "left" else ik.action_right)


def _topology_grasp_ok(
    raw_env,
    detector: AssemblyContactDetector,
    canonical: dict,
    *,
    object_name: Literal["tray", "peg"],
    tpsr_cfg: TpsrConfig,
    max_hand_rmse_m: float | None = None,
    max_lap_rmse_m: float = 0.040,
) -> tuple[bool, str]:
    side: Literal["left", "right"] = "left" if object_name == "tray" else "right"
    if max_hand_rmse_m is None:
        max_hand_rmse_m = 0.055 if object_name == "tray" else 0.095
    m = tpsr_metrics(
        raw_env, canonical, object_name=object_name, side=side, detector=detector, cfg=tpsr_cfg
    )
    if m.hand_rmse_m > max_hand_rmse_m:
        return False, f"hand_rmse={m.hand_rmse_m * 1e3:.1f}mm"
    if m.laplacian_rmse_m > max_lap_rmse_m:
        return False, f"lap={m.laplacian_rmse_m * 1e3:.1f}mm"
    if m.contact_count < MIN_GRASP_CONTACT_COUNT:
        return False, f"contact={m.contact_count}"
    return True, "ok"


def _record_tpsr_cfg(cfg: TpsrConfig) -> TpsrConfig:
    """Playback only: GenHand ramps, no sim_refine mocap search."""
    return replace(cfg, sim_search_iters=0, max_iters=0)


def _object_lift_m(raw_env, detector: AssemblyContactDetector, object_name: Literal["tray", "peg"]) -> float:
    body = TRAY_BODY if object_name == "tray" else PEG_BODY
    bid = int(raw_env._model.body(body).id)
    rest = float(detector._tray_rest_z if object_name == "tray" else detector._peg_rest_z)  # noqa: SLF001
    return float(raw_env._data.xpos[bid, 2]) - rest


L1_MIN_TRAY_LIFT_M = 0.018


def _tray_ready_for_lift(
    raw_env,
    detector: AssemblyContactDetector,
    *,
    tray_topo_ok: bool,
    tray_fc_ok: bool,
    l1_mode: bool,
) -> bool:
    cc = int(side_contact_count(detector, raw_env, object_name="tray"))
    if not tray_topo_ok or cc < MIN_GRASP_CONTACT_COUNT:
        return False
    if tray_topo_ok and tray_fc_ok:
        return True
    if not l1_mode:
        return False
    return cc >= MIN_GRASP_CONTACT_COUNT


def _peg_phase_ready(
    *,
    l1_mode: bool,
    tray_topo_ok: bool,
    tray_fc_ok: bool,
    tray_lifted: bool,
    tray_grasp_lift,
    tray_lift_m: float,
    tray_contact_pre_peg: int,
    tray_lift_hold_stable: bool | None,
) -> bool:
    if l1_mode:
        return bool(
            tray_lifted
            and tray_lift_m >= MIN_TRAY_LIFT_M
            and tray_contact_pre_peg >= max(2, MIN_GRASP_CONTACT_COUNT - 1)
        )
    return bool(
        tray_topo_ok
        and tray_fc_ok
        and tray_lifted
        and tray_grasp_lift.success
        and tray_grasp_lift.lift_height_ok
        and tray_lift_m >= MIN_TRAY_LIFT_M
    )


from interaction_retarget.grasp.plan_side_grasp import plan_side_grasp


def _playback_grasp_to_pose(
    raw_env,
    *,
    side: Literal["left", "right"],
    object_name: Literal["tray", "peg"],
    canonical: dict,
    grasp23: np.ndarray,
    hold_right: np.ndarray,
    hold_left: np.ndarray,
    pre_steps: int,
    grasp_steps: int,
    squeeze_steps: int,
) -> np.ndarray:
    """Recorded motion: home → pre → grasp → squeeze (no optimization)."""
    grasp23 = vec_to_arm_action(grasp23)
    hold_right = vec_to_arm_action(hold_right)
    hold_left = vec_to_arm_action(hold_left)
    home23 = read_arm_action(raw_env, side)
    pre23 = derive_pre_grasp_from_grasp(grasp23, side=side).action23
    squeeze23 = derive_squeeze23(
        grasp23, canonical, raw_env=raw_env, side=side, object_name=object_name
    )
    execute_pre_grasp_grasp_squeeze(
        raw_env,
        side=side,
        home23=home23,
        pre23=pre23,
        grasp23=grasp23,
        squeeze23=squeeze23,
        hold_right=hold_right,
        hold_left=hold_left,
        pre_steps=pre_steps,
        grasp_steps=grasp_steps,
        squeeze_steps=squeeze_steps,
    )
    return vec_to_arm_action(read_arm_action(raw_env, side))


def _exec_tpsr_cfg(cfg: TpsrConfig) -> TpsrConfig:
    """Lift squeeze/FC during playback (no sim search)."""
    return replace(cfg, sim_search_iters=0, max_iters=0)


def _current_scene_pose(raw_env) -> ScenePose:
    def _body(name: str) -> ObjectPose:
        bid = int(raw_env._model.body(name).id)
        data = raw_env._data
        return ObjectPose(
            pos=np.asarray(data.xpos[bid], dtype=np.float64).copy(),
            quat=np.asarray(data.xquat[bid], dtype=np.float64).copy(),
        )

    return ScenePose(tray=_body(TRAY_BODY), peg=_body(PEG_BODY))


@dataclass
class SkillReplayReport:
    seed: int
    demo_episode_index: int
    retrieval_distance_m: float
    tray_ik: GraspIkResult
    peg_ik: GraspIkResult
    repair: GraspRepairResult
    tray_laplacian_rmse_m: float
    peg_laplacian_rmse_m: float
    tray_lift_hold_stable: bool | None = None
    bench_tray: BenchHoldReport | None = None
    bench_peg: BenchHoldReport | None = None
    insert: InsertReport | None = None
    success: bool = False
    fail_reason: str = ""
    stage: str = "full"
    extra: dict[str, Any] = field(default_factory=dict)


def _manifest_entry(library: SkillLibrary, episode_index: int) -> dict[str, Any]:
    return next(
        e for e in library._entries if int(e["episode_index"]) == int(episode_index)
    )


def _restore_demo_layout(env, entry: dict[str, Any], detector: AssemblyContactDetector) -> None:
    _, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    if initial_state is None or not has_restorer("bimanual_assembly"):
        return
    config = CONFIG_MAPPING["bimanual_assembly"]()
    restore_initial_state(env, "bimanual_assembly", config, initial_state)
    detector.reset_reference(env.unwrapped)


def retrieve_demo(
    library: SkillLibrary,
    raw_env,
    *,
    force_episode: int | None = None,
    use_per_demo_canonical: bool = True,
) -> tuple[DemoSkill, float]:
    if force_episode is not None:
        skill = library.load_demo(
            int(force_episode), use_per_demo_canonical=use_per_demo_canonical
        )
        return skill, 0.0
    current = _current_scene_pose(raw_env)
    pose_index = library.build_pose_index()
    ep_idx, dist = nearest_demo_index(current, pose_index)
    return library.load_demo(ep_idx, use_per_demo_canonical=use_per_demo_canonical), float(dist)


def run_skill_replay(
    *,
    sidecar_dir: Path,
    seed: int = 0,
    hold_steps: int = 20,
    tray_lift_height_m: float = DEFAULT_TRAY_LIFT_M,
    tray_lift_steps: int | None = None,
    peg_lift_steps: int | None = None,
    tray_hold_max_steps: int = 72,
    hold_warmup_steps: int = 10,
    skip_insert: bool = False,
    skip_peg_lift: bool | None = None,
    force_demo_episode: int | None = None,
    restore_demo_layout: bool = False,
    fast: bool = True,
    tpsr_cfg: TpsrConfig | None = None,
    bench_cfg: BenchConfig | None = None,
    seed_base: int = 0,
    video_out: Path | None = None,
    video_fps: int = 30,
    allow_partial_video: bool = False,
    l1_mode: bool = True,
) -> SkillReplayReport:
    """Retrieve nearest demo for lift; grasp uses per-demo or median δ* + contact/FC."""
    sidecar_dir = Path(sidecar_dir)
    if skip_peg_lift is None:
        skip_peg_lift = False
    library = SkillLibrary(sidecar_dir, seed_base=seed_base, exclude_fallback=True)
    tpsr_cfg = tpsr_cfg or TpsrConfig()
    if fast:
        tpsr_cfg = replace(
            tpsr_cfg,
            max_iters=min(tpsr_cfg.max_iters, 10),
            sim_search_iters=min(tpsr_cfg.sim_search_iters, 10),
        )
    bench_cfg = bench_cfg or BenchConfig(hold_steps=hold_steps, warmup_steps=hold_warmup_steps)

    if fast:
        tray_hold_max_steps = min(tray_hold_max_steps, 16)
        hold_steps = min(hold_steps, 8)
        hold_warmup_steps = min(hold_warmup_steps, 4)
        lift_pre_settle = 4
        plan_lift_cap = 96 if l1_mode else 48
        record_lift_cap = 48
    else:
        lift_pre_settle = 12
        plan_lift_cap = None
        record_lift_cap = None
    contact_refine_iters = 16 if fast else 24
    lift_cfg = _exec_tpsr_cfg(tpsr_cfg)
    report_success = False
    deliver_video = False
    video_partial = False

    env = make_assembly_env(seed=int(seed), randomize=False)
    raw = env.unwrapped
    detector = AssemblyContactDetector(raw)
    demo_skill: DemoSkill | None = None
    retrieval_dist = float("nan")
    audit = TrajectoryAudit()
    video_rec: DexEnvVideoRecorder | None = None
    if video_out is not None:
        video_rec = DexEnvVideoRecorder(env, Path(video_out), fps=int(video_fps))
    on_frame = video_rec.capture if video_rec is not None else None

    def _mark(phase: str) -> None:
        audit.phases.append(
            snapshot_phase(raw, detector, phase=phase, step=sim_step_count())
        )

    try:
        env.reset()
        detector.reset_reference(raw)
        reset_sim_step()
        demo_skill, retrieval_dist = retrieve_demo(
            library,
            raw,
            force_episode=force_demo_episode,
            use_per_demo_canonical=l1_mode,
        )
        if restore_demo_layout or force_demo_episode is not None:
            _restore_demo_layout(env, _manifest_entry(library, demo_skill.episode_index), detector)
        canonical_tray = demo_skill.tray_canonical
        canonical_peg = demo_skill.peg_canonical
        manifest_entry = _manifest_entry(library, demo_skill.episode_index)
        if l1_mode and demo_skill.per_demo_canonical:
            enrich_canonical_on_env(raw, manifest_entry, canonical_tray, "tray")
            enrich_canonical_on_env(raw, manifest_entry, canonical_peg, "peg")
        demo_lift_ref = demo_skill.lift_ref
        lift_ref = demo_lift_ref

        left_agent = make_tray_agent_tpsr(canonical_tray, fast=fast)
        left_agent.tpsr_cfg = tpsr_cfg
        right_agent = make_peg_agent_tpsr(canonical_peg, fast=fast)
        right_agent.tpsr_cfg = tpsr_cfg

        right_home = vec_to_arm_action(read_arm_action(raw, "right"))
        left_home = vec_to_arm_action(read_arm_action(raw, "left"))
        right_hold = right_home.copy()
        left_hold = left_home.copy()
        repair_iters = 0

        sim_pre_tray = snapshot_sim(raw)
        use_privileged_demo = bool(
            force_demo_episode is not None and restore_demo_layout and not l1_mode
        )
        use_demo_grasp = use_privileged_demo
        if use_demo_grasp:
            left_target = demo_grasp_arm23(raw, manifest_entry, "tray")
            _, tray_ik = left_agent.plan(
                raw,
                hold_right=right_hold,
                hold_left=left_hold,
                detector=detector,
                restore_env=True,
                side_cfg=left_agent.side_cfg,
            )
        elif l1_mode:
            left_target = _canonical_arm_target(raw, canonical_tray, "tray")
            _, tray_ik = left_agent.plan(
                raw,
                hold_right=right_hold,
                hold_left=left_hold,
                detector=detector,
                restore_env=True,
                side_cfg=left_agent.side_cfg,
            )
        else:
            left_target, tray_ik = left_agent.plan(
                raw,
                hold_right=right_hold,
                hold_left=left_hold,
                detector=detector,
                restore_env=True,
                side_cfg=left_agent.side_cfg,
            )
            left_target = _ik_action_target(tray_ik, "left")
        tray_topo_ok = False
        tray_topo_msg = "pending"
        tray_lifted = False
        tray_fc_ok = False
        tray_qp_err = 1.0
        tray_grasp_pose = left_target.copy()
        tray_lift_hold_stable: bool | None = None
        hold_contact_min = 0
        peg_ik = _skipped_peg_ik()
        peg_topo_ok = False
        peg_topo_msg = "skipped"
        peg_fc_ok = False
        peg_qp_err = 1.0
        peg_lifted = False
        peg_grasp_pose = right_home.copy()
        peg_lift_m = 0.0
        tray_lift_m = 0.0

        # --- Plan (+ L1 live record): tray grasp + CP + lift ---
        l1_live_record = bool(l1_mode and video_out is not None and not use_privileged_demo)
        _plan_ctx = exec_recording(on_frame) if l1_live_record else nullcontext()
        _plan_ctx.__enter__()
        try:
            if l1_live_record:
                reset_sim_step()
            restore_sim(raw, sim_pre_tray)
            if l1_live_record and video_rec is not None:
                prime = getattr(raw, "_prime_rgb_array_renderer", None)
                if callable(prime):
                    prime()
                try:
                    video_rec.capture()
                except RuntimeError:
                    pass
            if l1_mode and not use_demo_grasp:
                dual = run_l1_dual_async(
                    raw,
                    left_agent=left_agent,
                    right_agent=right_agent,
                    canonical_tray=canonical_tray,
                    canonical_peg=canonical_peg,
                    detector=detector,
                    tpsr_cfg=tpsr_cfg,
                    lift_ref=lift_ref,
                    manifest_entry=manifest_entry,
                    contact_refine_iters=contact_refine_iters,
                    skip_peg_lift=skip_peg_lift,
                    live_record=l1_live_record,
                    topology_ok_fn=_topology_grasp_ok,
                )
                repair_iters += int(dual.repair_iters)
                tray_fin = dual.tray_fin
                peg_fin = dual.peg_fin
                peg_ik = dual.peg_ik
                tray_fc_ok = bool(dual.tray_fc_ok)
                peg_fc_ok = bool(dual.peg_fc_ok)
                tray_qp_err = float(dual.tray_qp_err)
                peg_qp_err = float(dual.peg_qp_err)
                tray_topo_ok = bool(dual.tray_topo_ok)
                tray_topo_msg = str(dual.tray_topo_msg)
                peg_topo_ok = bool(dual.peg_topo_ok)
                peg_topo_msg = str(dual.peg_topo_msg)
                tray_grasp_pose = dual.left_locked.copy()
                peg_grasp_pose = dual.right_hold.copy()
                left_hold = dual.left_hold.copy()
                right_hold = dual.right_hold.copy()
                left_locked = dual.left_locked.copy()
                tray_lifted = bool(dual.tray_lifted)
                peg_lifted = bool(dual.peg_lifted)
                tray_lift_m = float(dual.tray_lift_m)
                peg_lift_m = float(dual.peg_lift_m)
                tray_lift_hold_stable = bool(tray_lifted)
                hold_contact_min = int(dual.tray_lift.contact_min)
                tray_contact_pre_peg = int(side_contact_count(detector, raw, object_name="tray"))
                sim_pre_peg = snapshot_sim(raw)
                if l1_live_record:
                    _mark("tray_grasp_done")
                    if tray_lifted:
                        _mark("tray_lift_done")
                        _mark("tray_hold_done")
                    _mark("peg_grasp_done")
                    if peg_lifted:
                        _mark("peg_lift_done")
                tray_grasp_lift = verify_tray_grasp_lift(
                    raw,
                    detector=detector,
                    canonical_tray=canonical_tray,
                    lift_ref=lift_ref,
                    hold_contact_min=hold_contact_min,
                    cfg=LiftVerifyConfig(),
                    default_lift_height_m=tray_lift_height_m,
                    hold_stable=tray_lift_hold_stable,
                )
                peg_phase_ok = bool(
                    tray_lifted
                    and tray_lift_m >= MIN_TRAY_LIFT_M
                    and tray_contact_pre_peg >= max(2, MIN_GRASP_CONTACT_COUNT - 1)
                )
                right_agent.tpsr_cfg = tpsr_cfg
                sim_final = snapshot_sim(raw)
                plan_ok = bool(
                    tray_topo_ok
                    and peg_topo_ok
                    and tray_lifted
                    and tray_lift_m >= MIN_TRAY_LIFT_M
                    and peg_lifted
                    and peg_lift_m >= MIN_PEG_LIFT_M
                    and not skip_peg_lift
                )
                if l1_live_record and not tray_lifted:
                    print(
                        f"ABORT: tray not lifted dz={tray_lift_m:.3f}m < {MIN_TRAY_LIFT_M:.3f}m",
                        flush=True,
                    )
            elif use_demo_grasp:
                if l1_live_record:
                    lf = int(manifest_entry["timing"]["left_grasp_frame"])
                    replay_demo_segment(
                        raw,
                        manifest_entry,
                        start_frame=0,
                        end_frame=lf,
                    )
                    tray_grasp_pose = vec_to_arm_action(read_arm_action(raw, "left"))
                else:
                    tray_grasp_pose = apply_demo_grasp_frame(raw, manifest_entry, "tray")
                right_hold = right_home.copy()
                left_hold = tray_grasp_pose.copy()
                right_hold, left_hold, tray_fin = finalize_side_grasp(
                    raw,
                    side="left",
                    object_name="tray",
                    canonical=canonical_tray,
                    hold_right=right_hold,
                    hold_left=left_hold,
                    detector=detector,
                    tpsr_cfg=tpsr_cfg,
                    contact_refine_iters=contact_refine_iters,
                )
                repair_iters += int(tray_fin.tpsr_iters)
            else:
                right_hold, left_hold, tray_fin, n, _, _ = plan_side_grasp(
                    left_agent,
                    raw,
                    target23=left_target,
                    hold_right=right_hold,
                    hold_left=left_hold,
                    detector=detector,
                    ik=tray_ik,
                    tpsr_cfg=tpsr_cfg,
                    contact_refine_iters=contact_refine_iters,
                    live_record=l1_live_record,
                )
                repair_iters += n
                tray_grasp_pose = vec_to_arm_action(left_hold)
            if not (l1_mode and not use_demo_grasp):
                if l1_live_record:
                    _mark("tray_grasp_done")
                tray_fc_ok = bool(tray_fin.qp_ok)
                tray_qp_err = float(tray_fin.qp_max_error)
                tray_topo_ok, tray_topo_msg = _topology_grasp_ok(
                    raw, detector, canonical_tray, object_name="tray", tpsr_cfg=tpsr_cfg
                )
                if not tray_fc_ok:
                    tray_topo_msg = f"tray_fc:qp={tray_qp_err:.3f}"

                if _tray_ready_for_lift(
                    raw, detector, tray_topo_ok=tray_topo_ok, tray_fc_ok=tray_fc_ok, l1_mode=l1_mode
                ):
                    lift_cfg_use = replace(
                        lift_cfg,
                        squeeze_steps=lift_cfg.squeeze_steps + (16 if l1_mode else 0),
                    )
                    if l1_mode:
                        left_hold, _, _, _ = re_squeeze_fc(
                            raw,
                            side="left",
                            object_name="tray",
                            canonical=canonical_tray,
                            hold_right=right_home,
                            hold_left=left_hold,
                            tpsr_cfg=lift_cfg_use,
                            max_rounds=4,
                        )
                    tray_dz = (
                        demo_lift_world_dz(manifest_entry, lift_ref, "tray")
                        if l1_mode and lift_ref is not None
                        else tray_lift_height_m
                    )
                    left_hold = execute_tray_lift(
                        raw,
                        grasp_left=left_hold,
                        hold_right=right_home,
                        lift_ref=lift_ref if not l1_mode else None,
                        detector=detector,
                        lift_height_m=tray_dz,
                        steps=tray_lift_steps,
                        hold_steps=8 if l1_mode else (4 if fast else 8),
                        pre_lift_settle=max(lift_pre_settle, 12 if l1_mode else lift_pre_settle),
                        lift_exec_cap=record_lift_cap if l1_live_record else plan_lift_cap,
                        lock_passive_arm=l1_mode,
                        object_z_only=l1_mode,
                        canonical=canonical_tray,
                        tpsr_cfg=lift_cfg_use,
                    )
                    if l1_mode:
                        left_hold, _, _, _ = re_squeeze_fc(
                            raw,
                            side="left",
                            object_name="tray",
                            canonical=canonical_tray,
                            hold_right=right_home,
                            hold_left=left_hold,
                            tpsr_cfg=lift_cfg_use,
                            max_rounds=2,
                        )
                    tray_lift_m = _object_lift_m(raw, detector, "tray")
                    tray_lifted = tray_lift_m >= MIN_TRAY_LIFT_M
                    if not tray_lifted and l1_live_record:
                        print(
                            f"ABORT: tray not lifted dz={tray_lift_m:.3f}m < {MIN_TRAY_LIFT_M:.3f}m",
                            flush=True,
                        )
                    if tray_lifted:
                        left_hold, tray_lift_hold_stable, hold_contact_min = hold_tray_before_peg(
                        raw,
                        left_hold=left_hold,
                        right_home=right_home,
                        detector=detector,
                        lift_ref=lift_ref,
                        max_hold_steps=min(tray_hold_max_steps, 24) if l1_mode else tray_hold_max_steps,
                        warmup_steps=max(hold_warmup_steps, 8 if l1_mode else hold_warmup_steps),
                        )
                        if l1_live_record:
                            _mark("tray_lift_done")
                            _mark("tray_hold_done")
                    else:
                        tray_lift_hold_stable = False
                        hold_contact_min = int(side_contact_count(detector, raw, object_name="tray"))

                tray_grasp_lift = verify_tray_grasp_lift(
                    raw,
                    detector=detector,
                    canonical_tray=canonical_tray,
                    lift_ref=lift_ref,
                    hold_contact_min=hold_contact_min,
                    cfg=LiftVerifyConfig(),
                    default_lift_height_m=tray_lift_height_m,
                    hold_stable=tray_lift_hold_stable,
                )

                left_locked = vec_to_arm_action(left_hold)
                right_hold = right_home.copy()
                if l1_mode and tray_lifted:
                    settle_bimanual_actions(raw, right23=right_hold, left23=left_locked, n_substeps=8)
                    left_locked, _, _, _ = re_squeeze_fc(
                        raw,
                        side="left",
                        object_name="tray",
                        canonical=canonical_tray,
                        hold_right=right_hold,
                        hold_left=left_locked,
                        tpsr_cfg=_record_tpsr_cfg(tpsr_cfg),
                        max_rounds=2,
                    )
                    left_locked = vec_to_arm_action(read_arm_action(raw, "left"))
                    enforce_locked_passive(
                        raw, locked_left=left_locked, locked_right=right_hold, n_substeps=10
                    )
                else:
                    settle_bimanual_actions(raw, right23=right_hold, left23=left_locked, n_substeps=4)
                    right_hold, left_locked, _ = finalize_side_grasp(
                        raw,
                        side="left",
                        object_name="tray",
                        canonical=canonical_tray,
                        hold_right=right_hold,
                        hold_left=left_locked,
                        detector=detector,
                        tpsr_cfg=tpsr_cfg,
                        contact_refine_iters=contact_refine_iters,
                    )
                tray_contact_pre_peg = int(side_contact_count(detector, raw, object_name="tray"))
                sim_pre_peg = snapshot_sim(raw)

                peg_tpsr_cfg = replace(
                    tpsr_cfg,
                    squeeze_steps=tpsr_cfg.squeeze_steps + 20,
                    require_qp_fc=True,
                )
                right_agent.tpsr_cfg = peg_tpsr_cfg
                peg_phase_ok = _peg_phase_ready(
                    l1_mode=l1_mode,
                    tray_topo_ok=tray_topo_ok,
                    tray_fc_ok=tray_fc_ok,
                    tray_lifted=tray_lifted,
                    tray_grasp_lift=tray_grasp_lift,
                    tray_lift_m=tray_lift_m,
                    tray_contact_pre_peg=tray_contact_pre_peg,
                    tray_lift_hold_stable=tray_lift_hold_stable,
                )

                if peg_phase_ok:
                    restore_sim(raw, sim_pre_peg)
                    left_frozen = left_locked.copy()
                    if use_demo_grasp and not l1_mode:
                        rg = int(manifest_entry["timing"]["right_grasp_frame"])
                        peg_start = max(
                            0,
                            rg
                            - max(
                                right_agent.side_cfg.approach_pre_steps
                                + right_agent.side_cfg.approach_grasp_steps,
                                60,
                            ),
                        )
                        replay_demo_segment(
                            raw,
                            manifest_entry,
                            start_frame=peg_start,
                            end_frame=rg,
                            lock_left=left_locked,
                            active_side="right",
                        )
                        right_hold = vec_to_arm_action(read_arm_action(raw, "right"))
                        enforce_locked_passive(
                            raw, locked_left=left_locked, locked_right=right_hold, n_substeps=6
                        )
                        peg_cc = int(side_contact_count(detector, raw, object_name="peg"))
                        if peg_cc >= MIN_GRASP_CONTACT_COUNT:
                            right_hold, _, peg_fc_ok, peg_qp_err = re_squeeze_fc(
                                raw,
                                side="right",
                                object_name="peg",
                                canonical=canonical_peg,
                                hold_right=right_hold,
                                hold_left=left_locked,
                                tpsr_cfg=replace(peg_tpsr_cfg, require_qp_fc=False),
                                max_rounds=2,
                            )
                            peg_fin = GraspFinalizeReport(
                                contact_count=peg_cc,
                                qp_ok=bool(peg_fc_ok),
                                qp_max_error=float(peg_qp_err),
                                tpsr_iters=0,
                                contact_refine=None,
                            )
                        else:
                            right_hold, _, peg_fin = finalize_side_grasp(
                                raw,
                                side="right",
                                object_name="peg",
                                canonical=canonical_peg,
                                hold_right=right_hold,
                                hold_left=left_locked,
                                detector=detector,
                                tpsr_cfg=replace(peg_tpsr_cfg, require_qp_fc=False),
                                contact_refine_iters=min(contact_refine_iters, 6),
                            )
                        repair_iters += int(peg_fin.tpsr_iters)
                        enforce_locked_passive(
                            raw, locked_left=left_locked, locked_right=right_hold, n_substeps=6
                        )
                    else:
                        if l1_mode:
                            enforce_locked_passive(
                                raw, locked_left=left_frozen, locked_right=right_home, n_substeps=12
                            )
                            left_frozen, _, _, _ = re_squeeze_fc(
                                raw,
                                side="left",
                                object_name="tray",
                                canonical=canonical_tray,
                                hold_right=right_home,
                                hold_left=left_frozen,
                                tpsr_cfg=replace(peg_tpsr_cfg, require_qp_fc=False),
                                max_rounds=2,
                            )
                            left_frozen = vec_to_arm_action(read_arm_action(raw, "left"))
                        right_target, peg_ik = right_agent.plan(
                            raw,
                            hold_right=right_home,
                            hold_left=left_frozen if l1_mode else left_locked,
                            restore_env=True,
                            detector=detector,
                            side_cfg=right_agent.side_cfg,
                        )
                        peg_ik_ready = _ik_grasp_ready(peg_ik)
                        peg_direct: int | None = None
                        if not peg_ik_ready:
                            if peg_ik.contact_count > 0:
                                right_target = _ik_action_target(peg_ik, "right")
                                peg_direct = 70
                            else:
                                right_target = _canonical_arm_target(raw, canonical_peg, "peg")
                                peg_direct = 100
                        right_hold, _, peg_fin, n, _, _ = plan_side_grasp(
                            right_agent,
                            raw,
                            target23=right_target,
                            hold_right=right_home,
                            hold_left=left_frozen if l1_mode else left_locked,
                            detector=detector,
                            ik=peg_ik,
                            tpsr_cfg=peg_tpsr_cfg,
                            contact_refine_iters=contact_refine_iters,
                            direct_reach_steps=peg_direct,
                            live_record=l1_live_record,
                        )
                        repair_iters += n
                        if l1_mode:
                            left_frozen, _, _, _ = re_squeeze_fc(
                                raw,
                                side="left",
                                object_name="tray",
                                canonical=canonical_tray,
                                hold_right=right_home,
                                hold_left=left_frozen,
                                tpsr_cfg=replace(peg_tpsr_cfg, require_qp_fc=False),
                                max_rounds=1,
                            )
                            left_frozen = vec_to_arm_action(read_arm_action(raw, "left"))
                        enforce_locked_passive(
                            raw,
                            locked_left=left_frozen if l1_mode else left_locked,
                            locked_right=right_hold,
                            n_substeps=8,
                        )

                    left_hold = (left_frozen if l1_mode else left_locked).copy()
                    left_locked = left_frozen.copy() if l1_mode else left_locked
                    enforce_locked_passive(raw, locked_left=left_locked, locked_right=right_hold)
                    peg_fc_ok = bool(peg_fin.qp_ok)
                    peg_qp_err = float(peg_fin.qp_max_error)
                    peg_grasp_pose = vec_to_arm_action(right_hold)
                    peg_topo_ok, peg_topo_msg = _topology_grasp_ok(
                        raw, detector, canonical_peg, object_name="peg", tpsr_cfg=peg_tpsr_cfg
                    )
                    if l1_live_record:
                        _mark("peg_grasp_done")

                    if not skip_peg_lift and (peg_topo_ok or l1_mode):
                        peg_lift_cfg = replace(
                            peg_tpsr_cfg,
                            squeeze_steps=peg_tpsr_cfg.squeeze_steps + 16,
                            require_qp_fc=not l1_mode,
                        )
                        if not peg_fc_ok:
                            for _ in range(6 if l1_mode else 8):
                                right_hold, _, peg_fc_ok, peg_qp_err = re_squeeze_fc(
                                    raw,
                                    side="right",
                                    object_name="peg",
                                    canonical=canonical_peg,
                                    hold_right=right_hold,
                                    hold_left=left_locked,
                                    tpsr_cfg=peg_lift_cfg,
                                    max_rounds=1,
                                )
                                if l1_mode:
                                    left_locked, _, _, _ = re_squeeze_fc(
                                        raw,
                                        side="left",
                                        object_name="tray",
                                        canonical=canonical_tray,
                                        hold_right=right_home,
                                        hold_left=left_locked,
                                        tpsr_cfg=replace(peg_lift_cfg, require_qp_fc=False),
                                        max_rounds=1,
                                    )
                                if peg_fc_ok:
                                    break
                        left_hold = left_locked.copy()
                        enforce_locked_passive(
                            raw, locked_left=left_locked, locked_right=right_hold, n_substeps=8
                        )
                        right_hold = execute_peg_lift(
                            raw,
                            grasp_right=right_hold,
                            hold_left=left_locked,
                            lift_ref=lift_ref,
                            detector=detector,
                            steps=peg_lift_steps,
                            pre_lift_settle=max(lift_pre_settle, 12),
                            lift_exec_cap=record_lift_cap if l1_live_record else plan_lift_cap,
                            object_z_only=l1_mode,
                            canonical=canonical_peg,
                            tpsr_cfg=peg_lift_cfg,
                        )
                        peg_lifted = True
                        peg_lift_m = _object_lift_m(raw, detector, "peg")
                        left_hold = left_locked.copy()
                        enforce_locked_passive(
                            raw, locked_left=left_locked, locked_right=right_hold, n_substeps=4
                        )
                        if l1_live_record:
                            _mark("peg_lift_done")
                right_agent.tpsr_cfg = tpsr_cfg

                sim_final = snapshot_sim(raw)
                plan_ok = bool(
                    peg_phase_ok
                    and peg_topo_ok
                    and peg_lifted
                    and peg_lift_m >= MIN_PEG_LIFT_M
                    and not skip_peg_lift
                )
        finally:
            _plan_ctx.__exit__(None, None, None)

        # --- Record: L0 privileged zarr only (L1 records during plan above) ---
        record_ok = bool(
            plan_ok
            or (allow_partial_video and tray_lifted and tray_lift_m >= MIN_TRAY_LIFT_M)
        )
        if l1_live_record:
            deliver_video = bool(
                plan_ok
                and video_rec is not None
                and video_rec.frame_count > 0
                and tray_lift_m >= MIN_TRAY_LIFT_M
            )
            video_partial = False
        elif use_privileged_demo and video_out is not None and record_ok:
            reset_sim_step()
            with exec_recording(on_frame):
                restore_sim(raw, sim_pre_tray)
                if video_rec is not None:
                    prime = getattr(raw, "_prime_rgb_array_renderer", None)
                    if callable(prime):
                        prime()
                    try:
                        video_rec.capture()
                    except RuntimeError:
                        pass
                phase_frames, video_end = demo_video_phase_frames(manifest_entry, lift_ref)
                replay_demo_privileged_video(
                    raw,
                    manifest_entry,
                    start_frame=0,
                    end_frame=video_end,
                    phase_frames=phase_frames,
                    mark_fn=_mark,
                )
            deliver_video = bool(plan_ok)
            video_partial = bool(allow_partial_video and record_ok and not plan_ok)
            if video_partial and video_rec is not None and video_rec.frame_count > 0:
                deliver_video = True

        restore_sim(raw, sim_final)

        tray_min = max(2, MIN_GRASP_CONTACT_COUNT - 1)
        peg_contact_pre_lift = int(side_contact_count(detector, raw, object_name="peg"))
        bench_peg_pre = verify_side_hold(
            raw,
            object_name="peg",
            action_right=right_hold,
            action_left=left_locked,
            detector=detector,
            bench_cfg=bench_cfg,
            tpsr_cfg=tpsr_cfg,
        )
        pre_insert_ok = bool(
            tray_lift_hold_stable is not False
            and (hold_contact_min >= tray_min or tray_contact_pre_peg >= tray_min)
            and peg_contact_pre_lift >= MIN_GRASP_CONTACT_COUNT
            and bench_peg_pre.stable
        )

        left_hold = left_locked.copy()
        enforce_locked_passive(raw, locked_left=left_hold, locked_right=right_hold, n_substeps=12)
        repair = verify_grasp_hold(
            raw,
            detector,
            action_right=right_hold,
            action_left=left_hold,
            hold_steps=hold_steps,
            warmup_steps=hold_warmup_steps,
            tray_lifted=True,
            adjust_left=False,
            adjust_right=False,
            require_tray=True,
            require_peg=True,
        )
        repair.repair_iters = repair_iters

        bench_tray = verify_side_hold(
            raw,
            object_name="tray",
            action_right=right_hold,
            action_left=left_hold,
            detector=detector,
            bench_cfg=bench_cfg,
            tpsr_cfg=tpsr_cfg,
        )
        bench_peg = verify_side_hold(
            raw,
            object_name="peg",
            action_right=right_hold,
            action_left=left_hold,
            detector=detector,
            bench_cfg=bench_cfg,
            tpsr_cfg=tpsr_cfg,
        )

        tray_contact = int(side_contact_count(detector, raw, object_name="tray"))
        peg_contact = int(side_contact_count(detector, raw, object_name="peg"))
        tray_hold_ok = tray_lift_hold_stable is not False
        tray_contact_ok = (
            tray_contact >= tray_min
            and bench_tray is not None
            and bench_tray.stable
        )
        bimanual_ok = bool(
            tray_hold_ok
            and tray_contact_ok
            and peg_contact >= MIN_GRASP_CONTACT_COUNT
            and bench_peg is not None
            and bench_peg.stable
        )
        lift_ok = bool(tray_grasp_lift.success and tray_lift_hold_stable is not False) if skip_insert else tray_hold_ok

        insert_report: InsertReport | None = None
        insert_ok = False
        if not skip_insert and pre_insert_ok:
            insert_report = run_hybrid_insert(
                env,
                raw,
                max_steps=2500 if not fast else 800,
                tray_rest_z=float(detector._tray_rest_z),  # noqa: SLF001
                peg_rest_z=float(detector._peg_rest_z),  # noqa: SLF001
            )
            insert_ok = bool(insert_report.success)

        if skip_insert:
            success = bool(bimanual_ok and plan_ok)
            fail_reason = "ok" if success else _bimanual_fail_reason(
                tray_contact=tray_contact,
                peg_contact=peg_contact,
                tray_min=tray_min,
                hold_contact_min=hold_contact_min,
                bench_tray=bench_tray,
                bench_peg=bench_peg,
                tray_lift_hold_stable=tray_lift_hold_stable,
                skip_tray_lift=False,
            )
            if not tray_topo_ok or not tray_fc_ok:
                fail_reason = f"tray_topo:{tray_topo_msg}"
                success = False
            elif not tray_lifted or not tray_grasp_lift.lift_height_ok or tray_lift_m < 0.030:
                fail_reason = "tray_not_lifted"
                success = False
            elif skip_peg_lift:
                fail_reason = "peg_lift_skipped"
                success = False
            elif not peg_phase_ok:
                fail_reason = "peg_skipped_tray_not_ready"
                success = False
            elif not peg_topo_ok:
                fail_reason = f"peg_topo:{peg_topo_msg}"
                success = False
            elif not peg_lifted or peg_lift_m < MIN_PEG_LIFT_M:
                fail_reason = "peg_not_lifted"
                success = False
            elif not success:
                fail_reason = "bimanual_lift_fail"
        else:
            success = bool(insert_ok)
            if not pre_insert_ok:
                fail_reason = _bimanual_fail_reason(
                    tray_contact=tray_contact,
                    peg_contact=peg_contact_pre_lift,
                    tray_min=tray_min,
                    hold_contact_min=hold_contact_min,
                    bench_tray=bench_tray,
                    bench_peg=bench_peg_pre,
                    tray_lift_hold_stable=tray_lift_hold_stable,
                    skip_tray_lift=False,
                )
            elif insert_report is not None:
                fail_reason = insert_report.fail_reason if insert_ok else insert_report.fail_reason
            else:
                fail_reason = "insert_skipped_precondition"

        report_success = bool(success)
        return SkillReplayReport(
            seed=int(seed),
            demo_episode_index=int(demo_skill.episode_index),
            retrieval_distance_m=float(retrieval_dist),
            tray_ik=tray_ik,
            peg_ik=peg_ik,
            repair=repair,
            tray_laplacian_rmse_m=laplacian_rmse(raw, canonical_tray, object_name="tray"),
            peg_laplacian_rmse_m=laplacian_rmse(raw, canonical_peg, object_name="peg"),
            tray_lift_hold_stable=tray_lift_hold_stable,
            bench_tray=bench_tray,
            bench_peg=bench_peg,
            insert=insert_report,
            success=success,
            fail_reason=fail_reason if not success else "ok",
            stage="lift_only" if skip_insert else "full",
            extra={
                "tray_grasp_lift_ok": tray_grasp_lift.success,
                "hold_contact_min": hold_contact_min,
                "tray_fc_ok": bool(tray_fc_ok),
                "tray_qp_err": float(tray_qp_err),
                "peg_fc_ok": bool(peg_fc_ok),
                "peg_qp_err": float(peg_qp_err),
                "tray_lift_track": getattr(raw, "_last_tray_lift_meta", {}),
                "peg_lift_track": getattr(raw, "_last_peg_lift_meta", {}),
                "trajectory_audit": audit,
                "tray_topo_ok": bool(tray_topo_ok),
                "tray_topo_msg": str(tray_topo_msg),
                "tray_lifted": bool(tray_lifted),
                "peg_phase_ok": bool(peg_phase_ok),
                "peg_lifted": bool(peg_lifted),
                "peg_lift_m": float(peg_lift_m),
                "tray_lift_m": float(tray_lift_m),
                "plan_ok": bool(plan_ok),
                "deliver_video": bool(deliver_video),
            },
        )
    finally:
        if video_rec is not None:
            audit.video_frames = video_rec.frame_count
            audit.sim_steps = max(audit.sim_steps, sim_step_count())
            table_z = float(detector._tray_rest_z) if detector else 0.0  # noqa: SLF001
            peg_rest = float(detector._peg_rest_z) if detector else 0.0  # noqa: SLF001
            validate_delivery(
                audit,
                table_z=table_z,
                peg_rest_z=peg_rest,
                require_both_lifts=not video_partial,
            )
            if video_partial:
                audit.errors = [
                    e
                    for e in audit.errors
                    if not (
                        e.startswith("peg_grasp_no_contact")
                        or e.startswith("missing_phase:")
                        or e.startswith("sim_steps_too_many")
                    )
                ]
            print(audit.summary(), flush=True)
            save_video = bool(
                deliver_video
                and audit.video_frames > 0
                and (audit.ok_for_delivery() or video_partial)
            )
            if save_video:
                try:
                    path = video_rec.close()
                    print(f"video={path} duration={video_rec.duration_s:.1f}s", flush=True)
                except Exception as exc:
                    print(f"video_save_failed: {exc}", flush=True)
            else:
                print("video_skipped: trajectory audit failed", flush=True)
                if video_rec.out_path.exists():
                    video_rec.out_path.unlink(missing_ok=True)
        env.close()
