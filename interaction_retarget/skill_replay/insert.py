"""Insert phase: PoseInsert (demo_replay / policy handoff) or legacy hybrid geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol

import numpy as np
from scipy.spatial.transform import Rotation as R

from hybrid_insert.assembly_contacts import AssemblyContactLabeler
from hybrid_insert.config import HybridInsertConfig
from hybrid_insert.geometry import (
    hole_opening_axis,
    in_approach_cylinder,
    peg_insert_end_pos,
    tip_socket_distance,
    toward_socket_delta,
)
from hybrid_insert.integration import EvalHybridInsert

from interaction_retarget.constants import PEG_BODY
from interaction_retarget.grasp.approach import interpolate_arm_only
from interaction_retarget.grasp.repair import _step_side
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.replay import (
    policy_dual_arm_to_raw,
    raw_flat_to_dict,
    rotvec_dual_arm_to_policy,
    zarr_action_to_policy46,
)
from interaction_retarget.sim.settle import read_arm_action, vec_to_arm_action

InsertBackend = Literal["hybrid", "poseinsert"]
PoseInsertReachMode = Literal["demo_replay", "policy"]


class _InsertRunner(Protocol):
    controller: object | None

    def on_reset(self, gym_env) -> None: ...

    def observe(self, gym_env, policy_action44: np.ndarray) -> None: ...

    def merge(self, gym_env, policy_action44: np.ndarray) -> np.ndarray: ...

    @property
    def active(self) -> bool: ...


_SOCKET_SITE = "industreal_tray_insert_round_peg_8mm_socket_site"
_BOTTOM_GEOM = "industreal_tray_insert_round_peg_8mm_bottom_contact"


def dual_arm23_to_action44(left23: np.ndarray, right23: np.ndarray) -> np.ndarray:
    left23 = np.asarray(left23, dtype=np.float64).reshape(23)
    right23 = np.asarray(right23, dtype=np.float64).reshape(23)
    r_rot = R.from_quat(right23[3:7], scalar_first=True).as_rotvec()
    l_rot = R.from_quat(left23[3:7], scalar_first=True).as_rotvec()
    return np.concatenate(
        [right23[0:3], r_rot, right23[7:23], left23[0:3], l_rot, left23[7:23]],
        dtype=np.float64,
    )


def action44_to_raw_dict(action44: np.ndarray) -> dict[str, np.ndarray]:
    action46 = rotvec_dual_arm_to_policy(np.asarray(action44, dtype=np.float64).reshape(44))
    return policy_dual_arm_to_raw(action46)


@dataclass
class InsertReport:
    success: bool
    steps: int
    insert_ok: bool
    handoff: bool
    phase: str
    peg_lift_m: float
    fail_reason: str = ""


def _env_step_info(env, action, video_cb: Callable[[dict], None] | None = None) -> dict:
    out = env.step(action)
    obs = out[0]
    info = out[4] if len(out) == 5 else out[3]
    if video_cb is not None:
        video_cb(obs)
    return info


def _peg_lift_m(raw_env, peg_rest_z: float) -> float:
    peg_id = int(raw_env._model.body(PEG_BODY).id)
    return float(raw_env._data.xpos[peg_id, 2]) - float(peg_rest_z)


def _insert_geometry(raw_env) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    data = raw_env._data
    model = raw_env._model
    peg_id = int(model.body(PEG_BODY).id)
    socket_id = int(model.site(_SOCKET_SITE).id)
    bottom_id = int(model.geom(_BOTTOM_GEOM).id)
    tip = peg_insert_end_pos(data.xpos[peg_id], data.xmat[peg_id])
    socket = np.asarray(data.site_xpos[socket_id], dtype=np.float64)
    hole_axis = hole_opening_axis(
        socket,
        data.site_xmat[socket_id],
        np.asarray(data.geom_xpos[bottom_id], dtype=np.float64),
    )
    dist = tip_socket_distance(tip, socket)
    return tip, socket, hole_axis, dist


def _log_insert_trace(
    raw_env,
    *,
    step: int,
    phase: str,
    wrist_xyz: np.ndarray | None = None,
) -> None:
    tip, socket, _, dist_mm = _insert_geometry(raw_env)
    dist_mm *= 1000.0
    wrist = np.asarray(wrist_xyz, dtype=np.float64).reshape(3) if wrist_xyz is not None else None
    wrist_s = (
        f" wrist=[{wrist[0]:.3f},{wrist[1]:.3f},{wrist[2]:.3f}]"
        if wrist is not None
        else ""
    )
    print(
        f"insert trace step={step} phase={phase} "
        f"peg_tip=[{tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}] "
        f"socket=[{socket[0]:.3f},{socket[1]:.3f},{socket[2]:.3f}] "
        f"tip_dist_mm={dist_mm:.1f}{wrist_s}",
        flush=True,
    )


def _insert_peg_lift(
    raw_env,
    left: np.ndarray,
    right: np.ndarray,
    *,
    target_dz: float = 0.07,
    steps: int = 72,
) -> np.ndarray:
    """Vertical peg lift via mocap ramp (left frozen)."""
    left = vec_to_arm_action(left)
    right = vec_to_arm_action(right)
    hand = right[7:23].copy()
    z0 = float(right[2])
    z1 = z0 + float(target_dz)
    goal = right.copy()
    goal[2] = z1
    for i in range(max(int(steps), 1)):
        u = (i + 1) / max(int(steps), 1)
        active = interpolate_arm_only(right, goal, u, hand=hand)
        _step_side(raw_env, side="right", active23=active, hold_right=active, hold_left=left)
    return vec_to_arm_action(read_arm_action(raw_env, "right"))


def _try_force_handoff(
    env,
    raw_env,
    insert_runner: _InsertRunner,
    labeler: AssemblyContactLabeler,
    *,
    cfg: HybridInsertConfig,
    peg_rest_z: float,
) -> bool:
    if insert_runner.controller is None or insert_runner.active:
        return insert_runner.controller is not None and insert_runner.active
    outcome = labeler.compute(raw_env)
    tip, socket, hole_axis, dist = _insert_geometry(raw_env)
    peg_dz = _peg_lift_m(raw_env, peg_rest_z)
    in_cyl = in_approach_cylinder(
        tip,
        socket,
        hole_axis,
        xy_tol_m=cfg.approach_xy_m,
        z_min_m=cfg.approach_z_min_m,
        z_max_m=cfg.approach_z_max_m,
    )
    ready = (
        outcome.tray_contact_count >= 2
        and outcome.peg_contact_count >= 3
        and peg_dz >= cfg.lift_ready_m - 0.015
        and (in_cyl or dist < 0.12)
    )
    if not ready:
        return False
    left = read_arm_action(raw_env, "left")
    right = read_arm_action(raw_env, "right")
    policy44 = dual_arm23_to_action44(left, right)
    insert_runner.controller._handoff_streak = int(cfg.handoff_confirm_frames)  # noqa: SLF001
    insert_runner.observe(env, policy44)
    return bool(insert_runner.active)


def _privileged_approach_to_handoff(
    env,
    raw_env,
    insert_runner: _InsertRunner,
    labeler: AssemblyContactLabeler,
    *,
    cfg: HybridInsertConfig,
    max_steps: int = 900,
) -> None:
    """Move peg toward socket + lift (skill replay has no learned policy motion)."""
    peg_rest_z = float(labeler._peg_rest_z)  # noqa: SLF001

    for step in range(int(max_steps)):
        outcome = labeler.compute(raw_env)
        left = vec_to_arm_action(read_arm_action(raw_env, "left"))
        right = vec_to_arm_action(read_arm_action(raw_env, "right"))
        tip, socket, hole_axis, dist = _insert_geometry(raw_env)
        peg_dz = _peg_lift_m(raw_env, peg_rest_z)
        in_cyl = in_approach_cylinder(
            tip,
            socket,
            hole_axis,
            xy_tol_m=cfg.approach_xy_m,
            z_min_m=cfg.approach_z_min_m,
            z_max_m=cfg.approach_z_max_m,
        )

        policy44 = dual_arm23_to_action44(left, right)
        insert_runner.observe(env, policy44)
        if insert_runner.controller is not None and insert_runner.active:
            return

        delta = toward_socket_delta(tip, socket, gain=0.55, max_step_m=0.005)
        if peg_dz < cfg.lift_ready_m:
            delta[2] += min(0.004, cfg.lift_ready_m - peg_dz)
        norm = float(np.linalg.norm(delta))
        if norm > 0.006:
            delta = delta * (0.006 / norm)
        right[0:3] = right[0:3] + delta

        policy44 = dual_arm23_to_action44(left, right)
        insert_runner.observe(env, policy44)
        if insert_runner.controller is not None and insert_runner.active:
            return
        action46 = rotvec_dual_arm_to_policy(np.asarray(policy44, dtype=np.float64).reshape(44))
        _env_step_info(env, action46.astype(np.float32))
        insert_runner.observe(env, policy44)
        if insert_runner.controller is not None and insert_runner.active:
            return

        if step % 80 == 0:
            print(
                f"skill_insert: approach step={step} peg_dz={peg_dz:.3f} "
                f"dist={dist*1000:.1f}mm in_cyl={in_cyl} "
                f"tray_ok={outcome.tray_ok} peg_ok={outcome.peg_ok}",
                flush=True,
            )

    _try_force_handoff(env, raw_env, insert_runner, labeler, cfg=cfg, peg_rest_z=peg_rest_z)


def demo_replay_to_pre_insert(
    env,
    raw_env,
    *,
    zarr_path: Path | str,
    stop_frame: int,
    initial_state: dict | None = None,
    video_cb: Callable[[dict], None] | None = None,
    labeler: AssemblyContactLabeler | None = None,
) -> int:
    """Open-loop zarr replay through peg lift end (before insert)."""
    from dexjoco.tasks import CONFIG_MAPPING
    from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

    actions, _, zarr_init = load_zarr_episode(Path(zarr_path))
    init = initial_state if initial_state is not None else zarr_init
    env.reset()
    if init is not None and has_restorer("bimanual_assembly"):
        restore_initial_state(env, "bimanual_assembly", CONFIG_MAPPING["bimanual_assembly"](), init)

    if labeler is not None:
        labeler.reset_reference(raw_env)

    if video_cb is not None:
        obs = env.observation(raw_env._compute_observation())
        video_cb(obs)

    limit = min(len(actions), int(stop_frame) + 1)
    for frame_idx in range(limit):
        if video_cb is not None:
            action46 = zarr_action_to_policy46(actions[frame_idx])
            obs, _, _, _, _ = env.step(action46.astype(np.float32))
            video_cb(obs)
        else:
            raw_env.step(raw_flat_to_dict(actions[frame_idx]))
    return limit - 1


def make_insert_runner(
    backend: InsertBackend,
    *,
    hybrid_cfg: HybridInsertConfig | None = None,
    poseinsert_ckpt: Path | str | None = None,
    poseinsert_data_root: Path | str | None = None,
    poseinsert_config: "PoseInsertAdapterConfig | None" = None,
    bimanual: bool = True,
    insert_mode: str = "auto",
    zarr_path: Path | str | None = None,
    insert_start_frame: int | None = None,
    insert_end_frame: int | None = None,
) -> _InsertRunner:
    if backend == "hybrid":
        return EvalHybridInsert(
            task="bimanual_assembly",
            enabled=True,
            config=hybrid_cfg,
        )
    from pose_insert.config import PoseInsertAdapterConfig
    from pose_insert.integration import make_eval_insert_runner

    cfg = poseinsert_config or PoseInsertAdapterConfig(
        freeze_left_arm_at_handoff=not bimanual,
    )
    return make_eval_insert_runner(
        task="bimanual_assembly",
        ckpt_path=Path(poseinsert_ckpt) if poseinsert_ckpt is not None else None,
        data_root=Path(poseinsert_data_root) if poseinsert_data_root is not None else None,
        config=cfg,
        insert_mode=insert_mode,
        zarr_path=zarr_path,
        insert_start_frame=insert_start_frame,
        insert_end_frame=insert_end_frame,
    )


def run_pose_insert_phase(
    env,
    raw_env,
    *,
    reach_mode: PoseInsertReachMode = "demo_replay",
    max_steps: int = 900,
    insert_cfg: HybridInsertConfig | None = None,
    poseinsert_ckpt: Path | str | None = None,
    poseinsert_data_root: Path | str | None = None,
    manifest_entry: dict | None = None,
    sidecar_dir: Path | str | None = None,
    tray_rest_z: float | None = None,
    peg_rest_z: float | None = None,
    peg_lift_end_frame: int | None = None,
    video_cb: Callable[[dict], None] | None = None,
    debug: bool = False,
    bimanual: bool = True,
    insert_mode: str = "auto",
    insert_end_frame: int | None = None,
) -> InsertReport:
    """Reach pre-insert (lift done), then PoseInsert insert.

    demo_replay: zarr -> peg_lift_end -> PoseInsert
    policy: caller/VLA already lifted; observe until pre_insert -> PoseInsert
    """
    if reach_mode == "demo_replay" and manifest_entry is None:
        raise ValueError("demo_replay requires manifest_entry")

    cfg = insert_cfg or HybridInsertConfig(handoff_confirm_frames=3, approach_xy_m=0.10)
    labeler = AssemblyContactLabeler(raw_env)
    if tray_rest_z is not None and peg_rest_z is not None:
        labeler._tray_rest_z = float(tray_rest_z)  # noqa: SLF001
        labeler._peg_rest_z = float(peg_rest_z)  # noqa: SLF001

    pre_insert_ok = False

    if reach_mode == "demo_replay":
        assert manifest_entry is not None
        from pose_insert.pre_insert import resolve_peg_lift_end_frame

        if sidecar_dir is None and peg_lift_end_frame is None:
            raise ValueError("demo_replay requires sidecar_dir or peg_lift_end_frame")
        stop_frame = (
            int(peg_lift_end_frame)
            if peg_lift_end_frame is not None
            else resolve_peg_lift_end_frame(manifest_entry, sidecar_dir)
        )
        _, _, initial_state = load_zarr_episode(Path(manifest_entry["zarr_path"]))
        demo_replay_to_pre_insert(
            env,
            raw_env,
            zarr_path=manifest_entry["zarr_path"],
            stop_frame=stop_frame,
            initial_state=initial_state,
            video_cb=video_cb,
            labeler=labeler,
        )
        peg_rest_z = float(labeler._peg_rest_z)  # noqa: SLF001

    from pose_insert.config import PoseInsertAdapterConfig

    zarr_oracle = insert_mode == "zarr_oracle"
    insert_end = insert_end_frame
    if zarr_oracle:
        assert manifest_entry is not None
        if peg_lift_end_frame is None:
            from pose_insert.pre_insert import resolve_peg_lift_end_frame

            peg_lift_end_frame = resolve_peg_lift_end_frame(manifest_entry, sidecar_dir)
        if insert_end is None:
            meta_path = Path(poseinsert_data_root or "") / "train" / str(manifest_entry["episode_index"]) / "meta.json"
            if meta_path.is_file():
                import json

                seg = json.loads(meta_path.read_text(encoding="utf-8")).get("segment", {})
                if seg.get("end_frame") is not None:
                    insert_end = int(seg["end_frame"])
    elif poseinsert_ckpt is None:
        raise ValueError("policy insert requires poseinsert_ckpt")

    policy_insert = insert_mode not in ("zarr_oracle",)
    insert_runner = make_insert_runner(
        "poseinsert",
        hybrid_cfg=cfg,
        poseinsert_ckpt=poseinsert_ckpt,
        poseinsert_data_root=poseinsert_data_root,
        poseinsert_config=PoseInsertAdapterConfig(
            freeze_left_arm_at_handoff=not bimanual and not zarr_oracle,
            left_insert_coop_gain=0.0,
            handoff_settle_frames=10 if policy_insert else 30,
            peg_lost_abort_frames=60 if policy_insert else 15,
            action_blend=0.35 if policy_insert else 0.45,
        ),
        bimanual=bimanual,
        insert_mode=insert_mode,
        zarr_path=manifest_entry["zarr_path"] if zarr_oracle and manifest_entry else None,
        insert_start_frame=int(peg_lift_end_frame) if zarr_oracle and peg_lift_end_frame is not None else None,
        insert_end_frame=insert_end,
    )
    insert_runner.on_reset(
        env,
        peg_rest_z=peg_rest_z if reach_mode == "demo_replay" else None,
        tray_rest_z=float(labeler._tray_rest_z) if reach_mode == "demo_replay" else None,  # noqa: SLF001
    )

    if reach_mode == "demo_replay":
        left = read_arm_action(raw_env, "left")
        right = read_arm_action(raw_env, "right")
        policy44 = dual_arm23_to_action44(left, right)
        if insert_runner.controller is not None:
            pre_insert_ok = insert_runner.controller.begin_pose_insert(raw_env, policy44)
    else:
        left = read_arm_action(raw_env, "left")
        right = read_arm_action(raw_env, "right")
        policy44 = dual_arm23_to_action44(left, right)
        if insert_runner.controller is not None:
            pre_insert_ok = insert_runner.controller.begin_pose_insert(raw_env, policy44)

    success = False
    insert_ok = False
    step = 0
    if pre_insert_ok:
        ctrl = insert_runner.controller
        if debug and ctrl is not None:
            right = read_arm_action(raw_env, "right")
            _log_insert_trace(
                raw_env,
                step=0,
                phase=ctrl.phase_name,
                wrist_xyz=right[0:3],
            )
        for step in range(int(max_steps)):
            left = read_arm_action(raw_env, "left")
            right = read_arm_action(raw_env, "right")
            policy44 = dual_arm23_to_action44(left, right)
            merged = insert_runner.merge(env, policy44)
            action46 = rotvec_dual_arm_to_policy(np.asarray(merged, dtype=np.float64).reshape(44))
            info = _env_step_info(env, action46.astype(np.float32), video_cb=video_cb)
            outcome = labeler.compute(raw_env)
            insert_ok = bool(outcome.insert_ok)
            if debug and ctrl is not None and (step == 0 or (step + 1) % 30 == 0):
                _log_insert_trace(
                    raw_env,
                    step=step + 1,
                    phase=ctrl.phase_name,
                    wrist_xyz=merged[0:3],
                )
            if info.get("succeed"):
                success = True
                break
            if ctrl is not None and ctrl.phase_name == "RELEASE":
                if insert_ok:
                    success = True
                    break
            if ctrl is not None and ctrl.insert_done:
                if insert_ok:
                    success = True
                break

    phase = insert_runner.controller.phase_name if insert_runner.controller else "disabled"
    started = bool(insert_runner.controller and insert_runner.controller.handoff_happened)
    peg_dz = _peg_lift_m(raw_env, peg_rest_z)

    fail_reason = "ok" if success else "insert_timeout"
    if not pre_insert_ok:
        fail_reason = "pre_insert_not_ready"
    elif not started:
        fail_reason = "poseinsert_not_started"
    elif not success:
        fail_reason = f"insert_incomplete_{phase.lower()}"

    return InsertReport(
        success=bool(success),
        steps=int(step),
        insert_ok=insert_ok,
        handoff=started,
        phase=phase,
        peg_lift_m=peg_dz,
        fail_reason=fail_reason,
    )


def run_hybrid_insert(
    env,
    raw_env,
    *,
    max_steps: int = 2500,
    insert_cfg: HybridInsertConfig | None = None,
    settle_steps: int = 40,
    tray_rest_z: float | None = None,
    peg_rest_z: float | None = None,
) -> InsertReport:
    """Hybrid geometry insert (legacy skill_replay path, not PoseInsert)."""
    cfg = insert_cfg or HybridInsertConfig(handoff_confirm_frames=3, approach_xy_m=0.10)
    insert_runner = make_insert_runner("hybrid", hybrid_cfg=cfg)
    labeler = AssemblyContactLabeler(raw_env)
    if tray_rest_z is not None and peg_rest_z is not None:
        labeler._tray_rest_z = float(tray_rest_z)  # noqa: SLF001
        labeler._peg_rest_z = float(peg_rest_z)  # noqa: SLF001
    else:
        labeler.reset_reference(raw_env)
    peg_rest_z = float(labeler._peg_rest_z)  # noqa: SLF001

    insert_runner.on_reset(env)
    if insert_runner.controller is not None and tray_rest_z is not None:
        insert_runner.controller._peg_rest_z = float(peg_rest_z)  # noqa: SLF001
        if insert_runner.controller._labeler is not None:
            insert_runner.controller._labeler._tray_rest_z = float(tray_rest_z)  # noqa: SLF001
            insert_runner.controller._labeler._peg_rest_z = float(peg_rest_z)  # noqa: SLF001

    for _ in range(max(int(settle_steps), 1)):
        left = read_arm_action(raw_env, "left")
        right = read_arm_action(raw_env, "right")
        raw_env.step(action44_to_raw_dict(dual_arm23_to_action44(left, right)))

    from interaction_retarget.grasp.lift import execute_peg_lift
    from interaction_retarget.grasp.locked_hold import enforce_locked_passive
    from interaction_retarget.sim.contact import AssemblyContactDetector

    left = vec_to_arm_action(read_arm_action(raw_env, "left"))
    right = vec_to_arm_action(read_arm_action(raw_env, "right"))
    lift_det = AssemblyContactDetector(raw_env)
    lift_det._peg_rest_z = float(peg_rest_z)  # noqa: SLF001
    if tray_rest_z is not None:
        lift_det._tray_rest_z = float(tray_rest_z)  # noqa: SLF001
    if _peg_lift_m(raw_env, peg_rest_z) < cfg.lift_ready_m - 0.005:
        right = execute_peg_lift(
            raw_env,
            grasp_right=right,
            hold_left=left,
            detector=lift_det,
            for_insert=True,
            steps=100,
        )
        enforce_locked_passive(raw_env, locked_left=left, locked_right=right, n_substeps=20)

    _privileged_approach_to_handoff(env, raw_env, insert_runner, labeler, cfg=cfg, max_steps=900)
    _try_force_handoff(env, raw_env, insert_runner, labeler, cfg=cfg, peg_rest_z=peg_rest_z)

    success = False
    insert_ok = False
    step = 0
    for step in range(int(max_steps)):
        left = read_arm_action(raw_env, "left")
        right = read_arm_action(raw_env, "right")
        policy44 = dual_arm23_to_action44(left, right)
        insert_runner.observe(env, policy44)
        merged = insert_runner.merge(env, policy44)
        action46 = rotvec_dual_arm_to_policy(np.asarray(merged, dtype=np.float64).reshape(44))
        info = _env_step_info(env, action46.astype(np.float32))
        outcome = labeler.compute(raw_env)
        insert_ok = bool(outcome.insert_ok)
        if info.get("succeed"):
            success = True
            break
        if insert_runner.controller is not None and insert_runner.controller.phase_name == "RELEASE":
            if insert_ok:
                success = True
                break

    phase = insert_runner.controller.phase_name if insert_runner.controller else "disabled"
    handoff = bool(insert_runner.controller and insert_runner.controller.handoff_happened)
    peg_dz = _peg_lift_m(raw_env, peg_rest_z)
    fail_reason = "ok" if success else "insert_timeout"
    if not handoff and not success:
        fail_reason = "handoff_never"
    elif handoff and not success:
        fail_reason = f"insert_incomplete_{phase.lower()}"

    return InsertReport(
        success=bool(success),
        steps=int(step),
        insert_ok=insert_ok,
        handoff=handoff,
        phase=phase,
        peg_lift_m=peg_dz,
        fail_reason=fail_reason,
    )


# Backward-compatible alias for eval script imports.
run_insert_phase = run_pose_insert_phase
