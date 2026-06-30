"""Export privileged sim poses for PoseInsert training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_insert.assembly_contacts import AssemblyContactLabeler
from hybrid_insert.geometry import (
    hole_opening_axis,
    in_approach_cylinder,
    peg_insert_end_pos,
    tip_socket_distance,
)
from pose_insert.poses import source_in_target_poses, xpos_xmat_to_pose7, xpos_xquat_to_pose7
from pose_insert.wrist_actions import zarr_flat_to_dual_wrist12, zarr_flat_to_action44
from pose_insert.segments import InsertSegment, detect_insert_segment
from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state
from interaction_retarget.constants import LEFT_HAND_BODIES, PEG_BODY, RIGHT_HAND_BODIES, TRAY_BODY
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.contact import AssemblyContactDetector
from interaction_retarget.sim.grasp_timing import GraspTiming
from interaction_retarget.sim.hand_geom import hand_keypoints_world
from interaction_retarget.sim.replay import ReplayStep, ReplayTrace, make_assembly_env, raw_flat_to_dict

_SOCKET_SITE = "industreal_tray_insert_round_peg_8mm_socket_site"
_BOTTOM_GEOM = "industreal_tray_insert_round_peg_8mm_bottom_contact"
_FLANGE_POS_SENSOR = "franka/flange_pos_right"
_FLANGE_QUAT_SENSOR = "franka/flange_quat_right"
_FLANGE_POS_LEFT_SENSOR = "franka/flange_pos_left"
_FLANGE_QUAT_LEFT_SENSOR = "franka/flange_quat_left"
_DEFAULT_APPROACH_XY_M = 0.06
_DEFAULT_APPROACH_Z_MIN_M = 0.01
_DEFAULT_MAX_LAST_TIP_DIST_MM = 20.0
_DEFAULT_MAX_MIN_TIP_DIST_MM = 15.0


@dataclass
class ExportReport:
    episode_index: int
    zarr_path: str
    output_dir: Path
    segment: InsertSegment
    num_frames: int
    first_tip_socket_dist_mm: float
    last_tip_socket_dist_mm: float
    min_tip_socket_dist_mm: float
    has_insert_ok: bool


@dataclass(frozen=True)
class ExportSkip:
    episode_index: int
    zarr_path: str
    reason: str
    has_insert_ok: bool
    min_tip_socket_dist_mm: float
    last_tip_socket_dist_mm: float


@dataclass
class _FrameRecord:
    source_pose7: np.ndarray
    target_pose7: np.ndarray
    ee_pose7: np.ndarray
    left_ee_pose7: np.ndarray
    insert_ok: bool
    approach_ready: bool
    tip_socket_dist_m: float


def _timing_from_entry(entry: dict[str, Any]) -> GraspTiming:
    timing = entry.get("timing", {})
    return GraspTiming(
        left_grasp_frame=timing.get("left_grasp_frame"),
        right_grasp_frame=timing.get("right_grasp_frame"),
        tray_lift_start=timing.get("tray_lift_start"),
        peg_lift_start=timing.get("peg_lift_start"),
        left_grasp_fallback=bool(timing.get("left_grasp_fallback", False)),
        right_grasp_fallback=bool(timing.get("right_grasp_fallback", False)),
    )


def _gripper_speed(data, qvel_adrs: np.ndarray) -> float:
    return float(np.linalg.norm(data.qvel[qvel_adrs]))


def _approach_ready_at_step(raw_env, *, approach_xy_m: float, approach_z_min_m: float) -> bool:
    model = raw_env._model
    data = raw_env._data
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
    return bool(
        in_approach_cylinder(
            tip,
            socket,
            hole_axis,
            xy_tol_m=approach_xy_m,
            z_min_m=approach_z_min_m,
            z_max_m=None,
        )
    )


def _record_frame(raw_env, labeler: AssemblyContactLabeler) -> _FrameRecord:
    model = raw_env._model
    data = raw_env._data
    peg_id = int(model.body(PEG_BODY).id)
    socket_id = int(model.site(_SOCKET_SITE).id)

    source_pose7 = xpos_xquat_to_pose7(data.xpos[peg_id], data.xquat[peg_id])
    target_pose7 = xpos_xmat_to_pose7(data.site_xpos[socket_id], data.site_xmat[socket_id])
    ee_pose7 = xpos_xquat_to_pose7(
        data.sensor(_FLANGE_POS_SENSOR).data,
        data.sensor(_FLANGE_QUAT_SENSOR).data,
    )
    left_ee_pose7 = xpos_xquat_to_pose7(
        data.sensor(_FLANGE_POS_LEFT_SENSOR).data,
        data.sensor(_FLANGE_QUAT_LEFT_SENSOR).data,
    )

    tip = peg_insert_end_pos(data.xpos[peg_id], data.xmat[peg_id])
    socket = np.asarray(data.site_xpos[socket_id], dtype=np.float64)
    outcome = labeler.compute(raw_env)
    return _FrameRecord(
        source_pose7=source_pose7,
        target_pose7=target_pose7,
        ee_pose7=ee_pose7,
        left_ee_pose7=left_ee_pose7,
        insert_ok=bool(outcome.insert_ok),
        approach_ready=_approach_ready_at_step(
            raw_env,
            approach_xy_m=_DEFAULT_APPROACH_XY_M,
            approach_z_min_m=_DEFAULT_APPROACH_Z_MIN_M,
        ),
        tip_socket_dist_m=tip_socket_distance(tip, socket),
    )


def replay_pose_sequence(
    entry: dict[str, Any],
    *,
    seed: int = 0,
) -> tuple[list[_FrameRecord], ReplayTrace]:
    """Privileged replay of one zarr episode; record peg/socket/flange poses each step."""
    actions, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    env = make_assembly_env(seed=int(seed), randomize=False)
    raw = env.unwrapped
    model = raw._model
    config = CONFIG_MAPPING["bimanual_assembly"]()
    labeler = AssemblyContactLabeler(raw)
    detector = AssemblyContactDetector(raw)
    tray_id = int(model.body(TRAY_BODY).id)
    peg_id = int(model.body(PEG_BODY).id)

    left_qvel_adr = np.asarray(
        [int(model.joint(n).dofadr) for n in raw._allegro_joint_left_names], dtype=int
    )
    right_qvel_adr = np.asarray(
        [int(model.joint(n).dofadr) for n in raw._allegro_joint_right_names], dtype=int
    )

    frames: list[_FrameRecord] = []
    trace = ReplayTrace()
    try:
        env.reset()
        if initial_state is not None and has_restorer("bimanual_assembly"):
            restore_initial_state(env, "bimanual_assembly", config, initial_state)
        labeler.reset_reference(raw)
        detector.reset_reference(raw)

        for action in actions:
            raw.step(raw_flat_to_dict(action))
            frames.append(_record_frame(raw, labeler))
            data = raw._data
            contact = detector.compute(raw)
            trace.steps.append(
                ReplayStep(
                    contact=contact,
                    tray_pos=data.xpos[tray_id].copy(),
                    tray_quat=data.xquat[tray_id].copy(),
                    peg_pos=data.xpos[peg_id].copy(),
                    peg_quat=data.xquat[peg_id].copy(),
                    left_hand_world=hand_keypoints_world(model, data, LEFT_HAND_BODIES),
                    right_hand_world=hand_keypoints_world(model, data, RIGHT_HAND_BODIES),
                    left_gripper_speed=_gripper_speed(data, left_qvel_adr),
                    right_gripper_speed=_gripper_speed(data, right_qvel_adr),
                    tray_z=float(data.xpos[tray_id, 2]),
                    peg_z=float(data.xpos[peg_id, 2]),
                )
            )

        trace.info = {
            "num_steps": len(trace.steps),
            "seed": int(seed),
            "used_initial_state": initial_state is not None,
        }
        return frames, trace
    finally:
        env.close()


def _validate_export(
    *,
    insert_ok: np.ndarray,
    tip_socket_dist_m: np.ndarray,
    segment: InsertSegment,
    require_insert_ok: bool,
    max_last_tip_dist_mm: float,
    max_min_tip_dist_mm: float,
) -> ExportSkip | None:
    has_ok = bool(insert_ok.any())
    min_mm = float(segment.min_tip_dist_m * 1000.0)
    last_mm = float(tip_socket_dist_m[segment.end_frame] * 1000.0)

    if require_insert_ok and not has_ok:
        return ExportSkip(
            episode_index=-1,
            zarr_path="",
            reason="no_insert_contact_in_replay",
            has_insert_ok=False,
            min_tip_socket_dist_mm=min_mm,
            last_tip_socket_dist_mm=last_mm,
        )
    if min_mm > max_min_tip_dist_mm:
        return ExportSkip(
            episode_index=-1,
            zarr_path="",
            reason=f"min_tip_dist>{max_min_tip_dist_mm:.0f}mm",
            has_insert_ok=has_ok,
            min_tip_socket_dist_mm=min_mm,
            last_tip_socket_dist_mm=last_mm,
        )
    if has_ok and last_mm > max_last_tip_dist_mm:
        return ExportSkip(
            episode_index=-1,
            zarr_path="",
            reason=f"last_tip_dist>{max_last_tip_dist_mm:.0f}mm",
            has_insert_ok=True,
            min_tip_socket_dist_mm=min_mm,
            last_tip_socket_dist_mm=last_mm,
        )
    return None


def _save_episode_bundle(
    out_demo_dir: Path,
    *,
    frames: list[_FrameRecord],
    segment: InsertSegment,
    entry: dict[str, Any],
    seed: int,
    zarr_actions: np.ndarray,
) -> None:
    sl = slice(segment.start_frame, segment.end_frame + 1)
    subset = frames[sl]
    source_pose = np.stack([f.source_pose7 for f in subset], axis=0)
    target_pose = np.stack([f.target_pose7 for f in subset], axis=0)
    ee_pose = np.stack([f.ee_pose7 for f in subset], axis=0)
    left_ee_pose = np.stack([f.left_ee_pose7 for f in subset], axis=0)
    source_in_target = source_in_target_poses(source_pose, target_pose)
    zarr_seg = np.asarray(zarr_actions[sl], dtype=np.float64)
    dual_wrist = np.stack([zarr_flat_to_dual_wrist12(a) for a in zarr_seg], axis=0)
    action44 = np.stack([zarr_flat_to_action44(a) for a in zarr_seg], axis=0)

    out_demo_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_demo_dir / "source_pose.npy", source_pose)
    np.save(out_demo_dir / "target_pose.npy", target_pose)
    np.save(out_demo_dir / "ee_pose.npy", ee_pose)
    np.save(out_demo_dir / "left_ee_pose.npy", left_ee_pose)
    np.save(out_demo_dir / "source_in_target.npy", source_in_target)
    np.save(out_demo_dir / "dual_wrist_action.npy", dual_wrist)
    np.save(out_demo_dir / "action44.npy", action44)

    meta = {
        "episode_index": int(entry["episode_index"]),
        "zarr_path": str(entry["zarr_path"]),
        "seed": int(seed),
        "segment": asdict(segment),
        "num_frames": int(source_pose.shape[0]),
        "coord_frame": "world",
        "source_body": PEG_BODY,
        "target_site": _SOCKET_SITE,
        "ee_sensor": _FLANGE_POS_SENSOR,
        "left_ee_sensor": _FLANGE_POS_LEFT_SENSOR,
        "action_format": "action44",
        "action_dim": 44,
    }
    (out_demo_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def export_episode(
    entry: dict[str, Any],
    output_root: Path,
    *,
    split: str = "train",
    demo_folder: str | None = None,
    seed: int = 0,
    require_insert_ok: bool = True,
    max_last_tip_dist_mm: float = _DEFAULT_MAX_LAST_TIP_DIST_MM,
    max_min_tip_dist_mm: float = _DEFAULT_MAX_MIN_TIP_DIST_MM,
) -> ExportReport | ExportSkip:
    """Export one manifest episode; skip when replay insert validation fails."""
    zarr_actions, _, _ = load_zarr_episode(Path(entry["zarr_path"]))
    frames, trace = replay_pose_sequence(entry, seed=seed)
    timing = _timing_from_entry(entry)
    insert_ok = np.asarray([f.insert_ok for f in frames], dtype=bool)
    approach_ready = np.asarray([f.approach_ready for f in frames], dtype=bool)
    tip_socket_dist_m = np.asarray([f.tip_socket_dist_m for f in frames], dtype=np.float64)
    segment = detect_insert_segment(
        trace,
        timing,
        insert_ok=insert_ok,
        approach_ready=approach_ready,
        tip_socket_dist_m=tip_socket_dist_m,
    )

    skip = _validate_export(
        insert_ok=insert_ok,
        tip_socket_dist_m=tip_socket_dist_m,
        segment=segment,
        require_insert_ok=require_insert_ok,
        max_last_tip_dist_mm=max_last_tip_dist_mm,
        max_min_tip_dist_mm=max_min_tip_dist_mm,
    )
    if skip is not None:
        return ExportSkip(
            episode_index=int(entry["episode_index"]),
            zarr_path=str(entry["zarr_path"]),
            reason=skip.reason,
            has_insert_ok=bool(insert_ok.any()),
            min_tip_socket_dist_mm=float(segment.min_tip_dist_m * 1000.0),
            last_tip_socket_dist_mm=float(tip_socket_dist_m[segment.end_frame] * 1000.0),
        )

    folder = demo_folder if demo_folder is not None else str(int(entry["episode_index"]))
    out_demo_dir = output_root / split / folder
    _save_episode_bundle(
        out_demo_dir,
        frames=frames,
        segment=segment,
        entry=entry,
        seed=seed,
        zarr_actions=zarr_actions,
    )

    subset = frames[segment.start_frame : segment.end_frame + 1]
    return ExportReport(
        episode_index=int(entry["episode_index"]),
        zarr_path=str(entry["zarr_path"]),
        output_dir=out_demo_dir,
        segment=segment,
        num_frames=segment.num_frames,
        first_tip_socket_dist_mm=float(subset[0].tip_socket_dist_m * 1000.0),
        last_tip_socket_dist_mm=float(subset[-1].tip_socket_dist_m * 1000.0),
        min_tip_socket_dist_mm=float(segment.min_tip_dist_m * 1000.0),
        has_insert_ok=bool(insert_ok.any()),
    )


def export_manifest(
    manifest_path: Path,
    output_root: Path,
    *,
    episode_indices: list[int] | None = None,
    split: str = "train",
    seed: int = 0,
    require_insert_ok: bool = True,
    max_last_tip_dist_mm: float = _DEFAULT_MAX_LAST_TIP_DIST_MM,
    max_min_tip_dist_mm: float = _DEFAULT_MAX_MIN_TIP_DIST_MM,
    show_progress: bool = True,
) -> tuple[list[ExportReport], list[ExportSkip]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = []
    for entry in manifest["episodes"]:
        ep = int(entry["episode_index"])
        if episode_indices is not None and ep not in episode_indices:
            continue
        timing = entry.get("timing", {})
        if timing.get("peg_lift_start") is None or timing.get("right_grasp_frame") is None:
            continue
        candidates.append(entry)

    iterator = candidates
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(candidates, desc="export insert", unit="ep")
        except ImportError:
            pass

    reports: list[ExportReport] = []
    skipped: list[ExportSkip] = []
    for entry in iterator:
        ep = int(entry["episode_index"])
        result = export_episode(
            entry,
            output_root,
            split=split,
            seed=seed,
            require_insert_ok=require_insert_ok,
            max_last_tip_dist_mm=max_last_tip_dist_mm,
            max_min_tip_dist_mm=max_min_tip_dist_mm,
        )
        if isinstance(result, ExportSkip):
            skipped.append(result)
            msg = (
                f"skip ep{ep}: {result.reason} "
                f"min={result.min_tip_socket_dist_mm:.1f}mm "
                f"last={result.last_tip_socket_dist_mm:.1f}mm "
                f"insert_ok={result.has_insert_ok}"
            )
            if show_progress and hasattr(iterator, "write"):
                iterator.write(msg)
            else:
                print(msg, flush=True)
            continue

        reports.append(result)
        msg = (
            f"ok ep{ep}: frames={result.num_frames} "
            f"[{result.segment.start_frame},{result.segment.end_frame}] "
            f"tip {result.first_tip_socket_dist_mm:.1f}->{result.last_tip_socket_dist_mm:.1f}mm "
            f"min={result.min_tip_socket_dist_mm:.1f}mm"
        )
        if show_progress and hasattr(iterator, "write"):
            iterator.write(msg)
        else:
            print(msg, flush=True)

    summary = {
        "task": manifest.get("task", "bimanual_assembly"),
        "num_exported": len(reports),
        "num_skipped": len(skipped),
        "require_insert_ok": require_insert_ok,
        "episodes": [
            {
                "episode_index": r.episode_index,
                "output_dir": str(r.output_dir),
                "num_frames": r.num_frames,
                "segment": asdict(r.segment),
                "min_tip_socket_dist_mm": r.min_tip_socket_dist_mm,
                "last_tip_socket_dist_mm": r.last_tip_socket_dist_mm,
                "has_insert_ok": r.has_insert_ok,
            }
            for r in reports
        ],
        "skipped": [asdict(s) for s in skipped],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return reports, skipped
