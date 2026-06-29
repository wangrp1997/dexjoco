"""Extract lift segment waypoints from one demo (GenHand grasp→liftup + object-frame mocap)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from interaction_retarget.constants import MIN_GRASP_CONTACT_COUNT, PEG_BODY, TRAY_BODY
from interaction_retarget.io.sidecar import timing_from_trace
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.grasp_timing import GraspTiming
from interaction_retarget.sim.replay import ReplayTrace, make_assembly_env, raw_flat_to_dict, replay_episode
from interaction_retarget.sim.settle import read_arm_action
from interaction_retarget.transforms import relative_mocap_in_object_frame

Side = Literal["left", "right"]
ObjectName = Literal["tray", "peg"]

DEMO_LIFT_REFERENCE_NAME = "demo_lift_reference.npz"
_TRAY_LIFT_END_MARGIN = 5
_PEG_LIFT_SEARCH_FRAMES = 150
_MAX_LIFT_WAYPOINTS = 64


@dataclass
class LiftEndFrames:
    tray_grasp_frame: int
    tray_lift_start_frame: int
    tray_lift_end_frame: int
    peg_grasp_frame: int
    peg_lift_start_frame: int
    peg_lift_end_frame: int


@dataclass
class LiftSegmentObj:
    mocap_pos_obj: np.ndarray
    mocap_quat_obj: np.ndarray
    num_demo_frames: int


@dataclass
class DemoLiftReference:
    episode_index: int
    zarr_path: str
    frames: LiftEndFrames
    tray: LiftSegmentObj
    peg: LiftSegmentObj
    tray_hold_steps_before_peg: int


def _entry_usable(entry: dict[str, Any]) -> bool:
    timing = entry.get("timing", {})
    need = ("left_grasp_frame", "tray_lift_start", "right_grasp_frame", "peg_lift_start")
    return all(timing.get(k) is not None for k in need)


def _object_z(step, object_name: ObjectName) -> float:
    return float(step.tray_z if object_name == "tray" else step.peg_z)


def _contact_count(step, object_name: ObjectName) -> int:
    if object_name == "tray":
        return int(step.contact.tray_contact_count)
    return int(step.contact.peg_contact_count)


def detect_lift_end_frames(trace: ReplayTrace, timing: GraspTiming) -> LiftEndFrames:
    """Lift-end = max object_z with contact before next phase (peg: before descent)."""
    if timing.tray_lift_start is None or timing.left_grasp_frame is None:
        raise ValueError("missing tray grasp/lift timing")
    if timing.right_grasp_frame is None or timing.peg_lift_start is None:
        raise ValueError("missing peg grasp/lift timing")

    tray_end = _max_z_contact_frame(
        trace,
        object_name="tray",
        start=int(timing.tray_lift_start),
        end=max(int(timing.tray_lift_start) + 1, int(timing.right_grasp_frame) - _TRAY_LIFT_END_MARGIN),
    )
    peg_start = int(timing.peg_lift_start)
    peg_end = _max_z_contact_frame(
        trace,
        object_name="peg",
        start=peg_start,
        end=min(len(trace.steps), peg_start + _PEG_LIFT_SEARCH_FRAMES),
    )
    return LiftEndFrames(
        tray_grasp_frame=int(timing.left_grasp_frame),
        tray_lift_start_frame=int(timing.tray_lift_start),
        tray_lift_end_frame=tray_end,
        peg_grasp_frame=int(timing.right_grasp_frame),
        peg_lift_start_frame=int(timing.peg_lift_start),
        peg_lift_end_frame=peg_end,
    )


def _max_z_contact_frame(
    trace: ReplayTrace,
    *,
    object_name: ObjectName,
    start: int,
    end: int,
    min_contact: int = MIN_GRASP_CONTACT_COUNT,
) -> int:
    start = int(np.clip(start, 0, len(trace.steps) - 1))
    end = int(np.clip(end, start + 1, len(trace.steps)))
    best_t = start
    best_z = -np.inf
    for t in range(start, end):
        step = trace.steps[t]
        if _contact_count(step, object_name) < min_contact:
            continue
        z = _object_z(step, object_name)
        if z > best_z or (z == best_z and t > best_t):
            best_z = z
            best_t = t
    return best_t


def _subsample_indices(start: int, end: int, *, max_points: int = _MAX_LIFT_WAYPOINTS) -> np.ndarray:
    start, end = int(start), int(end)
    if end <= start:
        return np.asarray([start], dtype=np.int32)
    full = np.arange(start, end + 1, dtype=np.int32)
    if full.size <= max_points:
        return full
    idx = np.linspace(0, full.size - 1, num=max_points, dtype=np.int64)
    return full[idx]


def _replay_lift_segments(
    entry: dict,
    *,
    lift_frames: LiftEndFrames,
    seed: int,
) -> tuple[LiftSegmentObj, LiftSegmentObj]:
    """One sim replay: mocap in object frame along lift_start→lift_end (GenHand lift segment)."""
    from dexjoco.tasks import CONFIG_MAPPING
    from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

    actions, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    tray_idx = _subsample_indices(lift_frames.tray_lift_start_frame, lift_frames.tray_lift_end_frame)
    peg_idx = _subsample_indices(lift_frames.peg_lift_start_frame, lift_frames.peg_lift_end_frame)
    need = sorted(set(tray_idx.tolist()) | set(peg_idx.tolist()))
    end = need[-1]

    tray_pos: list[np.ndarray] = []
    tray_quat: list[np.ndarray] = []
    peg_pos: list[np.ndarray] = []
    peg_quat: list[np.ndarray] = []
    tray_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    peg_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    env = make_assembly_env(seed=int(seed), randomize=False)
    raw = env.unwrapped
    try:
        env.reset()
        config = CONFIG_MAPPING["bimanual_assembly"]()
        if initial_state is not None and has_restorer("bimanual_assembly"):
            restore_initial_state(env, "bimanual_assembly", config, initial_state)
        tray_id = raw._model.body(TRAY_BODY).id
        peg_id = raw._model.body(PEG_BODY).id
        for i, action in enumerate(actions[: end + 1]):
            raw.step(raw_flat_to_dict(action))
            if i not in need:
                continue
            data = raw._data
            tray_p = np.asarray(data.xpos[tray_id], dtype=np.float64)
            tray_q = np.asarray(data.xquat[tray_id], dtype=np.float64)
            peg_p = np.asarray(data.xpos[peg_id], dtype=np.float64)
            peg_q = np.asarray(data.xquat[peg_id], dtype=np.float64)
            left = read_arm_action(raw, "left")
            right = read_arm_action(raw, "right")
            tray_by_frame[i] = relative_mocap_in_object_frame(left[0:3], left[3:7], tray_p, tray_q)
            peg_by_frame[i] = relative_mocap_in_object_frame(right[0:3], right[3:7], peg_p, peg_q)
        for fi in tray_idx:
            if fi not in tray_by_frame:
                raise RuntimeError(f"tray lift frame {fi} missing")
            p, q = tray_by_frame[int(fi)]
            tray_pos.append(p)
            tray_quat.append(q)
        for fi in peg_idx:
            if fi not in peg_by_frame:
                raise RuntimeError(f"peg lift frame {fi} missing")
            p, q = peg_by_frame[int(fi)]
            peg_pos.append(p)
            peg_quat.append(q)
    finally:
        env.close()

    tray_seg = LiftSegmentObj(
        mocap_pos_obj=np.stack(tray_pos, axis=0),
        mocap_quat_obj=np.stack(tray_quat, axis=0),
        num_demo_frames=int(lift_frames.tray_lift_end_frame - lift_frames.tray_lift_start_frame),
    )
    peg_seg = LiftSegmentObj(
        mocap_pos_obj=np.stack(peg_pos, axis=0),
        mocap_quat_obj=np.stack(peg_quat, axis=0),
        num_demo_frames=int(lift_frames.peg_lift_end_frame - lift_frames.peg_lift_start_frame),
    )
    return tray_seg, peg_seg


def extract_demo_lift_reference(
    sidecar_dir: Path,
    *,
    episode_index: int | None = None,
    pick_seed: int = 0,
) -> DemoLiftReference:
    sidecar_dir = Path(sidecar_dir)
    manifest = json.loads((sidecar_dir / "manifest.json").read_text(encoding="utf-8"))
    usable = [e for e in manifest["episodes"] if _entry_usable(e)]
    if not usable:
        raise ValueError("No episode with full tray/peg grasp+lift timing")

    if episode_index is None:
        rng = np.random.default_rng(int(pick_seed))
        entry = usable[int(rng.integers(0, len(usable)))]
    else:
        by_idx = {int(e["episode_index"]): e for e in usable}
        if int(episode_index) not in by_idx:
            raise ValueError(f"episode {episode_index} missing or incomplete timing")
        entry = by_idx[int(episode_index)]

    ep_idx = int(entry["episode_index"])
    actions, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    trace = replay_episode(
        actions,
        seed=int(pick_seed) + ep_idx,
        initial_state=initial_state,
        randomize=False,
    )
    timing = timing_from_trace(trace)
    lift_frames = detect_lift_end_frames(trace, timing)
    tray_seg, peg_seg = _replay_lift_segments(
        entry,
        lift_frames=lift_frames,
        seed=int(pick_seed) + ep_idx,
    )
    return DemoLiftReference(
        episode_index=ep_idx,
        zarr_path=str(entry["zarr_path"]),
        frames=lift_frames,
        tray=tray_seg,
        peg=peg_seg,
        tray_hold_steps_before_peg=max(
            int(lift_frames.peg_grasp_frame) - int(lift_frames.tray_lift_start_frame),
            1,
        ),
    )


def save_demo_lift_reference(ref: DemoLiftReference, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = ref.frames
    np.savez_compressed(
        out_path,
        episode_index=np.asarray([ref.episode_index], dtype=np.int32),
        zarr_path=np.asarray([ref.zarr_path]),
        tray_grasp_frame=np.asarray([f.tray_grasp_frame], dtype=np.int32),
        tray_lift_start_frame=np.asarray([f.tray_lift_start_frame], dtype=np.int32),
        tray_lift_end_frame=np.asarray([f.tray_lift_end_frame], dtype=np.int32),
        peg_grasp_frame=np.asarray([f.peg_grasp_frame], dtype=np.int32),
        peg_lift_start_frame=np.asarray([f.peg_lift_start_frame], dtype=np.int32),
        peg_lift_end_frame=np.asarray([f.peg_lift_end_frame], dtype=np.int32),
        tray_mocap_pos_obj=ref.tray.mocap_pos_obj,
        tray_mocap_quat_obj=ref.tray.mocap_quat_obj,
        tray_lift_num_demo_frames=np.asarray([ref.tray.num_demo_frames], dtype=np.int32),
        peg_mocap_pos_obj=ref.peg.mocap_pos_obj,
        peg_mocap_quat_obj=ref.peg.mocap_quat_obj,
        peg_lift_num_demo_frames=np.asarray([ref.peg.num_demo_frames], dtype=np.int32),
        tray_hold_steps_before_peg=np.asarray([ref.tray_hold_steps_before_peg], dtype=np.int32),
    )
    meta = {
        "episode_index": ref.episode_index,
        "zarr_path": ref.zarr_path,
        "frames": asdict(f),
        "tray_waypoints": int(ref.tray.mocap_pos_obj.shape[0]),
        "peg_waypoints": int(ref.peg.mocap_pos_obj.shape[0]),
        "tray_hold_steps_before_peg": ref.tray_hold_steps_before_peg,
        "refs": "GenHand trajectory.py grasp→liftup; hold gap demo peg_grasp-tray_lift_start; verify_side_hold (spider)",
        "definition": (
            "lift_end = max object_z+contact; hold before peg = demo peg_grasp-tray_lift_start; "
            "verify_side_hold (spider) before right arm moves"
        ),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_path


def load_demo_lift_reference(npz_path: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    out: dict[str, Any] = {
        "episode_index": int(np.asarray(data["episode_index"]).reshape(-1)[0]),
        "tray_lift_end_frame": int(np.asarray(data["tray_lift_end_frame"]).reshape(-1)[0]),
        "peg_lift_end_frame": int(np.asarray(data["peg_lift_end_frame"]).reshape(-1)[0]),
    }
    if "tray_mocap_pos_obj" in data:
        out["tray_mocap_pos_obj"] = np.asarray(data["tray_mocap_pos_obj"], dtype=np.float64)
        out["tray_mocap_quat_obj"] = np.asarray(data["tray_mocap_quat_obj"], dtype=np.float64)
        out["tray_lift_num_demo_frames"] = int(np.asarray(data["tray_lift_num_demo_frames"]).reshape(-1)[0])
        out["peg_mocap_pos_obj"] = np.asarray(data["peg_mocap_pos_obj"], dtype=np.float64)
        out["peg_mocap_quat_obj"] = np.asarray(data["peg_mocap_quat_obj"], dtype=np.float64)
        out["peg_lift_num_demo_frames"] = int(np.asarray(data["peg_lift_num_demo_frames"]).reshape(-1)[0])
    if "tray_hold_steps_before_peg" in data:
        out["tray_hold_steps_before_peg"] = int(np.asarray(data["tray_hold_steps_before_peg"]).reshape(-1)[0])
    elif "peg_grasp_frame" in data and "tray_lift_start_frame" in data:
        out["tray_hold_steps_before_peg"] = max(
            int(np.asarray(data["peg_grasp_frame"]).reshape(-1)[0])
            - int(np.asarray(data["tray_lift_start_frame"]).reshape(-1)[0]),
            1,
        )
    else:
        # legacy: endpoint delta only
        out["tray_mocap_delta_world"] = np.asarray(data["tray_mocap_delta_world"], dtype=np.float64)
        out["peg_mocap_delta_world"] = np.asarray(data["peg_mocap_delta_world"], dtype=np.float64)
    return out


def default_demo_lift_path(sidecar_dir: Path) -> Path:
    return Path(sidecar_dir) / DEMO_LIFT_REFERENCE_NAME
