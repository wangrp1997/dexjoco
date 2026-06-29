"""Distill canonical tray lift delta from demo zarr actions (no sim replay)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.replay import raw_flat_to_dict


@dataclass
class DistillLiftReport:
    num_episodes_used: int
    episode_indices: list[int]
    excluded_episode_indices: list[int]
    mocap_pos_std_m: float
    tray_z_delta_median_m: float


@dataclass
class CanonicalTrayLift:
    mocap_delta_world: np.ndarray
    hand_joint_median: np.ndarray
    tray_z_delta_world_m: float
    source_episode_indices: np.ndarray
    report: DistillLiftReport


def _episode_usable(entry: dict[str, Any], *, exclude_fallback: bool) -> bool:
    timing = entry.get("timing", {})
    if timing.get("tray_lift_start") is None:
        return False
    if timing.get("left_grasp_frame") is None:
        return False
    if not exclude_fallback:
        return True
    if timing.get("left_grasp_fallback"):
        return False
    return "tray_grasp_used_fallback" not in entry.get("timing_warnings", [])


def _left_arm23_from_action(action_flat: np.ndarray) -> np.ndarray:
    return np.asarray(raw_flat_to_dict(action_flat)["left"], dtype=np.float64).reshape(23)


def _lift_delta_from_zarr(
    entry: dict,
    *,
    grasp_frame: int,
    lift_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    actions, _, _ = load_zarr_episode(Path(entry["zarr_path"]))
    grasp_frame = int(np.clip(grasp_frame, 0, len(actions) - 1))
    lift_frame = int(np.clip(lift_frame, 0, len(actions) - 1))
    grasp = _left_arm23_from_action(actions[grasp_frame])
    lift = _left_arm23_from_action(actions[lift_frame])
    return lift[0:3] - grasp[0:3], lift[7:23].copy()


def distill_canonical_tray_lift(
    sidecar_dir: Path,
    *,
    exclude_fallback: bool = False,
    max_episodes: int = 20,
) -> CanonicalTrayLift:
    """Median left mocap world delta grasp→tray_lift_start from zarr commands."""
    sidecar_dir = Path(sidecar_dir)
    manifest = json.loads((sidecar_dir / "manifest.json").read_text(encoding="utf-8"))
    used: list[int] = []
    excluded: list[int] = []
    delta_list: list[np.ndarray] = []
    hand_list: list[np.ndarray] = []
    z_delta_list: list[float] = []

    for entry in manifest["episodes"]:
        ep_idx = int(entry["episode_index"])
        if not _episode_usable(entry, exclude_fallback=exclude_fallback):
            excluded.append(ep_idx)
            continue
        lift_frame = int(entry["timing"]["tray_lift_start"])
        grasp_frame = int(entry["timing"]["left_grasp_frame"])
        delta, lift_hand = _lift_delta_from_zarr(entry, grasp_frame=grasp_frame, lift_frame=lift_frame)
        delta_list.append(delta)
        hand_list.append(lift_hand)
        z_delta_list.append(float(delta[2]))
        used.append(ep_idx)
        if len(used) >= int(max_episodes):
            break

    if not used:
        raise ValueError("No episodes with tray_lift_start for lift distill")

    delta_stack = np.stack(delta_list, axis=0)
    delta_median = np.median(delta_stack, axis=0)
    hand_median = np.median(np.stack(hand_list, axis=0), axis=0)

    report = DistillLiftReport(
        num_episodes_used=len(used),
        episode_indices=used,
        excluded_episode_indices=excluded,
        mocap_pos_std_m=float(np.mean(np.linalg.norm(delta_stack - delta_median, axis=1))),
        tray_z_delta_median_m=float(np.median(z_delta_list)),
    )
    return CanonicalTrayLift(
        mocap_delta_world=delta_median,
        hand_joint_median=hand_median,
        tray_z_delta_world_m=float(np.median(z_delta_list)),
        source_episode_indices=np.asarray(used, dtype=np.int32),
        report=report,
    )


def save_canonical_tray_lift(proto: CanonicalTrayLift, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        mocap_delta_world=proto.mocap_delta_world,
        hand_joint_median=proto.hand_joint_median,
        tray_z_delta_world_m=np.asarray([proto.tray_z_delta_world_m], dtype=np.float64),
        source_episode_indices=proto.source_episode_indices,
    )
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps({"report": asdict(proto.report)}, indent=2), encoding="utf-8")
    return out_path


def load_canonical_tray_lift(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    if "mocap_delta_world" in data:
        delta = np.asarray(data["mocap_delta_world"], dtype=np.float64)
        tz = float(np.asarray(data["tray_z_delta_world_m"]).reshape(-1)[0])
    else:
        delta = np.zeros(3, dtype=np.float64)
        tz = float(np.asarray(data.get("tray_z_delta_obj_m", [0.02])).reshape(-1)[0])
    return {
        "mocap_delta_world": delta,
        "hand_joint_median": np.asarray(data["hand_joint_median"], dtype=np.float64),
        "tray_z_delta_world_m": tz,
        "source_episode_indices": np.asarray(data["source_episode_indices"], dtype=np.int32),
    }
