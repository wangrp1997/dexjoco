"""Offline skill library: median δ* grasp + per-demo lift / retrieval pose."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dexjoco.tasks import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

from interaction_retarget.constants import PEG_BODY, TRAY_BODY
from interaction_retarget.grasp.distill import load_canonical_grasp
from interaction_retarget.grasp.lift_reference import (
    DemoLiftReference,
    _entry_usable,
    extract_demo_lift_reference,
    load_demo_lift_reference,
    save_demo_lift_reference,
)
from interaction_retarget.skill_replay.demo_canonical import load_demo_canonical_grasp
from interaction_retarget.io.zarr_io import load_zarr_episode
from interaction_retarget.sim.replay import make_assembly_env
from interaction_retarget.skill_replay.retrieval import ObjectPose, ScenePose


def demo_lift_ref_as_dict(ref: DemoLiftReference) -> dict[str, Any]:
    f = ref.frames
    return {
        "episode_index": ref.episode_index,
        "tray_lift_end_frame": f.tray_lift_end_frame,
        "peg_lift_end_frame": f.peg_lift_end_frame,
        "tray_mocap_pos_obj": ref.tray.mocap_pos_obj,
        "tray_mocap_quat_obj": ref.tray.mocap_quat_obj,
        "tray_lift_num_demo_frames": ref.tray.num_demo_frames,
        "peg_mocap_pos_obj": ref.peg.mocap_pos_obj,
        "peg_mocap_quat_obj": ref.peg.mocap_quat_obj,
        "peg_lift_num_demo_frames": ref.peg.num_demo_frames,
        "tray_hold_steps_before_peg": ref.tray_hold_steps_before_peg,
    }


def _read_object_pose(raw_env, body_name: str) -> ObjectPose:
    bid = int(raw_env._model.body(body_name).id)
    data = raw_env._data
    return ObjectPose(
        pos=np.asarray(data.xpos[bid], dtype=np.float64).copy(),
        quat=np.asarray(data.xquat[bid], dtype=np.float64).copy(),
    )


def episode_initial_scene_pose(
    entry: dict[str, Any],
    *,
    seed_base: int = 0,
) -> ScenePose:
    """Object poses at demo start (after optional zarr initial_state restore)."""
    actions, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    del actions
    ep_idx = int(entry["episode_index"])
    env = make_assembly_env(seed=int(seed_base) + ep_idx, randomize=False)
    raw = env.unwrapped
    try:
        env.reset()
        config = CONFIG_MAPPING["bimanual_assembly"]()
        if initial_state is not None and has_restorer("bimanual_assembly"):
            restore_initial_state(env, "bimanual_assembly", config, initial_state)
        return ScenePose(
            tray=_read_object_pose(raw, TRAY_BODY),
            peg=_read_object_pose(raw, PEG_BODY),
        )
    finally:
        env.close()


@dataclass
class DemoSkill:
    episode_index: int
    tray_canonical: dict[str, Any]  # per-demo δ* when use_per_demo_canonical else median
    peg_canonical: dict[str, Any]
    lift_ref: dict[str, Any]  # per-demo lift anchor trajectory
    initial_pose: ScenePose
    per_demo_canonical: bool = True


class SkillLibrary:
    """Retrieve per-demo lift/skill; grasp δ* from distilled median canonical."""

    CACHE_DIR_NAME = "skill_replay_cache"

    def __init__(
        self,
        sidecar_dir: Path,
        *,
        seed_base: int = 0,
        exclude_fallback: bool = True,
    ) -> None:
        self.sidecar_dir = Path(sidecar_dir)
        self.seed_base = int(seed_base)
        self.exclude_fallback = bool(exclude_fallback)
        self.cache_dir = self.sidecar_dir / self.CACHE_DIR_NAME
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._median_tray = load_canonical_grasp(self.sidecar_dir / "canonical_tray_grasp.npz")
        self._median_peg = load_canonical_grasp(self.sidecar_dir / "canonical_peg_grasp.npz")
        manifest = json.loads((self.sidecar_dir / "manifest.json").read_text(encoding="utf-8"))
        self._entries: list[dict[str, Any]] = []
        for entry in manifest["episodes"]:
            if not _entry_usable(entry):
                continue
            if exclude_fallback:
                timing = entry.get("timing", {})
                if timing.get("left_grasp_fallback") or timing.get("right_grasp_fallback"):
                    continue
            self._entries.append(entry)
        self._pose_index: list[tuple[int, ScenePose]] | None = None

    @property
    def episode_indices(self) -> list[int]:
        return [int(e["episode_index"]) for e in self._entries]

    def build_pose_index(self) -> list[tuple[int, ScenePose]]:
        if self._pose_index is not None:
            return self._pose_index
        index_path = self.cache_dir / "initial_poses.json"
        cached: dict[str, Any] | None = None
        if index_path.is_file():
            cached = json.loads(index_path.read_text(encoding="utf-8"))
        poses: list[tuple[int, ScenePose]] = []
        records: list[dict[str, Any]] = []
        for entry in self._entries:
            ep_idx = int(entry["episode_index"])
            key = str(ep_idx)
            if cached and key in cached.get("episodes", {}):
                rec = cached["episodes"][key]
                scene = ScenePose(
                    tray=ObjectPose(
                        pos=np.asarray(rec["tray_pos"], dtype=np.float64),
                        quat=np.asarray(rec["tray_quat"], dtype=np.float64),
                    ),
                    peg=ObjectPose(
                        pos=np.asarray(rec["peg_pos"], dtype=np.float64),
                        quat=np.asarray(rec["peg_quat"], dtype=np.float64),
                    ),
                )
            else:
                scene = episode_initial_scene_pose(entry, seed_base=self.seed_base)
                records.append(
                    {
                        "episode_index": ep_idx,
                        "tray_pos": scene.tray.pos.tolist(),
                        "tray_quat": scene.tray.quat.tolist(),
                        "peg_pos": scene.peg.pos.tolist(),
                        "peg_quat": scene.peg.quat.tolist(),
                    }
                )
            poses.append((ep_idx, scene))
        if records or cached is None:
            ep_map = {str(r["episode_index"]): r for r in records}
            if cached:
                ep_map = {**cached.get("episodes", {}), **ep_map}
            index_path.write_text(
                json.dumps({"episodes": ep_map}, indent=2),
                encoding="utf-8",
            )
        self._pose_index = poses
        return poses

    def _lift_cache_path(self, episode_index: int) -> Path:
        return self.cache_dir / f"episode_{int(episode_index):03d}_lift_ref.npz"

    def load_lift_ref(self, episode_index: int) -> dict[str, Any]:
        path = self._lift_cache_path(episode_index)
        if path.is_file():
            return load_demo_lift_reference(path)
        ref = extract_demo_lift_reference(
            self.sidecar_dir,
            episode_index=int(episode_index),
            pick_seed=self.seed_base,
        )
        save_demo_lift_reference(ref, path)
        return demo_lift_ref_as_dict(ref)

    def load_demo(
        self,
        episode_index: int,
        *,
        use_per_demo_canonical: bool = True,
    ) -> DemoSkill:
        entry = next(
            e for e in self._entries if int(e["episode_index"]) == int(episode_index)
        )
        poses = self.build_pose_index()
        initial = next(p for i, p in poses if i == int(episode_index))
        if use_per_demo_canonical:
            tray_c = load_demo_canonical_grasp(
                self.sidecar_dir, entry, "tray", seed_base=self.seed_base
            )
            peg_c = load_demo_canonical_grasp(
                self.sidecar_dir, entry, "peg", seed_base=self.seed_base
            )
        else:
            tray_c = self._median_tray
            peg_c = self._median_peg
        return DemoSkill(
            episode_index=int(episode_index),
            tray_canonical=tray_c,
            peg_canonical=peg_c,
            lift_ref=self.load_lift_ref(episode_index),
            initial_pose=initial,
            per_demo_canonical=bool(use_per_demo_canonical),
        )
