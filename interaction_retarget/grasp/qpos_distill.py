"""Distill object-frame qpos + MuJoCo contact targets from demo grasp frames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from interaction_retarget.grasp.contact_targets import ContactTargetSet, record_contact_targets_obj
from interaction_retarget.grasp.distill import (
    _episode_excluded,
    _grasp_frame_key,
    _hand_side,
    load_episode_snapshots,
)
from interaction_retarget.transforms import relative_mocap_in_object_frame


@dataclass
class QposGraspPrototype:
    object_name: str
    hand_side: str
    mocap_pos_obj: np.ndarray
    mocap_quat_obj: np.ndarray
    hand_joint: np.ndarray
    passive_action23: np.ndarray  # other arm at grasp frame (world)
    grasp_action23: np.ndarray  # active arm at grasp frame (world, settled)
    squeeze_action23: np.ndarray  # active arm at pre-lift squeeze frame (world)
    contact_targets: ContactTargetSet
    representative_episode_index: int
    source_episode_indices: list[int]

    def save_npz(self, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ct = self.contact_targets
        np.savez_compressed(
            out_path,
            object_name=np.asarray([self.object_name]),
            hand_side=np.asarray([self.hand_side]),
            mocap_pos_obj=self.mocap_pos_obj,
            mocap_quat_obj=self.mocap_quat_obj,
            hand_joint=self.hand_joint,
            passive_action23=self.passive_action23,
            grasp_action23=self.grasp_action23,
            squeeze_action23=self.squeeze_action23,
            contact_pos_obj=ct.pos_obj,
            contact_normal_obj=ct.normal_obj,
            contact_hand_bodies=np.asarray(ct.hand_bodies, dtype=object),
            contact_object_bodies=np.asarray(ct.object_bodies, dtype=object),
            representative_episode_index=np.asarray([self.representative_episode_index], dtype=np.int32),
            source_episode_indices=np.asarray(self.source_episode_indices, dtype=np.int32),
        )
        meta = {
            "object_name": self.object_name,
            "hand_side": self.hand_side,
            "npz_path": str(out_path),
            "representative_episode_index": self.representative_episode_index,
            "contact_count": ct.count,
            "source_episode_indices": self.source_episode_indices,
        }
        out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out_path


def load_qpos_grasp_npz(npz_path: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _squeeze_frame(timing: dict, object_name: str, *, n_actions: int) -> int:
    """Frame with tighter finger closure before lift (DexGraspBench squeeze stage)."""
    if object_name == "tray":
        grasp = int(timing["left_grasp_frame"])
        lift = timing.get("tray_lift_start")
    else:
        grasp = int(timing["right_grasp_frame"])
        lift = timing.get("peg_lift_start")
    if lift is None:
        return int(np.clip(grasp + 8, 0, n_actions - 1))
    return int(np.clip(min(grasp + 8, int(lift) - 1), grasp, n_actions - 1))


def _replay_unified_episode(
    entry: dict,
    *,
    seed_base: int = 0,
) -> dict[str, QposGraspPrototype]:
    """Single env replay → tray + peg prototypes (fast distill)."""
    from dexjoco.tasks import CONFIG_MAPPING
    from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

    from interaction_retarget.constants import PEG_BODY, TRAY_BODY
    from interaction_retarget.io.zarr_io import load_zarr_episode
    from interaction_retarget.sim.replay import make_assembly_env, raw_flat_to_dict
    from interaction_retarget.sim.settle import read_arm_action

    timing = entry["timing"]
    actions, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    ep_idx = int(entry["episode_index"])
    n_actions = len(actions)

    targets: dict[str, dict[str, int]] = {
        "tray": {
            "grasp": int(np.clip(int(timing["left_grasp_frame"]), 0, n_actions - 1)),
            "squeeze": _squeeze_frame(timing, "tray", n_actions=n_actions),
        },
        "peg": {
            "grasp": int(np.clip(int(timing["right_grasp_frame"]), 0, n_actions - 1)),
            "squeeze": _squeeze_frame(timing, "peg", n_actions=n_actions),
        },
    }
    end_frame = max(targets["tray"]["squeeze"], targets["peg"]["squeeze"])

    env = make_assembly_env(seed=int(seed_base) + ep_idx, randomize=False)
    raw = env.unwrapped
    snapshots: dict[str, dict] = {}
    try:
        env.reset()
        config = CONFIG_MAPPING["bimanual_assembly"]()
        if initial_state is not None and has_restorer("bimanual_assembly"):
            restore_initial_state(env, "bimanual_assembly", config, initial_state)

        for fi, action in enumerate(actions[: end_frame + 1]):
            raw.step(raw_flat_to_dict(action))
            for object_name in ("tray", "peg"):
                side = _hand_side(object_name)
                obj_body = TRAY_BODY if object_name == "tray" else PEG_BODY
                frames = targets[object_name]
                if fi not in (frames["grasp"], frames["squeeze"]):
                    continue
                obj_id = raw._model.body(obj_body).id
                obj_pos = np.asarray(raw._data.xpos[obj_id], dtype=np.float64).copy()
                obj_quat = np.asarray(raw._data.xquat[obj_id], dtype=np.float64).copy()
                arm23 = read_arm_action(raw, side)  # type: ignore[arg-type]
                passive_side = "right" if side == "left" else "left"
                passive23 = read_arm_action(raw, passive_side)  # type: ignore[arg-type]
                bucket = snapshots.setdefault(object_name, {})
                if fi == frames["grasp"]:
                    pos_obj, quat_obj = relative_mocap_in_object_frame(
                        arm23[0:3], arm23[3:7], obj_pos, obj_quat
                    )
                    bucket.update(
                        {
                            "mocap_pos_obj": pos_obj,
                            "mocap_quat_obj": quat_obj,
                            "hand_joint": arm23[7:23].copy(),
                            "passive_action23": passive23,
                            "grasp_action23": arm23.copy(),
                        }
                    )
                    bucket["grasp_contacts"] = record_contact_targets_obj(
                        raw,
                        side=side,  # type: ignore[arg-type]
                        object_name=object_name,  # type: ignore[arg-type]
                        source_episode_index=ep_idx,
                    )
                if fi == frames["squeeze"]:
                    bucket["squeeze_action23"] = arm23.copy()

        prototypes: dict[str, QposGraspPrototype] = {}
        for object_name in ("tray", "peg"):
            s = snapshots[object_name]
            squeeze23 = s.get("squeeze_action23", s["grasp_action23"])
            prototypes[object_name] = QposGraspPrototype(
                object_name=object_name,
                hand_side=_hand_side(object_name),
                mocap_pos_obj=s["mocap_pos_obj"],
                mocap_quat_obj=s["mocap_quat_obj"],
                hand_joint=s["hand_joint"],
                passive_action23=s["passive_action23"],
                grasp_action23=s["grasp_action23"],
                squeeze_action23=squeeze23,
                contact_targets=s["grasp_contacts"],
                representative_episode_index=ep_idx,
                source_episode_indices=[],
            )
        return prototypes
    finally:
        env.close()


def _replay_episode_grasp_qpos(
    entry: dict,
    *,
    object_name: str,
    seed_base: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ContactTargetSet]:
    """Replay demo to grasp frame; return mocap/fingers in object frame + passive arm + contacts."""
    from dexjoco.tasks import CONFIG_MAPPING
    from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

    from interaction_retarget.constants import PEG_BODY, TRAY_BODY
    from interaction_retarget.io.zarr_io import load_zarr_episode
    from interaction_retarget.sim.replay import make_assembly_env, raw_flat_to_dict
    from interaction_retarget.sim.settle import read_arm_action

    side = _hand_side(object_name)
    frame_key = _grasp_frame_key(object_name)
    actions, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    ep_idx = int(entry["episode_index"])
    frame = int(np.clip(int(entry["timing"][frame_key]), 0, len(actions) - 1))

    env = make_assembly_env(seed=int(seed_base) + ep_idx, randomize=False)
    raw = env.unwrapped
    obj_body = TRAY_BODY if object_name == "tray" else PEG_BODY
    try:
        env.reset()
        config = CONFIG_MAPPING["bimanual_assembly"]()
        if initial_state is not None and has_restorer("bimanual_assembly"):
            restore_initial_state(env, "bimanual_assembly", config, initial_state)
        n_sub = max(int(getattr(raw, "_n_substeps", 1)), 1)
        for action in actions[: frame + 1]:
            act = raw_flat_to_dict(action)
            raw.step(act)
        obj_id = raw._model.body(obj_body).id
        obj_pos = np.asarray(raw._data.xpos[obj_id], dtype=np.float64).copy()
        obj_quat = np.asarray(raw._data.xquat[obj_id], dtype=np.float64).copy()
        arm23 = read_arm_action(raw, side)  # type: ignore[arg-type]
        passive_side = "right" if side == "left" else "left"
        passive23 = read_arm_action(raw, passive_side)  # type: ignore[arg-type]
        pos_obj, quat_obj = relative_mocap_in_object_frame(arm23[0:3], arm23[3:7], obj_pos, obj_quat)
        contacts = record_contact_targets_obj(
            raw,
            side=side,  # type: ignore[arg-type]
            object_name=object_name,  # type: ignore[arg-type]
            source_episode_index=ep_idx,
        )
        return pos_obj, quat_obj, arm23[7:23].copy(), passive23, contacts
    finally:
        env.close()


def _fill_grasp_squeeze_on_proto(
    proto: QposGraspPrototype,
    entry: dict,
    *,
    seed_base: int = 0,
) -> QposGraspPrototype:
    """Backfill grasp/squeeze world qpos when loading old npz without those fields."""
    side = proto.hand_side
    timing = entry["timing"]
    actions, _, initial_state = load_zarr_episode(Path(entry["zarr_path"]))
    from dexjoco.tasks import CONFIG_MAPPING
    from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state
    from interaction_retarget.sim.replay import make_assembly_env, raw_flat_to_dict
    from interaction_retarget.sim.settle import read_arm_action

    object_name = proto.object_name
    grasp_f = int(timing[_grasp_frame_key(object_name)])
    squeeze_f = _squeeze_frame(timing, object_name, n_actions=len(actions))
    end_f = max(grasp_f, squeeze_f)
    env = make_assembly_env(seed=int(seed_base) + int(entry["episode_index"]), randomize=False)
    raw = env.unwrapped
    try:
        env.reset()
        config = CONFIG_MAPPING["bimanual_assembly"]()
        if initial_state is not None and has_restorer("bimanual_assembly"):
            restore_initial_state(env, "bimanual_assembly", config, initial_state)
        grasp23 = squeeze23 = None
        for fi, action in enumerate(actions[: end_f + 1]):
            raw.step(raw_flat_to_dict(action))
            if fi == grasp_f:
                grasp23 = read_arm_action(raw, side)  # type: ignore[arg-type]
            if fi == squeeze_f:
                squeeze23 = read_arm_action(raw, side)  # type: ignore[arg-type]
        assert grasp23 is not None and squeeze23 is not None
        return QposGraspPrototype(
            object_name=proto.object_name,
            hand_side=proto.hand_side,
            mocap_pos_obj=proto.mocap_pos_obj,
            mocap_quat_obj=proto.mocap_quat_obj,
            hand_joint=proto.hand_joint,
            passive_action23=proto.passive_action23,
            grasp_action23=grasp23,
            squeeze_action23=squeeze23,
            contact_targets=proto.contact_targets,
            representative_episode_index=proto.representative_episode_index,
            source_episode_indices=proto.source_episode_indices,
        )
    finally:
        env.close()


def _pick_representative_episode(
    sidecar_dir: Path,
    *,
    object_name: str,
    used_indices: list[int],
    seed_base: int = 0,
) -> int:
    """Medoid by mean contact-position RMSE across episodes."""
    manifest = json.loads((Path(sidecar_dir) / "manifest.json").read_text(encoding="utf-8"))
    by_index = {int(e["episode_index"]): e for e in manifest["episodes"]}

    summaries: list[tuple[int, ContactTargetSet, np.ndarray]] = []
    for ep in used_indices:
        entry = by_index[int(ep)]
        pos, quat, hand, _passive, ct = _replay_episode_grasp_qpos(entry, object_name=object_name, seed_base=seed_base)
        summaries.append((int(ep), ct, pos))

    if len(summaries) == 1:
        return summaries[0][0]

    # Prefer episode with most contacts; tie-break by lowest mean match cost to others.
    best_ep = summaries[0][0]
    best_score = float("inf")
    for i, (ep_i, ct_i, _) in enumerate(summaries):
        if ct_i.count == 0:
            continue
        costs: list[float] = []
        for j, (_, ct_j, _) in enumerate(summaries):
            if i == j or ct_j.count == 0:
                continue
            n = min(ct_i.count, ct_j.count)
            d = np.linalg.norm(ct_i.pos_obj[:n] - ct_j.pos_obj[:n], axis=1)
            costs.append(float(np.mean(d)))
        score = float(np.mean(costs)) if costs else 0.0
        score -= 0.001 * ct_i.count
        if score < best_score:
            best_score = score
            best_ep = ep_i
    return int(best_ep)


def _rep_from_lap_summary(sidecar_dir: Path, object_name: str) -> int | None:
    summary_path = Path(sidecar_dir) / "canonical_grasp_summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    obj = summary.get("objects", {}).get(object_name, {})
    rep = obj.get("report", {}).get("representative_episode_index")
    return int(rep) if rep is not None else None


def distill_qpos_from_sidecar_dir(
    sidecar_dir: Path,
    *,
    out_dir: Path | None = None,
    exclude_fallback: bool = False,
    filter_peg_off_table: bool = True,
    peg_grasp_max_z_delta_m: float = 0.008,
    seed_base: int = 0,
    representative_episode: dict[str, int | None] | None = None,
    unified_rep_episode: int | None = None,
) -> dict[str, QposGraspPrototype]:
    """Build canonical qpos+contact prototypes (one per object).

    If ``unified_rep_episode`` is set, both tray/peg use that episode for qpos/contacts.
    """
    sidecar_dir = Path(sidecar_dir)
    out_dir = out_dir if out_dir is not None else sidecar_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((sidecar_dir / "manifest.json").read_text(encoding="utf-8"))
    by_index = {int(e["episode_index"]): e for e in manifest["episodes"]}

    tray_used, _, _ = load_episode_snapshots(
        sidecar_dir,
        object_name="tray",
        exclude_fallback=exclude_fallback,
        filter_peg_off_table=False,
        seed_base=seed_base,
    )
    if unified_rep_episode is not None:
        # Fast path: skip per-episode MuJoCo replay for peg-on-table filter.
        peg_used = [
            int(e["episode_index"])
            for e in manifest["episodes"]
            if not _episode_excluded(e, object_name="peg", exclude_fallback=exclude_fallback)
        ]
    else:
        peg_used, _, _ = load_episode_snapshots(
            sidecar_dir,
            object_name="peg",
            exclude_fallback=exclude_fallback,
            filter_peg_off_table=filter_peg_off_table,
            peg_grasp_max_z_delta_m=peg_grasp_max_z_delta_m,
            seed_base=seed_base,
        )

    rep_ep = unified_rep_episode
    if rep_ep is None and representative_episode is not None:
        rep_ep = representative_episode.get("tray") or representative_episode.get("peg")
    if rep_ep is None:
        rep_ep = _rep_from_lap_summary(sidecar_dir, "tray") or _rep_from_lap_summary(sidecar_dir, "peg")
    if rep_ep is None:
        rep_ep = _pick_representative_episode(
            sidecar_dir, object_name="tray", used_indices=tray_used, seed_base=seed_base
        )
    rep_ep = int(rep_ep)
    entry = by_index[rep_ep]
    unified = _replay_unified_episode(entry, seed_base=seed_base)

    prototypes: dict[str, QposGraspPrototype] = {}
    for object_name, used in (("tray", tray_used), ("peg", peg_used)):
        proto = unified[object_name]
        proto = QposGraspPrototype(
            object_name=proto.object_name,
            hand_side=proto.hand_side,
            mocap_pos_obj=proto.mocap_pos_obj,
            mocap_quat_obj=proto.mocap_quat_obj,
            hand_joint=proto.hand_joint,
            passive_action23=proto.passive_action23,
            grasp_action23=proto.grasp_action23,
            squeeze_action23=proto.squeeze_action23,
            contact_targets=proto.contact_targets,
            representative_episode_index=rep_ep,
            source_episode_indices=list(used),
        )
        out_npz = out_dir / f"canonical_{object_name}_qpos_grasp.npz"
        proto.save_npz(out_npz)
        prototypes[object_name] = proto

    summary = {
        "sidecar_dir": str(sidecar_dir),
        "objects": {
            name: {
                "npz_path": str(out_dir / f"canonical_{name}_qpos_grasp.npz"),
                "representative_episode_index": p.representative_episode_index,
                "contact_count": p.contact_targets.count,
                "source_episode_indices": p.source_episode_indices,
            }
            for name, p in prototypes.items()
        },
    }
    (out_dir / "canonical_qpos_grasp_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return prototypes
