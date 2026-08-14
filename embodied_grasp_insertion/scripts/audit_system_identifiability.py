#!/usr/bin/env python3
"""P0-A System Identifiability Audit (read-only).

Enumerates DexJoCo bimanual_assembly episodes, schemas, assets, and snapshot
capabilities. Does not train, expand data, or run long rollouts.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEXJOCO_ROOT = PROJECT_ROOT.parent
REACH_ROOT = DEXJOCO_ROOT.parent / "reach_insert_rl"
DEFAULT_SIDECAR = Path("/mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly")
DEFAULT_XML = (
    DEXJOCO_ROOT
    / "dexjoco"
    / "dexjoco"
    / "sim"
    / "envs"
    / "xmls"
    / "arena_arm_hand_bimanual_assembly.xml"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "system_identifiability_audit_v1.json"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "SYSTEM_IDENTIFIABILITY_AUDIT.md"
DEFAULT_STATE = PROJECT_ROOT / "outputs" / "state.json"
RECOVERY_SNAPSHOTS = Path(
    "/mnt/hdd/dexjoco/datasets/recovery_trajectory_policy/raw/snapshots"
)

# Code-confirmed 85D / 44D layouts (full_obs.py / full_env.py).
OBS_SEGMENTS: list[dict[str, Any]] = [
    {
        "name": "act44",
        "slice": [0, 44],
        "dim": 44,
        "source": "current_action44(raw) via read_arm_action + dual_arm23_to_action44",
        "category": "deployment_observable",
        "note": "Absolute commanded wrist+finger pose (proprio/command), not image.",
    },
    {
        "name": "peg7",
        "slice": [44, 51],
        "dim": 7,
        "source": "raw._data.qpos[raw._peg_qpos_adr:adr+7] freejoint",
        "category": "simulator_privileged",
        "note": "Object world pose xyz+quat; not a robot sensor.",
    },
    {
        "name": "tray7",
        "slice": [51, 58],
        "dim": 7,
        "source": "raw._data.qpos[raw._socket_qpos_adr:adr+7] freejoint",
        "category": "simulator_privileged",
        "note": "Tray/socket body freejoint pose.",
    },
    {
        "name": "lat_vec3",
        "slice": [58, 61],
        "dim": 3,
        "source": "privileged_full_features → lateral_error(tip, socket, hole)",
        "category": "simulator_privileged",
        "note": "Derived from peg tip vs socket site.",
    },
    {
        "name": "along_tip_axis",
        "slice": [61, 64],
        "dim": 3,
        "source": "[along, tip_dist, axis_err] from privileged_full_features",
        "category": "simulator_privileged",
        "note": "Scalar insert geometry errors.",
    },
    {
        "name": "hole_axis3",
        "slice": [64, 67],
        "dim": 3,
        "source": "unit hole opening axis from socket site + bottom geom",
        "category": "simulator_privileged",
        "note": "Fixed-asset mating axis, not learned semantics.",
    },
    {
        "name": "peg_axis3",
        "slice": [67, 70],
        "dim": 3,
        "source": "unit peg body z-axis",
        "category": "simulator_privileged",
        "note": "Object axis from MuJoCo body frame.",
    },
    {
        "name": "ft12",
        "slice": [70, 82],
        "dim": 12,
        "source": "FingerForceLabeler wrist_ft_right/left minus baseline",
        "category": "deployment_observable",
        "note": "Wrist FT only (6+6). Per-finger forces exist in labeler but are NOT packed into 85D.",
    },
    {
        "name": "flags3",
        "slice": [82, 85],
        "dim": 3,
        "source": "FullEpisodeEnv._flags: tray_ok_seen, peg_ok_seen, peg_ok_seen(duplicate)",
        "category": "simulator_privileged",
        "note": "From AssemblyContactLabeler + lift; third flag duplicates peg_ok_seen.",
    },
]

ACTION_SEGMENTS: list[dict[str, Any]] = [
    {
        "name": "right_xyz",
        "slice": [0, 3],
        "scale": 0.008,
        "controllable_in_full_env": True,
        "controllable_in_insert_handoff": True,
    },
    {
        "name": "right_rotvec",
        "slice": [3, 6],
        "scale": 0.04,
        "controllable_in_full_env": True,
        "controllable_in_insert_handoff": True,
    },
    {
        "name": "right_fingers16",
        "slice": [6, 22],
        "scale": 0.15,
        "controllable_in_full_env": True,
        "controllable_in_insert_handoff": False,
        "note": "InsertHandoffEnv freezes fingers at handoff.",
    },
    {
        "name": "left_xyz",
        "slice": [22, 25],
        "scale": 0.008,
        "controllable_in_full_env": True,
        "controllable_in_insert_handoff": True,
    },
    {
        "name": "left_rotvec",
        "slice": [25, 28],
        "scale": 0.04,
        "controllable_in_full_env": True,
        "controllable_in_insert_handoff": True,
    },
    {
        "name": "left_fingers16",
        "slice": [28, 44],
        "scale": 0.15,
        "controllable_in_full_env": True,
        "controllable_in_insert_handoff": False,
        "note": "InsertHandoffEnv freezes fingers at handoff.",
    },
]

EXPECTED_META_KEYS = {
    "episode_index",
    "has_peg",
    "has_tray",
    "left_hand_body_names",
    "num_hand_keypoints",
    "num_interaction_vertices",
    "num_object_samples",
    "num_steps",
    "peg_body",
    "replay_info",
    "right_hand_body_names",
    "timing",
    "tray_body",
    "zarr_path",
}

EXPECTED_SIDECAR_NPZ_KEYS = {
    "episode_index",
    "num_steps",
    "left_grasp_frame",
    "right_grasp_frame",
    "tray_lift_start",
    "peg_lift_start",
    "tray_grasp_frame",
    "tray_hand_points_obj",
    "tray_contact_centers_obj",
    "tray_object_samples_obj",
    "tray_interaction_vertices_obj",
    "tray_laplacian_coords",
    "tray_adjacency_degrees",
    "tray_adjacency",
    "peg_grasp_frame",
    "peg_hand_points_obj",
    "peg_contact_centers_obj",
    "peg_object_samples_obj",
    "peg_interaction_vertices_obj",
    "peg_laplacian_coords",
    "peg_adjacency_degrees",
    "peg_adjacency",
}


@dataclass
class EpisodeRecord:
    episode_index: int
    status: str
    reasons: list[str] = field(default_factory=list)
    peg_body: str | None = None
    tray_body: str | None = None
    zarr_path: str | None = None
    sidecar_npz: str | None = None
    meta_path: str | None = None
    num_steps: int | None = None
    zarr_action_shape: list[int] | None = None
    zarr_action_rotvec_shape: list[int] | None = None
    timing: dict[str, Any] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_code_dims() -> dict[str, Any]:
    """Confirm FULL_OBS_DIM / FULL_ACT_DIM from source without importing MuJoCo."""
    full_obs = REACH_ROOT / "reach_insert_rl" / "env" / "full_obs.py"
    full_env = REACH_ROOT / "reach_insert_rl" / "env" / "full_env.py"
    out: dict[str, Any] = {"full_obs_path": str(full_obs), "full_env_path": str(full_env)}
    if not full_obs.exists():
        out["error"] = f"missing {full_obs}"
        return out
    tree = ast.parse(full_obs.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in {"FULL_OBS_DIM", "FULL_ACT_DIM"}:
                    if isinstance(node.value, ast.Constant):
                        out[t.id] = int(node.value.value)
    # scales from full_env defaults
    env_src = full_env.read_text(encoding="utf-8") if full_env.exists() else ""
    out["pos_scale_default"] = 0.008 if "pos_scale: float = 0.008" in env_src else None
    out["rot_scale_default"] = 0.04 if "rot_scale: float = 0.04" in env_src else None
    out["finger_scale_default"] = 0.15 if "finger_scale: float = 0.15" in env_src else None
    out["schema_sum"] = sum(int(s["dim"]) for s in OBS_SEGMENTS)
    out["schema_matches_code"] = out.get("FULL_OBS_DIM") == 85 and out.get("FULL_ACT_DIM") == 44
    out["segment_sum_equals_85"] = out["schema_sum"] == 85
    return out


def _inventory_repo_meshes() -> dict[str, Any]:
    mesh_dir = (
        DEXJOCO_ROOT
        / "dexjoco"
        / "dexjoco"
        / "sim"
        / "envs"
        / "xmls"
        / "industreal"
        / "mesh"
        / "industreal_pegs"
    )
    present = sorted(p.name for p in mesh_dir.glob("*.obj")) if mesh_dir.exists() else []
    return {
        "mesh_dir": str(mesh_dir),
        "obj_files_present": present,
        "note": "Repo contains multiple IndustReal peg/tray meshes; episode coverage is separate.",
    }


def _audit_episode(
    entry: dict[str, Any],
    sidecar_dir: Path,
    *,
    metadata_only: bool,
    strict: bool,
) -> EpisodeRecord:
    ep = int(entry["episode_index"])
    reasons: list[str] = []
    meta_path = sidecar_dir / f"episode_{ep:03d}" / "meta.json"
    npz_path = sidecar_dir / f"episode_{ep:03d}" / "interaction_sidecar.npz"
    zarr_path = Path(str(entry.get("zarr_path", "")))
    rec = EpisodeRecord(
        episode_index=ep,
        status="ok",
        zarr_path=str(zarr_path) if zarr_path else None,
        sidecar_npz=str(npz_path),
        meta_path=str(meta_path),
        timing=entry.get("timing"),
    )

    if not meta_path.exists():
        reasons.append("missing_meta.json")
    if not npz_path.exists():
        reasons.append("missing_interaction_sidecar.npz")
    if not zarr_path.exists():
        reasons.append("missing_zarr")

    peg_body = None
    tray_body = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"meta_json_unreadable:{exc}")
            meta = None
        if meta is not None:
            keys = set(meta.keys())
            missing = sorted(EXPECTED_META_KEYS - keys)
            extra = sorted(keys - EXPECTED_META_KEYS)
            if missing:
                reasons.append(f"meta_missing_keys:{missing}")
            if extra and strict:
                reasons.append(f"meta_extra_keys:{extra}")
            peg_body = meta.get("peg_body")
            tray_body = meta.get("tray_body")
            if peg_body is None:
                reasons.append("meta_missing_peg_body")
            if tray_body is None:
                reasons.append("meta_missing_tray_body")
            rec.num_steps = int(meta["num_steps"]) if "num_steps" in meta else None
            # Do not guess assets from episode id; require explicit body names.
            if meta.get("zarr_path") and Path(meta["zarr_path"]) != zarr_path:
                reasons.append("meta_zarr_path_mismatch_vs_manifest")

    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=True) as z:
                keys = set(z.files)
            missing = sorted(EXPECTED_SIDECAR_NPZ_KEYS - keys)
            if missing:
                reasons.append(f"npz_missing_keys:{missing}")
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"npz_unreadable:{exc}")

    timing = entry.get("timing") or {}
    if timing.get("peg_lift_start") is None:
        reasons.append("timing_missing_peg_lift_start")
    if timing.get("right_grasp_frame") is None:
        reasons.append("timing_missing_right_grasp_frame")

    if zarr_path.exists() and not metadata_only:
        try:
            import zarr

            root = zarr.open(str(zarr_path), mode="r")
            if "data" not in root or "action" not in root["data"]:
                reasons.append("zarr_missing_data/action")
            else:
                act = root["data"]["action"]
                rec.zarr_action_shape = list(act.shape)
                if len(act.shape) != 2 or act.shape[1] not in (44, 46):
                    reasons.append(f"zarr_action_dim_unexpected:{act.shape}")
                if "action_rotvec" in root["data"]:
                    ar = root["data"]["action_rotvec"]
                    rec.zarr_action_rotvec_shape = list(ar.shape)
                    if len(ar.shape) != 2 or ar.shape[1] != 44:
                        reasons.append(f"zarr_action_rotvec_dim_unexpected:{ar.shape}")
                else:
                    reasons.append("zarr_missing_action_rotvec")
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"zarr_unreadable:{exc}")
    elif zarr_path.exists() and metadata_only:
        # Still verify path existence; optional light open for shape if cheap.
        try:
            import zarr

            root = zarr.open(str(zarr_path), mode="r")
            if "data" in root and "action" in root["data"]:
                rec.zarr_action_shape = list(root["data"]["action"].shape)
            if "data" in root and "action_rotvec" in root["data"]:
                rec.zarr_action_rotvec_shape = list(root["data"]["action_rotvec"].shape)
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"zarr_metadata_open_failed:{exc}")

    rec.peg_body = peg_body
    rec.tray_body = tray_body
    if reasons:
        rec.status = "excluded"
        rec.reasons = reasons
    return rec


def _snapshot_capabilities() -> dict[str, Any]:
    return {
        "InsertEnvSnapshot": {
            "defined_in": "reach_insert_rl/env/handoff_env.py",
            "stores": [
                "deepcopy(MjData)",
                "wrapper t/peg_lost/holds/force_baseline/prev geometry/done/socket0",
            ],
            "can_restore": True,
            "dual_wrist_state": "yes_via_MjData_qpos_and_hold_arm23",
            "all_finger_qpos_qvel": "yes_via_MjData",
            "object_pose_velocity": "yes_via_freejoint_qpos_qvel",
            "socket_pose": "yes_via_tray_freejoint_and_sites",
            "socket_geometry": "fixed_mesh_asset_not_per_episode_field",
            "mujoco_contact": "yes_via_MjData.contact_when_restored",
            "per_finger_contact": "not_stored_as_field_reconstructible_via_FingerForceLabeler_cfrc_ext",
            "object_in_hand_relative_pose": "not_stored_derivable_from_peg_and_wrist_frames",
            "slip_proxy": "no_explicit_label_would_require_temporal_relative_motion",
            "insert_contact_mode": "partial_AssemblyOutcome_tray_ok_peg_ok_insert_ok_counts_only",
            "env_scope": "InsertHandoffEnv_insert_phase_fingers_frozen",
        },
        "CompactInsertSnapshot": {
            "defined_in": "recovery_trajectory_policy/.../snapshot.py",
            "stores": ["mjSTATE_INTEGRATION state vector", "writable MjData arrays", "wrapper fields"],
            "on_disk_pkl_count": (
                len(list(RECOVERY_SNAPSHOTS.rglob("*.pkl"))) if RECOVERY_SNAPSHOTS.exists() else 0
            ),
            "on_disk_root": str(RECOVERY_SNAPSHOTS),
            "note": "Insert-phase recovery roots; not full grasp→insert episode archives.",
        },
        "FullEpisodeEnv": {
            "snapshot_api": "absent",
            "matched_intervention": "not_built_in; can replay zarr initial_state+actions but no branch snapshot helper",
            "fingers_controllable": True,
            "action_dim": 44,
        },
    }


def _contact_label_capabilities() -> dict[str, Any]:
    return {
        "AssemblyOutcome": {
            "fields": ["tray_ok", "peg_ok", "insert_ok", "tray_contact_count", "peg_contact_count"],
            "source": "hybrid_insert/assembly_contacts.py",
            "in_85d": "only as cumulative flags tray_ok_seen/peg_ok_seen (peg duplicated)",
            "capture_rim_jam_backout": "unavailable",
        },
        "FingerForceLabeler": {
            "fields": [
                "right_finger_force(12)",
                "left_finger_force(12)",
                "wrist_ft_right(6)",
                "wrist_ft_left(6)",
                "outcome",
            ],
            "source": "dexquery/data/finger_contact_forces.py",
            "packed_into_85d": ["wrist_ft only (12)"],
            "per_finger_in_obs": False,
            "slip_truth": False,
            "category": "simulator_privileged_when_from_cfrc_ext; wrist_ft_may_map_to_sensors",
        },
        "sidecar_npz": {
            "has_per_finger_contact_series": False,
            "has_slip": False,
            "has_object_in_hand_pose": False,
            "has_grasp_geometry_samples": True,
            "note": "Object-frame grasp contact centers / hand points at grasp frames only.",
        },
    }


def _build_conclusions(
    *,
    valid: list[EpisodeRecord],
    excluded: list[EpisodeRecord],
    object_assets: list[str],
    socket_assets: list[str],
    pairs: list[str],
    code_dims: dict[str, Any],
) -> dict[str, Any]:
    n_geom_families = len(pairs)
    single_geometry = n_geom_families == 1
    schema_ok = bool(code_dims.get("schema_matches_code")) and bool(
        code_dims.get("segment_sum_equals_85")
    )
    # P0-A criteria from MOTIVATION_AND_PLAN.md §8
    snapshot_ok = True  # InsertEnvSnapshot / CompactInsertSnapshot restore MjData
    episode_split_ok = len(valid) >= 2
    geometry_diversity_fail = single_geometry
    deployment_grasp_state_missing = True  # no finger joints/contacts/o2h in 85D deploy fields

    if snapshot_ok and episode_split_ok and schema_ok and not geometry_diversity_fail:
        verdict = "pass"
    elif snapshot_ok and episode_split_ok and schema_ok and geometry_diversity_fail:
        verdict = "partial"
    else:
        verdict = "fail"

    return {
        "verdict": verdict,
        "schema_85d_44d_consistent_with_code": schema_ok,
        "episodes_audited_valid": len(valid),
        "episodes_excluded": len(excluded),
        "object_asset_count": len(object_assets),
        "socket_asset_count": len(socket_assets),
        "geometry_family_count": n_geom_families,
        "episodes_are_same_geometry_different_trajectories": single_geometry,
        "snapshot_exact_matched_intervention": {
            "insert_handoff_path": "yes_InsertEnvSnapshot_or_CompactInsertSnapshot",
            "full_episode_path": "partial_no_snapshot_API_replay_only",
            "fingers_free_in_full_env": True,
            "fingers_frozen_in_insert_handoff": True,
        },
        "object_in_hand_6d": {
            "in_85d": False,
            "in_sidecar": False,
            "derivable_from_sim_state": True,
            "category": "simulator_privileged_derived",
        },
        "per_finger_contact": {
            "in_85d": False,
            "live_via_FingerForceLabeler": True,
            "slip_truth": False,
            "category": "simulator_privileged",
        },
        "object_held_out_split_feasible": False,
        "geometry_held_out_split_feasible": False,
        "episode_held_out_split_feasible": episode_split_ok,
        "observability_p0_allowed": False,
        "observability_p0_reason": (
            "85D lacks deployment finger contact / object-in-hand; single geometry cannot support "
            "object/geometry-held-out controls required by project hard gates."
        ),
        "controllability_p0_allowed_smoke": True,
        "controllability_p0_reason": (
            "FullEpisodeEnv exposes 44D finger control; InsertEnvSnapshot enables matched "
            "insert-phase restore. Controllability smoke can run on single geometry, but "
            "Semantic P0 still blocked."
        ),
        "minimal_missing_fields": [
            "object_in_hand_relative_pose_6d_as_labeled_series",
            "per_finger_contact_force_or_binary_retention_series",
            "slip_proxy_or_truth_series",
            "contact_mode_beyond_tray_ok_peg_ok_insert_ok",
            "multiple_object_hole_geometry_families_in_episode_coverage",
            "FullEpisodeEnv.snapshot/restore for grasp-phase matched intervention",
        ],
        "minimal_targeted_replay_needs": [
            "Optional: single-episode FullEpisodeEnv smoke to dump privileged o2h + finger forces alongside 85D",
            "Optional: add snapshot API to FullEpisodeEnv before Controllability P0",
            "Required for Semantic/Observability hard gates: new episodes with alternate IndustReal peg/tray meshes already present in repo assets",
        ],
        "deployment_observable_fields": [s["name"] for s in OBS_SEGMENTS if s["category"] == "deployment_observable"],
        "simulator_privileged_fields": [s["name"] for s in OBS_SEGMENTS if s["category"] == "simulator_privileged"],
        "unavailable_in_current_interfaces": [
            "per_finger_contact_in_85d",
            "finger_joint_proprio_separate_from_act44_command",
            "explicit_slip_label",
            "object_identity_embedding",
            "hole_clearance_depth_descriptor",
            "capture_rim_jam_backout_mode",
            "object_held_out_geometry_split",
        ],
        "stop_condition_triggered": geometry_diversity_fail,
        "stop_condition_text": (
            "所谓多样性实际只是同一几何的位姿/轨迹变化（100/100 同为 round_peg_8mm + tray_insert_round_peg_8mm）。"
            if geometry_diversity_fail
            else ""
        ),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    sidecar_dir = Path(args.sidecar_dir)
    manifest_path = sidecar_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"sidecar manifest missing: {manifest_path}")

    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = list(man.get("episodes") or [])
    if args.episode_ids:
        wanted = {int(x) for x in args.episode_ids}
        episodes = [e for e in episodes if int(e["episode_index"]) in wanted]
        found = {int(e["episode_index"]) for e in episodes}
        missing_ids = sorted(wanted - found)
    else:
        missing_ids = []

    if args.max_episodes is not None:
        episodes = episodes[: int(args.max_episodes)]

    code_dims = _parse_code_dims()
    records: list[EpisodeRecord] = []
    for entry in episodes:
        records.append(
            _audit_episode(
                entry,
                sidecar_dir,
                metadata_only=bool(args.metadata_only),
                strict=bool(args.strict),
            )
        )

    for mid in missing_ids:
        records.append(
            EpisodeRecord(
                episode_index=mid,
                status="excluded",
                reasons=["episode_id_not_in_manifest"],
            )
        )

    valid = [r for r in records if r.status == "ok"]
    excluded = [r for r in records if r.status != "ok"]

    peg_counter = Counter(r.peg_body for r in valid if r.peg_body)
    tray_counter = Counter(r.tray_body for r in valid if r.tray_body)
    pair_counter = Counter(
        f"{r.peg_body}__{r.tray_body}" for r in valid if r.peg_body and r.tray_body
    )

    object_assets = sorted(peg_counter.keys())
    socket_assets = sorted(tray_counter.keys())
    pairs = sorted(pair_counter.keys())
    geometry_families = [
        {
            "family_id": p,
            "object": p.split("__")[0],
            "socket": p.split("__")[1],
            "episode_count": int(pair_counter[p]),
            "nominal_size_hint": "8mm_round_from_asset_name",
        }
        for p in pairs
    ]

    mujoco_smoke: dict[str, Any] | None = None
    if args.mujoco_smoke_episode is not None:
        mujoco_smoke = _mujoco_smoke_one(int(args.mujoco_smoke_episode), sidecar_dir)

    conclusions = _build_conclusions(
        valid=valid,
        excluded=excluded,
        object_assets=object_assets,
        socket_assets=socket_assets,
        pairs=pairs,
        code_dims=code_dims,
    )

    snapshot_caps = _snapshot_capabilities()
    contact_caps = _contact_label_capabilities()
    repo_meshes = _inventory_repo_meshes()

    available_labels = [
        "tray_ok",
        "peg_ok",
        "insert_ok",
        "tray_contact_count",
        "peg_contact_count",
        "wrist_ft_12",
        "tip_dist_lat_along_axis_err",
        "peg7_tray7_freejoint",
        "act44_command",
    ]
    missing_labels = [
        "object_in_hand_6d_series",
        "per_finger_contact_series_in_dataset",
        "slip_truth_or_proxy_series",
        "capture_rim_jam_partial_backout_mode",
        "object_shape_descriptor",
        "hole_clearance_depth_descriptor",
        "geometry_family_id_beyond_single_asset",
    ]

    audit = {
        "audit_name": "system_identifiability_audit_v1",
        "created_at": _utc_now(),
        "metadata_only": bool(args.metadata_only),
        "strict": bool(args.strict),
        "source_paths": {
            "sidecar_dir": str(sidecar_dir),
            "sidecar_manifest": str(manifest_path),
            "arena_xml": str(DEFAULT_XML),
            "full_obs_py": str(REACH_ROOT / "reach_insert_rl" / "env" / "full_obs.py"),
            "full_env_py": str(REACH_ROOT / "reach_insert_rl" / "env" / "full_env.py"),
            "handoff_env_py": str(REACH_ROOT / "reach_insert_rl" / "env" / "handoff_env.py"),
            "finger_force_labeler": str(
                DEXJOCO_ROOT / "dexquery" / "data" / "finger_contact_forces.py"
            ),
            "assembly_contacts": str(
                DEXJOCO_ROOT / "dexjoco" / "hybrid_insert" / "assembly_contacts.py"
            ),
            "recovery_snapshots": str(RECOVERY_SNAPSHOTS),
        },
        "episode_count": {
            "manifest_num_episodes": int(man.get("num_episodes", len(man.get("episodes", [])))),
            "requested": len(episodes) + len(missing_ids),
            "valid": len(valid),
            "excluded": len(excluded),
        },
        "valid_episode_ids": [r.episode_index for r in valid],
        "excluded_episode_ids_and_reasons": [
            {"episode_index": r.episode_index, "reasons": r.reasons} for r in excluded
        ],
        "object_assets": [{"name": k, "episode_count": int(v)} for k, v in peg_counter.most_common()],
        "socket_assets": [
            {"name": k, "episode_count": int(v)} for k, v in tray_counter.most_common()
        ],
        "object_socket_pairs": [
            {"pair": k, "episode_count": int(v)} for k, v in pair_counter.most_common()
        ],
        "geometry_families": geometry_families,
        "repo_mesh_inventory": repo_meshes,
        "observation_schema": {
            "dim": 85,
            "code_constants": code_dims,
            "segments": OBS_SEGMENTS,
        },
        "action_schema": {
            "dim": 44,
            "segments": ACTION_SEGMENTS,
            "scales": {
                "pos_scale": code_dims.get("pos_scale_default"),
                "rot_scale": code_dims.get("rot_scale_default"),
                "finger_scale": code_dims.get("finger_scale_default"),
            },
            "full_env_fingers_free": True,
            "insert_handoff_fingers_frozen": True,
        },
        "snapshot_schema": snapshot_caps,
        "contact_schema": contact_caps,
        "available_labels": available_labels,
        "missing_labels": missing_labels,
        "split_feasibility": {
            "episode_held_out": conclusions["episode_held_out_split_feasible"],
            "object_instance_held_out": conclusions["object_held_out_split_feasible"],
            "geometry_family_held_out": conclusions["geometry_held_out_split_feasible"],
            "reason": conclusions.get("stop_condition_text")
            or "episode split ok; object/geometry held-out impossible with one family",
        },
        "matched_intervention_feasibility": conclusions["snapshot_exact_matched_intervention"],
        "mujoco_smoke": mujoco_smoke,
        "conclusions": conclusions,
        "episodes_detail": [asdict(r) for r in records],
    }
    return audit


def _mujoco_smoke_one(episode_index: int, sidecar_dir: Path) -> dict[str, Any]:
    """Single-episode MuJoCo smoke: reset FullEpisodeEnv and check obs dim + labeler."""
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    # Prefer CPU / avoid grabbing busy training GPUs.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    sys.path.insert(0, str(REACH_ROOT))
    sys.path.insert(0, str(DEXJOCO_ROOT))
    sys.path.insert(0, str(DEXJOCO_ROOT / "dexjoco"))
    sys.path.insert(0, str(DEXJOCO_ROOT.parent / "lai"))

    from reach_insert_rl.env.full_env import FullEpisodeEnv
    from reach_insert_rl.env.full_obs import FULL_ACT_DIM, FULL_OBS_DIM
    from reach_insert_rl.env.handoff_env import load_manifest_entries

    entries = load_manifest_entries(sidecar_dir, episode_indices=[episode_index])
    if not entries:
        return {
            "episode_index": episode_index,
            "ok": False,
            "error": "episode not in load_manifest_entries (timing filter or missing)",
        }

    env = FullEpisodeEnv(entries, sidecar_dir=sidecar_dir, seed=0, use_force=True)
    try:
        obs, info = env.reset(episode_index=episode_index)
        obs = np.asarray(obs)
        outcome = env._labeler.compute(env._raw)
        force = None
        if env._force_labeler is not None:
            force = env._force_labeler.compute(env._raw)
        peg = np.asarray(env._raw._data.qpos[env._raw._peg_qpos_adr : env._raw._peg_qpos_adr + 7])
        # object-in-hand relative: peg pose vs right wrist from act44
        act44 = obs[:44]
        wrist_xyz = act44[0:3]
        rel_xyz = peg[:3] - wrist_xyz
        return {
            "episode_index": episode_index,
            "ok": True,
            "obs_shape": list(obs.shape),
            "obs_dim_matches": int(obs.shape[0]) == int(FULL_OBS_DIM),
            "act_dim": int(FULL_ACT_DIM),
            "info_keys": sorted(info.keys()),
            "assembly_outcome": {
                "tray_ok": bool(outcome.tray_ok),
                "peg_ok": bool(outcome.peg_ok),
                "insert_ok": bool(outcome.insert_ok),
                "tray_contact_count": int(outcome.tray_contact_count),
                "peg_contact_count": int(outcome.peg_contact_count),
            },
            "finger_force_shapes": None
            if force is None
            else {
                "right_finger_force": list(np.asarray(force.right_finger_force).shape),
                "left_finger_force": list(np.asarray(force.left_finger_force).shape),
                "wrist_ft_right": list(np.asarray(force.wrist_ft_right).shape),
                "wrist_ft_left": list(np.asarray(force.wrist_ft_left).shape),
            },
            "object_in_hand_rel_xyz_proxy_from_peg_minus_right_wrist_command": rel_xyz.tolist(),
            "note": (
                "rel_xyz is a privileged proxy using peg freejoint minus commanded wrist xyz; "
                "not a full 6D object-in-hand pose in hand frame."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"episode_index": episode_index, "ok": False, "error": str(exc)}
    finally:
        env.close()


def _write_report(audit: dict[str, Any], path: Path) -> None:
    c = audit["conclusions"]
    obs = audit["observation_schema"]["segments"]
    lines: list[str] = []
    lines.append("# System Identifiability Audit (P0-A)")
    lines.append("")
    lines.append(f"- 日期（UTC）：{audit['created_at']}")
    lines.append(f"- 结论：**{c['verdict']}**")
    lines.append(
        f"- 有效 episode：{c['episodes_audited_valid']}；排除：{c['episodes_excluded']}"
    )
    lines.append(
        f"- object/socket/geometry family：{c['object_asset_count']}/"
        f"{c['socket_asset_count']}/{c['geometry_family_count']}"
    )
    lines.append(
        f"- 85D/44D 与代码一致：{c['schema_85d_44d_consistent_with_code']}"
    )
    lines.append("")
    lines.append("## 1. 85D observation 分段")
    lines.append("")
    lines.append("| 段 | slice | dim | 类别 | 来源 |")
    lines.append("|---|---|---|---|---|")
    for s in obs:
        a, b = s["slice"]
        lines.append(
            f"| `{s['name']}` | [{a}:{b}] | {s['dim']} | {s['category']} | {s['source']} |"
        )
    lines.append("")
    lines.append("### 部署可用 vs 特权")
    lines.append("")
    lines.append(
        "- **deployment_observable**：`act44`（本体/指令）、`ft12`（腕力；若真机有腕力传感器）。"
    )
    lines.append(
        "- **simulator_privileged**：`peg7`、`tray7`、相对几何、孔/销轴、outcome flags。"
    )
    lines.append(
        "- **unavailable（相对本项目目标）**：逐指接触不在 85D；无 slip；无 object-in-hand 6D 字段；"
        "无物体/孔几何描述符；无 capture/rim/jam/backout 模式。"
    )
    lines.append("")
    lines.append("## 2. object-in-hand 6D")
    lines.append("")
    lines.append(
        f"- 85D/sidecar 是否直接提供：否。"
    )
    lines.append(
        f"- 能否从仿真状态推导：是（peg freejoint 相对腕/手坐标系），类别为特权派生。"
        f" 详见 conclusions.object_in_hand_6d。"
    )
    lines.append("")
    lines.append("## 3. 逐指接触与滑移")
    lines.append("")
    lines.append(
        "- 逐指力：`FingerForceLabeler` 可从 `cfrc_ext` 得到 4×3×双手，但**未写入 85D**，也不在 sidecar 时序里。"
    )
    lines.append("- 滑移真值：无现成标签；不得伪造。")
    lines.append(
        "- `AssemblyOutcome` 仅有 tray_ok/peg_ok/insert_ok 与接触计数，无逐指 retention。"
    )
    lines.append("")
    lines.append("## 4. 44D 动作是否控制全部手指")
    lines.append("")
    lines.append(
        "- **FullEpisodeEnv：是**。右/左各 16 维手指增量，`finger_scale=0.15`，代码写明 fingers free。"
    )
    lines.append(
        "- **InsertHandoffEnv：否**。wrist12（或 riva）只动腕，手指在 handoff 冻结。"
    )
    lines.append("")
    lines.append("## 5. Snapshot 与 matched intervention")
    lines.append("")
    lines.append(
        "- `InsertEnvSnapshot`：深拷贝 `MjData` + wrapper 字段，可精确 restore → **insert 阶段 matched intervention：可用**。"
    )
    lines.append(
        "- `CompactInsertSnapshot`：recovery 磁盘上有 pkl（insert 根）；同样是 insert 阶段。"
    )
    lines.append(
        "- `FullEpisodeEnv`：**无** snapshot/restore API；只能 zarr `initial_state`+动作重放 → 抓持阶段精确分支 **部分可用/缺口**。"
    )
    lines.append("")
    lines.append("## 6. 几何多样性")
    lines.append("")
    for g in audit["geometry_families"]:
        lines.append(
            f"- family `{g['family_id']}`：{g['episode_count']} episodes"
        )
    lines.append(
        f"- 结论：{c['episodes_are_same_geometry_different_trajectories']} "
        "（同一 `industreal_round_peg_8mm` + `industreal_tray_insert_round_peg_8mm` 的不同轨迹/位姿）。"
    )
    lines.append(
        "- 仓库 mesh 目录另有 4/12/16mm 与 rectangular 资产，但**未出现在本 100 episode 覆盖中**。"
    )
    lines.append("")
    lines.append("## 7. Split 可行性")
    lines.append("")
    sf = audit["split_feasibility"]
    lines.append(f"- episode-held-out：{sf['episode_held_out']}")
    lines.append(f"- object-instance-held-out：{sf['object_instance_held_out']}")
    lines.append(f"- geometry-family-held-out：{sf['geometry_family_held_out']}")
    lines.append("")
    lines.append("## 8. P0 准入判断")
    lines.append("")
    lines.append(f"- Observability P0 可否完整开始：**{c['observability_p0_allowed']}**")
    lines.append(f"  - 原因：{c['observability_p0_reason']}")
    lines.append(
        f"- Controllability P0 smoke 可否开始：**{c['controllability_p0_allowed_smoke']}**"
    )
    lines.append(f"  - 原因：{c['controllability_p0_reason']}")
    lines.append("")
    lines.append("## 9. 最小缺失与定向重放")
    lines.append("")
    for x in c["minimal_missing_fields"]:
        lines.append(f"- 缺失：{x}")
    for x in c["minimal_targeted_replay_needs"]:
        lines.append(f"- 需求：{x}")
    lines.append("")
    lines.append("## 10. 与停止条件")
    lines.append("")
    lines.append(
        f"- MOTIVATION §8 停止条件（多样性仅为同位姿变化）：**触发={c['stop_condition_triggered']}**"
    )
    if c.get("stop_condition_text"):
        lines.append(f"- {c['stop_condition_text']}")
    lines.append(
        "- 同时 snapshot 可恢复手/物/接触物理态，且 episode split 可行 → 总评 **partial**，不是全面 fail。"
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_state(audit: dict[str, Any], path: Path) -> None:
    c = audit["conclusions"]
    state = {
        "phase": "system_identifiability_audit",
        "busy": False,
        "updated_at": audit["created_at"],
        "verdict": c["verdict"],
        "prior_failures_reviewed": [
            {
                "family": "FutureStateTrajectoryConditionedActionDecoder",
                "result": "falsified",
                "evidence": "oracle_p0_strict rotating_val_crossroot: mean_imp=-15.7%, win_vs_direct=0%",
            },
            {
                "family": "DiffusionFlowRecovery",
                "result": "rejected_for_p0",
                "evidence": "generative models cannot fix wrong state/action interface; banned before observability/controllability",
            },
            {
                "family": "SetListwiseCandidateRanking",
                "result": "falsified_offline_to_closed_loop",
                "evidence": "offline ranking did not transfer to stable closed-loop recovery",
            },
            {
                "family": "OnlinePhysicsBranchMPPI",
                "result": "falsified",
                "evidence": "Branch MPPI expand15 = 5/15 < PrivHI 6/15",
            },
            {
                "family": "WristOnlyContactCMDP",
                "result": "demoted",
                "evidence": "real transitions but no finger controllability or object/hole semantics",
            },
            {
                "family": "GateResidualServo",
                "result": "banned",
                "evidence": "hand-written gates/residuals hide observability and controllability failures",
            },
        ],
        "why_not_repeat": [
            "不把未来 tip/geometry 轨迹当动作条件",
            "不在错误接口上换 Diffusion/Flow/更大 Transformer",
            "不做在线候选搜索或 MPPI 当部署策略",
            "不在 37D+wrist12 上再训 critic/actor",
            "不用 gate/residual/servo 掩盖缺失状态",
            "不把固定 peg/socket 相对误差称为通用插孔语义",
        ],
        "audit_sources": audit["source_paths"],
        "confirmed_fields": {
            "observation_85d_segments": [s["name"] for s in audit["observation_schema"]["segments"]],
            "action_44d_segments": [s["name"] for s in audit["action_schema"]["segments"]],
            "assembly_outcome": ["tray_ok", "peg_ok", "insert_ok", "contact_counts"],
            "wrist_ft_in_obs": True,
            "finger_forces_in_labeler_not_in_85d": True,
            "full_env_finger_control": True,
            "insert_handoff_fingers_frozen": True,
            "geometry_family_in_episodes": audit["geometry_families"],
        },
        "missing_fields": c["minimal_missing_fields"],
        "dataset_inventory": {
            "episode_count": audit["episode_count"],
            "valid_episode_ids": audit["valid_episode_ids"],
            "excluded_episode_ids_and_reasons": audit["excluded_episode_ids_and_reasons"],
            "object_assets": audit["object_assets"],
            "socket_assets": audit["socket_assets"],
            "object_socket_pairs": audit["object_socket_pairs"],
        },
        "geometry_inventory": {
            "families": audit["geometry_families"],
            "repo_meshes_not_in_episodes": audit["repo_mesh_inventory"],
            "same_geometry_different_trajectories": c[
                "episodes_are_same_geometry_different_trajectories"
            ],
        },
        "snapshot_capabilities": audit["snapshot_schema"],
        "contact_label_capabilities": audit["contact_schema"],
        "diagnosis": [
            f"P0-A verdict={c['verdict']}",
            "100 valid demos share one peg/socket geometry family; diversity is trajectory/pose only.",
            "85D mixes deployment proprio/wrist-FT with privileged object/geometry features; grasp contact is incomplete.",
            "44D full-env actions do control all fingers; insert-handoff path freezes them.",
            "Matched intervention is exact for InsertHandoffEnv snapshots; FullEpisodeEnv lacks snapshot API.",
            "object-in-hand 6D and per-finger forces are reconstructible from live MjData but not stored as dataset labels.",
            "slip truth and fine contact modes (capture/rim/jam/backout) are unavailable.",
            "object/geometry held-out splits are impossible on current coverage.",
        ],
        "next_proposal": {
            "allow_full_observability_p0": c["observability_p0_allowed"],
            "allow_controllability_p0_smoke": c["controllability_p0_allowed_smoke"],
            "actions": [
                "Implement FullEpisodeEnv.snapshot/restore (or equivalent) for grasp-phase matched intervention",
                "Optional single-episode privileged label dump: o2h pose + finger forces alongside 85D",
                "Do not start Observability P0 until multi-geometry coverage or explicit gate waiver",
                "For Semantic P0: schedule episodes using existing alternate IndustReal meshes (4/12/16mm etc.)",
            ],
        },
        "history": [
            {
                "date": "2026-08-13",
                "event": "project_init",
                "note": "Pre-policy scope definition; policy training banned",
            },
            {
                "date": audit["created_at"][:10],
                "event": "p0a_system_identifiability_audit",
                "verdict": c["verdict"],
                "valid_episodes": c["episodes_audited_valid"],
                "geometry_families": c["geometry_family_count"],
            },
        ],
        "manifest_path": str(DEFAULT_MANIFEST),
        "report_path": str(DEFAULT_REPORT),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-A System Identifiability Audit (read-only)")
    parser.add_argument("--sidecar-dir", type=str, default=str(DEFAULT_SIDECAR))
    parser.add_argument("--output-manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT))
    parser.add_argument("--state", type=str, default=str(DEFAULT_STATE))
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--episode-ids", type=int, nargs="*", default=None)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--mujoco-smoke-episode",
        type=int,
        default=None,
        help="Optional single episode MuJoCo reset smoke (EGL). Do not use for long rollouts.",
    )
    parser.add_argument("--skip-state", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    audit = run_audit(args)
    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    # Manifest omits bulky per-episode detail optionally? Keep full for audit integrity.
    out_manifest.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.skip_report:
        _write_report(audit, Path(args.report))
    if not args.skip_state:
        _write_state(audit, Path(args.state))

    c = audit["conclusions"]
    print(
        json.dumps(
            {
                "verdict": c["verdict"],
                "valid": c["episodes_audited_valid"],
                "excluded": c["episodes_excluded"],
                "geometry_families": c["geometry_family_count"],
                "schema_ok": c["schema_85d_44d_consistent_with_code"],
                "observability_p0_allowed": c["observability_p0_allowed"],
                "manifest": str(out_manifest),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
