#!/usr/bin/env python3
"""P0-L0 Label Derivability Audit — readonly smoke (no collect/train/write)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT.parent), str(PROJECT_ROOT.parent / "dexjoco")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.geometry.target_hole import (  # noqa: E402
    semantic_target_features,
    target_hole_from_raw,
)
from embodied_grasp_insertion.physics.grasp_metrics import (  # noqa: E402
    compute_step_metrics,
    control_dt_seconds,
    object_in_hand_pose,
    peg_hand_contact_counts,
    relative_pose_error,
)
from embodied_grasp_insertion.pilot import WRITE_IMPLEMENTATION_ENABLED  # noqa: E402
from embodied_grasp_insertion.simulation.full_episode_snapshot import (  # noqa: E402
    FullEpisodeSnapshot,
)
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    make_full_env,
    replay_demo_to_frame,
    select_roots_for_episode,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _item(
    name: str,
    verdict: str,
    *,
    sources: list[str],
    formula: str,
    units_frame: str,
    deployment_or_privilege: str,
    restore_ok: bool | None,
    ambiguities: list[str],
    notes: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "verdict": verdict,
        "sources": sources,
        "formula": formula,
        "units_frame": units_frame,
        "deployment_or_privilege": deployment_or_privilege,
        "snapshot_restore_consistent": restore_ok,
        "ambiguities": ambiguities,
        "notes": notes,
    }


def static_items() -> list[dict[str, Any]]:
    """Code-derived judgments (always available without MuJoCo episode)."""
    return [
        _item(
            "object_in_hand_pose_6d",
            "derivable",
            sources=["MjData.xpos/xmat peg_body", "allegro_palm_right", "grasp_metrics.object_in_hand_pose"],
            formula="t_rel = R_palm^{-1}(p_peg-p_palm); rotvec(R_palm^{-1} R_peg)",
            units_frame="m, rad; palm body frame",
            deployment_or_privilege="privilege_label_only",
            restore_ok=None,
            ambiguities=["fixed right palm reference", "body origin != tip"],
        ),
        _item(
            "object_in_hand_velocity",
            "partial",
            sources=["finite difference of o2h", "optional MjData.cvel (not wrapped)"],
            formula="||Δt_rel||/dt_ctrl ; angular rate from Δrotvec/dt",
            units_frame="m/s, rad/s; requires >=2 frames",
            deployment_or_privilege="privilege_label_only",
            restore_ok=None,
            ambiguities=["finite-diff vs spatial velocity", "single-frame blocked"],
        ),
        _item(
            "per_finger_contact_retention",
            "partial",
            sources=["data.contact peg-hand counts", "FingerForceLabeler cfrc_ext", "retention vs root"],
            formula="min(c_t/c_root,1); force_active = ||f_i||>=eps",
            units_frame="unitless; force eps default 0.05",
            deployment_or_privilege="privilege_label_only",
            restore_ok=None,
            ambiguities=["count vs force disagree", "legacy retention vs first step"],
        ),
        _item(
            "slip_proxy_or_truth",
            "partial",
            sources=["o2h finite difference proxies in grasp_metrics"],
            formula="||Δ translation||/dt; drift rate vs root",
            units_frame="m/s proxy; NOT contact tangential truth",
            deployment_or_privilege="privilege_proxy_only",
            restore_ok=None,
            ambiguities=["explicitly not ground truth", "slip truth blocked"],
            notes="truth=blocked; proxy=derivable",
        ),
        _item(
            "contact_mode",
            "partial",
            sources=["AssemblyContactLabeler tray_ok/peg_ok/insert_ok", "contact counts"],
            formula="boolean outcome flags only",
            units_frame="boolean / counts",
            deployment_or_privilege="privilege_label_only",
            restore_ok=None,
            ambiguities=["capture/rim/jam/backout undefined"],
            notes="fine modes blocked until contract",
        ),
        _item(
            "regrasp_needed_peg_loss_risk",
            "partial",
            sources=["FullEpisodeEnv._peg_lost", "peg_ok", "drift/drop thresholds", "legacy peg_loss alias"],
            formula="heuristics; legacy peg_loss := not terminal peg_ok",
            units_frame="boolean / counters",
            deployment_or_privilege="privilege_heuristic",
            restore_ok=None,
            ambiguities=["legacy peg_loss != physical drop", "no horizon risk contract"],
        ),
        _item(
            "object_target_mating_transform",
            "partial",
            sources=["_insert_geometry tip/socket/hole_axis", "semantic_target_features", "body poses"],
            formula="tip-socket vector + axes; full SE(3) mating API not unified",
            units_frame="m; world / site frames",
            deployment_or_privilege="privilege_label_only",
            restore_ok=None,
            ambiguities=["85D tip/axis also privilege", "no single T_mating export"],
        ),
        _item(
            "provenance",
            "partial",
            sources=[
                "FullEpisodeSnapshot.episode_index/zarr_path/t",
                "names_from_raw family",
                "target_hole instance/site",
            ],
            formula="root_id convention episode:frame:phase (not stored in snapshot)",
            units_frame="ids / paths",
            deployment_or_privilege="metadata",
            restore_ok=None,
            ambiguities=["root_id not a snapshot field", "on-disk snapshot schema unset"],
            notes="episode/family/instance/core snapshot fields derivable",
        ),
    ]


def run_mujoco_smoke(episode_id: int, frame: int | None) -> dict[str, Any]:
    env = make_full_env([episode_id], seed=0)
    try:
        obs = env.reset()
        assert obs is not None
        if frame is not None:
            target_frame = int(frame)
        else:
            # select_roots advances env; then reset and seek to chosen root.
            roots = select_roots_for_episode(env)
            if not roots:
                raise RuntimeError("no roots selected")
            target_frame = int(roots[0].frame)
            env.reset()
        replay_demo_to_frame(env, target_frame)
        snap = FullEpisodeSnapshot.capture(env)
        raw = env._raw

        o2h_a = object_in_hand_pose(raw)
        contact_a = peg_hand_contact_counts(raw)
        metrics_a = compute_step_metrics(env)
        hole_a = target_hole_from_raw(raw).to_dict()
        feats_a = semantic_target_features(raw)

        # Restore and recompute.
        env2_obs = snap.restore(env)
        assert env2_obs is not None
        o2h_b = object_in_hand_pose(raw)
        contact_b = peg_hand_contact_counts(raw)
        metrics_b = compute_step_metrics(env)
        hole_b = target_hole_from_raw(raw).to_dict()

        dt_t, dt_r = relative_pose_error(o2h_a, o2h_b)
        restore_pose_ok = bool(dt_t < 1e-12 and dt_r < 1e-12)
        restore_contact_ok = contact_a.total == contact_b.total and contact_a.by_class == contact_b.by_class
        restore_hole_ok = hole_a == hole_b
        restore_flags_ok = (
            metrics_a.tray_ok == metrics_b.tray_ok
            and metrics_a.peg_ok == metrics_b.peg_ok
            and metrics_a.insert_ok == metrics_b.insert_ok
        )

        return {
            "ran": True,
            "episode_id": int(episode_id),
            "frame": int(target_frame),
            "control_dt_s": float(control_dt_seconds(env)),
            "o2h_translation": o2h_a.translation.tolist(),
            "o2h_rotvec": o2h_a.rotvec.tolist(),
            "contact_total": int(contact_a.total),
            "contact_by_class": dict(contact_a.by_class),
            "finger_force_norm": metrics_a.right_finger_force_norm.tolist(),
            "contact_active": metrics_a.contact_active.astype(bool).tolist(),
            "tray_ok": bool(metrics_a.tray_ok),
            "peg_ok": bool(metrics_a.peg_ok),
            "insert_ok": bool(metrics_a.insert_ok),
            "target_hole": hole_a,
            "tip_to_true_m": feats_a.get("tip_to_true_m"),
            "claim_matches_env": feats_a.get("claim_matches_env"),
            "snapshot_fields": {
                "episode_index": snap.episode_index,
                "zarr_path": snap.zarr_path,
                "t": snap.t,
                "raw_env_step": snap.raw_env_step,
                "peg_lost": snap.peg_lost,
            },
            "restore_consistency": {
                "o2h_pose": restore_pose_ok,
                "contact_counts": restore_contact_ok,
                "target_hole": restore_hole_ok,
                "outcome_flags": restore_flags_ok,
                "all_ok": bool(
                    restore_pose_ok and restore_contact_ok and restore_hole_ok and restore_flags_ok
                ),
            },
            "WRITE_IMPLEMENTATION_ENABLED": bool(WRITE_IMPLEMENTATION_ENABLED),
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-id", type=int, default=0)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--skip-mujoco", action="store_true")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "label_derivability_audit_v1.json",
    )
    args = ap.parse_args()

    if WRITE_IMPLEMENTATION_ENABLED:
        print("REFUSE: WRITE_IMPLEMENTATION_ENABLED must stay False", file=sys.stderr)
        return 2

    items = static_items()
    mujoco_block: dict[str, Any] = {"ran": False}
    if not args.skip_mujoco:
        try:
            mujoco_block = run_mujoco_smoke(args.episode_id, args.frame)
            # Fill restore_ok on items that were live-checked.
            restore_all = bool(mujoco_block.get("restore_consistency", {}).get("all_ok"))
            for it in items:
                if it["name"] in {
                    "object_in_hand_pose_6d",
                    "per_finger_contact_retention",
                    "object_target_mating_transform",
                    "provenance",
                    "contact_mode",
                }:
                    it["snapshot_restore_consistent"] = restore_all
        except Exception as e:
            mujoco_block = {"ran": False, "error": f"{type(e).__name__}: {e}"}

    verdicts = [it["verdict"] for it in items]
    n_der = verdicts.count("derivable")
    n_part = verdicts.count("partial")
    n_block = verdicts.count("blocked")
    mujoco_ok = bool(mujoco_block.get("ran")) and bool(
        mujoco_block.get("restore_consistency", {}).get("all_ok")
    )
    overall = "partial_majority_derivable_or_partial"
    if not mujoco_ok and not args.skip_mujoco:
        overall = "static_ok_mujoco_smoke_failed"
    elif n_block > 0 and n_der == 0:
        overall = "blocked_dominant"
    elif n_der >= 1 and n_part >= 1 and n_block == 0 and (mujoco_ok or args.skip_mujoco):
        overall = "ready_for_observability_label_smoke"

    out = {
        "protocol": "P0-L0",
        "created_at": _utc(),
        "overall_verdict": overall,
        "counts": {"derivable": n_der, "partial": n_part, "blocked": n_block},
        "items": items,
        "mujoco_smoke": mujoco_block,
        "guards": {
            "WRITE_IMPLEMENTATION_ENABLED": False,
            "collect": False,
            "train": False,
            "reopen_c0_c1_c1_1": False,
        },
        "next_branch": (
            "observability_label_smoke_privileged_only"
            if overall == "ready_for_observability_label_smoke"
            else "freeze_contracts_for_partial_or_blocked_labels"
        ),
        "report": "docs/P0_LABEL_DERIVABILITY_AUDIT.md",
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    # Only write audit manifest under data/manifests (existing audit path), never pilot out_root.
    if "pilot_micro_demo" in str(args.manifest.resolve()):
        print("REFUSE: will not write into pilot out_root", file=sys.stderr)
        return 2
    args.manifest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_verdict": overall,
                "manifest": str(args.manifest),
                "mujoco": mujoco_block.get("ran"),
                "mujoco_error": mujoco_block.get("error"),
                "restore_all_ok": (mujoco_block.get("restore_consistency") or {}).get("all_ok"),
            },
            indent=2,
        )
    )
    if args.skip_mujoco:
        return 0
    return 0 if mujoco_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
