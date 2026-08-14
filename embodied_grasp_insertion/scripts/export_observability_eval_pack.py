#!/usr/bin/env python3
"""P0-Obs-D1: export full readonly Observability evaluation pack.

No collect/train/write-switch/pilot/C0. Large binaries go to /mnt/hdd (not git).
"""

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

from embodied_grasp_insertion.io_paths import path_for_manifest  # noqa: E402
from embodied_grasp_insertion.labels.privileged_schema import (  # noqa: E402
    SCHEMA_VERSION,
    extract_privileged_frame,
)
from embodied_grasp_insertion.observability.eval_pack import (  # noqa: E402
    PACK_NAME,
    PRIMARY_H,
    PROTOCOL,
    STORE_H,
    build_fixed_split,
    pack_banner,
    sample_meta,
    validate_sample_arrays,
)
from embodied_grasp_insertion.observability.feasibility import (  # noqa: E402
    RootRec,
    assign_roots_to_splits,
    check_split_leakage,
    digest_obj,
)
from embodied_grasp_insertion.physics.grasp_metrics import control_dt_seconds  # noqa: E402
from embodied_grasp_insertion.pilot import ALLOWED_OUT_ROOT, WRITE_IMPLEMENTATION_ENABLED  # noqa: E402
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    make_full_env,
    replay_demo_one_step,
)

DEFAULT_OUT = Path(
    "/mnt/hdd/dexjoco/datasets/embodied_grasp_insertion/observability_eval_v1"
)
D0_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "observability_dataset_feasibility_v1.json"
ACT44 = slice(0, 44)
FT12 = slice(70, 82)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _guards(out_root: Path) -> None:
    if WRITE_IMPLEMENTATION_ENABLED:
        raise RuntimeError("WRITE_IMPLEMENTATION_ENABLED must stay False")
    if ALLOWED_OUT_ROOT.exists():
        raise RuntimeError(f"formal pilot out_root exists: {ALLOWED_OUT_ROOT}")
    rp = out_root.resolve()
    if "pilot_micro_demo_v0" in str(rp):
        raise RuntimeError("refuse writing into pilot_micro_demo_v0")
    # Prefer independent dataset path under /mnt/hdd
    if str(rp).startswith(str(PROJECT_ROOT.resolve())):
        raise RuntimeError("refuse packing large binaries inside git project tree")


def _load_d0_roots() -> dict[int, list[RootRec]]:
    man = json.loads(D0_MANIFEST.read_text(encoding="utf-8"))
    by: dict[int, list[RootRec]] = {}
    for ep in man["per_episode"]:
        e = int(ep["episode_index"])
        roots = []
        for r in ep["roots"]:
            roots.append(
                RootRec(
                    episode_index=int(r["episode_index"]),
                    frame=int(r["frame"]),
                    phase=str(r["phase"]),
                    tip_dist_m=float(r["tip_dist_m"]),
                    peg_ok=bool(r["peg_ok"]),
                    insert_ok=bool(r["insert_ok"]),
                    contact_total=int(r["contact_total"]),
                )
            )
        by[e] = roots
    return by


def _safe_name(root_id: str) -> str:
    return root_id.replace(":", "_")


def export_episode(
    episode_id: int,
    roots: list[RootRec],
    *,
    ep_to_split: dict[int, str],
    samples_dir: Path,
) -> list[dict[str, Any]]:
    need_end = max(int(r.frame) + STORE_H - 1 for r in roots)
    env = make_full_env([episode_id], seed=0)
    index_rows: list[dict[str, Any]] = []
    try:
        env.reset()
        dt_s = float(control_dt_seconds(env))
        # Collect frame buffers until need_end
        buf: dict[int, dict[str, Any]] = {}
        prev_o2h = None
        # Start from t=0; privilege velocity uses consecutive control frames globally,
        # but sample window resets availability at window[0] when assembling.
        while int(env._t) < need_end and not env._done:
            info = replay_demo_one_step(env)
            frame = int(env._t)
            obs = np.asarray(env._obs(), dtype=np.float64).reshape(-1)
            # For label extract, pass prev within global stream; window rewrites vel later.
            # Use a dummy root for provenance during collect; overwritten per-sample.
            label, o2h = extract_privileged_frame(
                env,
                episode_index=episode_id,
                frame=frame,
                root_id=f"{episode_id}:{frame}:collect",
                root_phase="collect",
                prev_o2h=prev_o2h,
                dt_s=dt_s,
            )
            buf[frame] = {
                "act44": obs[ACT44].astype(np.float64).copy(),
                "ft12": obs[FT12].astype(np.float64).copy(),
                "sim_time_s": float(env._raw._data.time),
                "label": label,
                "o2h": o2h,
                "info": info,
            }
            prev_o2h = o2h
            if info.get("terminated") or info.get("truncated"):
                break

        for root in roots:
            frames = list(range(int(root.frame), int(root.frame) + STORE_H))
            missing = [f for f in frames if f not in buf]
            if missing:
                raise RuntimeError(
                    f"ep{episode_id} root {root.root_id} missing frames {missing[:5]}..."
                )
            act44 = np.stack([buf[f]["act44"] for f in frames], axis=0)
            ft12 = np.stack([buf[f]["ft12"] for f in frames], axis=0)
            o2h_t = np.stack(
                [np.asarray(buf[f]["label"]["object_in_hand_pose_6d"]["translation_m"]) for f in frames],
                axis=0,
            )
            o2h_r = np.stack(
                [np.asarray(buf[f]["label"]["object_in_hand_pose_6d"]["rotvec_rad"]) for f in frames],
                axis=0,
            )
            # Recompute window-local velocity (first unavailable)
            from embodied_grasp_insertion.labels.privileged_schema import (
                null_velocity,
                o2h_velocity_from_poses,
            )
            from embodied_grasp_insertion.physics.grasp_metrics import (
                REFERENCE_BODY,
                ObjectInHandPose,
            )

            vel_lin = np.full((STORE_H, 3), np.nan, dtype=np.float64)
            vel_ang = np.full((STORE_H, 3), np.nan, dtype=np.float64)
            vel_av = np.zeros((STORE_H,), dtype=bool)
            poses = [
                ObjectInHandPose(
                    reference_body=REFERENCE_BODY,
                    translation=o2h_t[i],
                    rotvec=o2h_r[i],
                )
                for i in range(STORE_H)
            ]
            for i in range(STORE_H):
                if i == 0:
                    v = null_velocity()
                else:
                    v = o2h_velocity_from_poses(poses[i - 1], poses[i], dt_s)
                vel_av[i] = bool(v.available)
                if v.available:
                    vel_lin[i] = np.asarray(v.linear_mps, dtype=np.float64)
                    vel_ang[i] = np.asarray(v.angular_radps, dtype=np.float64)

            contact_total = np.asarray(
                [buf[f]["label"]["peg_hand_contact"]["total"] for f in frames], dtype=np.int64
            )
            by = ["palm", "index", "middle", "ring", "thumb"]
            contact_by = np.stack(
                [
                    np.asarray(
                        [buf[f]["label"]["peg_hand_contact"]["by_finger"][k] for k in by],
                        dtype=np.int64,
                    )
                    for f in frames
                ],
                axis=0,
            )
            force_norm = np.stack(
                [np.asarray(buf[f]["label"]["finger_force"]["right_force_norm_N"]) for f in frames],
                axis=0,
            )
            active = np.stack(
                [np.asarray(buf[f]["label"]["finger_force"]["contact_active"]) for f in frames],
                axis=0,
            )
            tray = np.asarray([buf[f]["label"]["outcome_raw"]["tray_ok"] for f in frames])
            peg = np.asarray([buf[f]["label"]["outcome_raw"]["peg_ok"] for f in frames])
            ins = np.asarray([buf[f]["label"]["outcome_raw"]["insert_ok"] for f in frames])
            sim_t = np.asarray([buf[f]["sim_time_s"] for f in frames], dtype=np.float64)

            lab0 = buf[frames[0]]["label"]
            meta = sample_meta(
                episode_index=episode_id,
                root_id=root.root_id,
                root_phase=root.phase,
                root_frame=root.frame,
                split=ep_to_split[episode_id],
                geometry_family_id=lab0["provenance"]["geometry_family_id"],
                target_instance_id=lab0["provenance"]["target_instance_id"],
                socket_site=lab0["provenance"]["socket_site"],
            )
            arrays = {
                "frames": np.asarray(frames, dtype=np.int64),
                "sim_time_s": sim_t,
                "act44": act44,
                "ft12": ft12,
                "o2h_translation_m": o2h_t.astype(np.float64),
                "o2h_rotvec_rad": o2h_r.astype(np.float64),
                "o2h_vel_linear_mps": vel_lin,
                "o2h_vel_angular_radps": vel_ang,
                "o2h_vel_available": vel_av,
                "contact_total": contact_total,
                "contact_by_finger": contact_by,
                "finger_force_norm_N": force_norm.astype(np.float64),
                "contact_active": active.astype(bool),
                "tray_ok": tray.astype(bool),
                "peg_ok": peg.astype(bool),
                "insert_ok": ins.astype(bool),
            }
            issues = validate_sample_arrays(arrays, meta)
            if issues:
                raise RuntimeError(f"sample invalid {root.root_id}: {issues}")

            fname = f"ep{episode_id:03d}_{root.phase}_f{root.frame}.npz"
            path = samples_dir / fname
            np.savez_compressed(path, meta_json=np.asarray(json.dumps(meta)), **arrays)
            index_rows.append(
                {
                    "file": fname,
                    "root_id": root.root_id,
                    "episode_index": episode_id,
                    "split": meta["split"],
                    "root_phase": root.phase,
                    "root_frame": root.frame,
                    "primary_horizon": PRIMARY_H,
                    "stored_horizon": STORE_H,
                }
            )
        return index_rows
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument(
        "--repo-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "observability_eval_pack_v1.json",
    )
    ap.add_argument(
        "--repo-report",
        type=Path,
        default=PROJECT_ROOT / "docs" / "OBSERVABILITY_EVAL_PACK_EXPORT.md",
    )
    args = ap.parse_args()

    try:
        _guards(args.out_root)
    except RuntimeError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2

    roots_by_ep = _load_d0_roots()
    episode_ids = sorted(roots_by_ep.keys())
    if args.max_episodes is not None:
        episode_ids = episode_ids[: int(args.max_episodes)]

    split_info = build_fixed_split(list(range(100)))
    ep_split = split_info["episodes"]
    ep_to_split = {e: sp for sp, ids in ep_split.items() for e in ids}

    out = args.out_root
    samples_dir = out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    (out / "PACK_BANNER.json").write_text(
        json.dumps(pack_banner(), indent=2) + "\n", encoding="utf-8"
    )
    (out / "split.json").write_text(json.dumps(split_info, indent=2) + "\n", encoding="utf-8")

    all_index: list[dict[str, Any]] = []
    all_roots: list[RootRec] = []
    for i, ep in enumerate(episode_ids):
        roots = roots_by_ep[ep]
        all_roots.extend(roots)
        print(f"[D1] export ep {ep} ({i+1}/{len(episode_ids)}) roots={len(roots)}", flush=True)
        all_index.extend(
            export_episode(ep, roots, ep_to_split=ep_to_split, samples_dir=samples_dir)
        )

    root_split = assign_roots_to_splits(all_roots, ep_split)
    leak = check_split_leakage(ep_split, root_split)

    # Second-pass digest: re-read all npz meta+shapes
    dig_payload = []
    val_issues = []
    for row in all_index:
        p = samples_dir / row["file"]
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            arrays = {k: z[k] for k in z.files if k != "meta_json"}
            # meta_json stored as 0-d array of str sometimes
            if isinstance(z["meta_json"], np.ndarray):
                meta = json.loads(z["meta_json"].item() if z["meta_json"].shape == () else str(z["meta_json"]))
            issues = validate_sample_arrays(arrays, meta)
            if issues:
                val_issues.append({"file": row["file"], "issues": issues})
            dig_payload.append(
                {
                    "file": row["file"],
                    "root_id": row["root_id"],
                    "split": row["split"],
                    "frames0": int(arrays["frames"][0]),
                    "act44_sum": float(np.asarray(arrays["act44"]).sum()),
                    "ft12_sum": float(np.asarray(arrays["ft12"]).sum()),
                }
            )
    dig1 = digest_obj(dig_payload)
    # Repeat digest from same files
    dig_payload2 = []
    for row in all_index:
        p = samples_dir / row["file"]
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "meta_json"}
            dig_payload2.append(
                {
                    "file": row["file"],
                    "root_id": row["root_id"],
                    "split": row["split"],
                    "frames0": int(arrays["frames"][0]),
                    "act44_sum": float(np.asarray(arrays["act44"]).sum()),
                    "ft12_sum": float(np.asarray(arrays["ft12"]).sum()),
                }
            )
    dig2 = digest_obj(dig_payload2)
    bit_exact = dig1 == dig2

    index_path = out / "index.json"
    index_path.write_text(json.dumps({"samples": all_index}, indent=2) + "\n", encoding="utf-8")

    overall = (
        "export_pass"
        if bit_exact and leak["ok"] and not val_issues and len(all_index) == len(all_roots)
        else "export_fail"
    )

    pack_manifest = {
        "protocol": PROTOCOL,
        "created_at": _utc(),
        "overall_verdict": overall,
        "out_root": str(out),
        "n_samples": len(all_index),
        "n_episodes": len(episode_ids),
        "primary_horizon": PRIMARY_H,
        "stored_horizon": STORE_H,
        "schema_version": SCHEMA_VERSION,
        "split": split_info,
        "checks": {
            "bit_exact_digest": bit_exact,
            "digest_1": dig1,
            "digest_2": dig2,
            "split_leakage_ok": leak["ok"],
            "sample_validation_issues": val_issues[:20],
            "n_validation_issues": len(val_issues),
            "WRITE_IMPLEMENTATION_ENABLED": False,
            "formal_pilot_out_root_exists": False,
        },
        "claims": {
            "evaluation_only": True,
            "training_authorized": False,
            "single_geometry_only": True,
            "claims_observability_p0_pass": False,
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(pack_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Repo-side slim stats (no binaries)
    repo_man = {
        **pack_manifest,
        "out_root": str(out),
        "git_policy": "binaries_not_committed_hdd_only",
        "report": path_for_manifest(args.repo_report, project_root=PROJECT_ROOT),
    }
    args.repo_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.repo_manifest.write_text(
        json.dumps(repo_man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.repo_report.write_text(
        "\n".join(
            [
                "# Observability Eval Pack Export (P0-Obs-D1)",
                "",
                f"- 日期：{repo_man['created_at']}",
                f"- overall_verdict：**{overall}**",
                f"- out_root：`{out}`（大文件不进 Git）",
                f"- samples：{len(all_index)}（期望 200）",
                f"- primary H={PRIMARY_H}；stored H={STORE_H}",
                f"- split digest：`{split_info['digest'][:16]}…` counts={split_info['counts']}",
                f"- bit-exact：{bit_exact}；leak_ok：{leak['ok']}；val_issues：{len(val_issues)}",
                "- evaluation_only=true；training_authorized=false；single_geometry_only=true",
                "- **claims_observability_p0_pass=false**",
                "- 未训练、未采集、写盘开关仍关、未写 pilot、未重开 C0/C1/C1.1",
                "",
                "## Next",
                "",
                "- 审查评测包内容与分布",
                "- **仍不训练**；另授权后才开小型 Obs baseline",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "overall_verdict": overall,
                "n_samples": len(all_index),
                "out_root": str(out),
                "bit_exact": bit_exact,
                "leak_ok": leak["ok"],
                "n_val_issues": len(val_issues),
                "claims_observability_p0_pass": False,
            },
            indent=2,
        )
    )
    return 0 if overall == "export_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
