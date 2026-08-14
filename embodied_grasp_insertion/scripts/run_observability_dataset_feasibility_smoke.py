#!/usr/bin/env python3
"""P0-Obs-D0 Readonly Dataset Feasibility Smoke.

Scan existing sidecar episodes only. No collect/train/write/pilot/C0 reopen.
Repo gets stats manifest + report; optional /tmp sample pack <=3 episodes.
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
from embodied_grasp_insertion.observability.feasibility import (  # noqa: E402
    HORIZONS,
    FrameRec,
    assign_roots_to_splits,
    atomic_episode_split,
    check_split_leakage,
    count_phase_contiguous_windows,
    count_root_anchored_windows,
    derive_roots_from_history,
    digest_obj,
)
from embodied_grasp_insertion.physics.grasp_metrics import (  # noqa: E402
    compute_step_metrics,
    control_dt_seconds,
    peg_hand_contact_counts,
)
from embodied_grasp_insertion.pilot import ALLOWED_OUT_ROOT, WRITE_IMPLEMENTATION_ENABLED  # noqa: E402
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    make_full_env,
    replay_demo_one_step,
)


PROTOCOL = "P0-Obs-D0"
ACT44_SLICE = slice(0, 44)
FT12_SLICE = slice(70, 82)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_guards(sample_dir: Path | None) -> None:
    if WRITE_IMPLEMENTATION_ENABLED:
        raise RuntimeError("WRITE_IMPLEMENTATION_ENABLED must be False")
    if ALLOWED_OUT_ROOT.exists():
        raise RuntimeError(f"formal pilot out_root exists: {ALLOWED_OUT_ROOT}")
    if sample_dir is not None:
        sd = sample_dir.resolve()
        if "pilot_micro_demo_v0" in str(sd):
            raise RuntimeError("refuse sample under pilot_micro_demo_v0")
        if not str(sd).startswith("/tmp"):
            raise RuntimeError("sample pack must be under /tmp")


class _RunningMoments:
    def __init__(self, dim: int):
        self.n = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)
        self.nan = 0
        self.inf = 0

    def update(self, x: np.ndarray) -> None:
        a = np.asarray(x, dtype=np.float64).reshape(-1)
        if a.size != self.sum.size:
            raise ValueError("dim mismatch")
        if not np.isfinite(a).all():
            self.nan += int(np.isnan(a).sum())
            self.inf += int(np.isinf(a).sum())
            return
        self.n += 1
        self.sum += a
        self.sumsq += a * a

    def summary(self) -> dict[str, Any]:
        if self.n <= 0:
            return {"n": 0, "missing_nan": self.nan, "missing_inf": self.inf}
        mean = self.sum / self.n
        var = self.sumsq / self.n - mean * mean
        return {
            "n": self.n,
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(var, 0.0)).tolist(),
            "missing_nan": self.nan,
            "missing_inf": self.inf,
            "missing_rate": float((self.nan + self.inf) / max(1, self.n * self.sum.size)),
        }


def scan_episode(
    episode_id: int,
    *,
    stop_after_transport_plus: int = 16,
    privilege_stride: int = 5,
) -> dict[str, Any]:
    env = make_full_env([episode_id], seed=0)
    try:
        env.reset()
        assert env._actions is not None
        n_actions = len(env._actions)
        dt_s = float(control_dt_seconds(env))
        history: list[FrameRec] = []
        prev_frame = None
        terminated_at = None
        act_stats = _RunningMoments(44)
        ft_stats = _RunningMoments(12)
        o2h_t = _RunningMoments(3)
        o2h_r = _RunningMoments(3)
        force_norm = _RunningMoments(4)
        contact_totals: list[int] = []
        outcomes = {"tray_ok": 0, "peg_ok": 0, "insert_ok": 0, "n": 0}
        exclusions = {
            "frame_gaps": 0,
            "early_terminate": 0,
            "label_invalid": 0,
            "no_contact_at_recorded": 0,
        }
        transport_frame = None
        first_peg_ok = None
        step_i = 0

        while int(env._t) < n_actions and not env._done:
            info = replay_demo_one_step(env)
            frame = int(env._t)
            gap = False if prev_frame is None else (frame != prev_frame + 1)
            if gap:
                exclusions["frame_gaps"] += 1
            contact = peg_hand_contact_counts(env._raw)
            if contact.total <= 0:
                exclusions["no_contact_at_recorded"] += 1

            obs = np.asarray(env._obs(), dtype=np.float64).reshape(-1)
            if obs.shape[0] < 85:
                exclusions["label_invalid"] += 1
            else:
                act_stats.update(obs[ACT44_SLICE])
                ft_stats.update(obs[FT12_SLICE])

            # Privilege metrics on stride (cheaper); outcomes always from info.
            outcomes["tray_ok"] += int(bool(info["tray_ok"]))
            outcomes["peg_ok"] += int(bool(info["peg_ok"]))
            outcomes["insert_ok"] += int(bool(info["insert_ok"]))
            outcomes["n"] += 1
            if step_i % int(privilege_stride) == 0:
                try:
                    m = compute_step_metrics(env)
                    o2h_t.update(m.object_in_hand.translation)
                    o2h_r.update(m.object_in_hand.rotvec)
                    force_norm.update(m.right_finger_force_norm)
                except Exception:
                    exclusions["label_invalid"] += 1

            rec = FrameRec(
                frame=frame,
                peg_ok=bool(info["peg_ok"]),
                insert_ok=bool(info["insert_ok"]),
                tip_dist_m=float(info["tip_dist_m"]),
                tray_ok=bool(info["tray_ok"]),
                contact_total=int(contact.total),
                terminated=bool(info["terminated"]),
                truncated=bool(info["truncated"]),
                gap_from_prev=bool(gap),
            )
            history.append(rec)
            contact_totals.append(int(contact.total))
            prev_frame = frame
            step_i += 1

            if first_peg_ok is None and rec.peg_ok and not rec.insert_ok:
                first_peg_ok = frame
            if (
                transport_frame is None
                and rec.peg_ok
                and not rec.insert_ok
                and first_peg_ok is not None
                and frame >= first_peg_ok + 20
                and rec.tip_dist_m >= 0.08
            ):
                transport_frame = frame

            if info["terminated"] or info["truncated"]:
                terminated_at = frame
                if frame < n_actions - 1:
                    exclusions["early_terminate"] += 1
                break

            if (
                transport_frame is not None
                and frame >= transport_frame + int(stop_after_transport_plus)
            ):
                break

        roots = derive_roots_from_history(episode_id, history)
        last_frame = history[-1].frame if history else -1
        root_windows = count_root_anchored_windows(
            roots, last_frame=last_frame, terminated_at=terminated_at
        )
        phase_windows = count_phase_contiguous_windows(history)

        return {
            "episode_index": int(episode_id),
            "n_actions": int(n_actions),
            "scanned_frames": len(history),
            "last_frame": int(last_frame),
            "terminated_at": terminated_at,
            "control_dt_s": dt_s,
            "roots": [r.__dict__ | {"root_id": r.root_id} for r in roots],
            "root_anchored_windows": root_windows,
            "phase_contiguous_windows": phase_windows,
            "exclusions": exclusions,
            "deployment_A_act44": {
                "shape": [44],
                "dtype": "float64",
                "units": "action44_m_rad_finger",
                "source": "obs85[0:44]",
                **act_stats.summary(),
            },
            "deployment_B_ft12": {
                "shape": [12],
                "dtype": "float64",
                "units": "wrist_wrench_N_Nm",
                "source": "obs85[70:82]",
                **ft_stats.summary(),
            },
            "privilege_labels": {
                "o2h_translation_m": o2h_t.summary(),
                "o2h_rotvec_rad": o2h_r.summary(),
                "finger_force_norm_N": force_norm.summary(),
                "contact_total_mean": float(np.mean(contact_totals)) if contact_totals else None,
                "contact_total_frac_zero": float(np.mean([c == 0 for c in contact_totals]))
                if contact_totals
                else None,
                "outcome_rates": {
                    k: (float(outcomes[k]) / outcomes["n"] if outcomes["n"] else None)
                    for k in ("tray_ok", "peg_ok", "insert_ok")
                },
            },
            "geometry_family_id": "round_8mm",
            "claims_observability_p0_pass": False,
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def export_tmp_sample(
    episode_ids: list[int],
    episode_rows: dict[int, dict[str, Any]],
    split: dict[str, list[int]],
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pack = {
        "protocol": PROTOCOL,
        "note": "sample_only_not_training_dataset",
        "episodes": [],
        "split_membership": {str(e): next(s for s, ids in split.items() if e in ids) for e in episode_ids},
        "schema_ref": "docs/PRIVILEGED_LABEL_SCHEMA_V1.md",
        "claims_observability_p0_pass": False,
    }
    for e in episode_ids:
        row = episode_rows[e]
        pack["episodes"].append(
            {
                "episode_index": e,
                "roots": row["roots"],
                "root_anchored_windows": row["root_anchored_windows"],
                "control_dt_s": row["control_dt_s"],
            }
        )
    path = out_dir / "sample_pack.json"
    path.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"path": str(path), "n_episodes": len(episode_ids)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="default: 0..99")
    ap.add_argument("--repeat", type=int, default=1, help="full-scan repeats (expensive; default 1)")
    ap.add_argument(
        "--verify-repeat-n",
        type=int,
        default=3,
        help="re-scan this many episodes to verify digest bit-exact",
    )
    ap.add_argument("--sample-tmp-dir", type=Path, default=Path("/tmp/obs_d0_pack_sample"))
    ap.add_argument("--skip-sample", action="store_true")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "observability_dataset_feasibility_v1.json",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "docs" / "OBSERVABILITY_DATASET_FEASIBILITY.md",
    )
    args = ap.parse_args()

    try:
        _refuse_guards(None if args.skip_sample else args.sample_tmp_dir)
    except RuntimeError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2

    episode_ids = list(args.episodes) if args.episodes else list(range(100))
    if "pilot_micro_demo" in str(args.manifest.resolve()):
        print("REFUSE: manifest in pilot root", file=sys.stderr)
        return 2

    def run_pass() -> tuple[list[dict[str, Any]], str]:
        rows = []
        for i, e in enumerate(episode_ids):
            print(f"[D0] scan episode {e} ({i+1}/{len(episode_ids)})", flush=True)
            rows.append(scan_episode(e))
        dig = digest_obj([{k: r[k] for k in (
            "episode_index", "roots", "root_anchored_windows", "phase_contiguous_windows",
            "exclusions", "scanned_frames", "last_frame",
        )} for r in rows])
        return rows, dig

    rows1, dig1 = run_pass()
    dig2 = dig1
    bit_exact = True
    if int(args.repeat) >= 2:
        rows2, dig2 = run_pass()
        bit_exact = dig1 == dig2
        rows1 = rows2
    elif int(args.verify_repeat_n) > 0:
        verify_ids = episode_ids[: int(args.verify_repeat_n)]
        v1 = [r for r in rows1 if r["episode_index"] in verify_ids]
        v2 = [scan_episode(e) for e in verify_ids]
        dig_a = digest_obj([{k: r[k] for k in (
            "episode_index", "roots", "root_anchored_windows", "phase_contiguous_windows",
            "exclusions", "scanned_frames", "last_frame",
        )} for r in v1])
        dig_b = digest_obj([{k: r[k] for k in (
            "episode_index", "roots", "root_anchored_windows", "phase_contiguous_windows",
            "exclusions", "scanned_frames", "last_frame",
        )} for r in v2])
        dig2 = dig_b
        bit_exact = dig_a == dig_b
        dig1 = dig_a

    all_roots = []
    for r in rows1:
        for root in r["roots"]:
            from embodied_grasp_insertion.observability.feasibility import RootRec

            all_roots.append(
                RootRec(
                    episode_index=int(root["episode_index"]),
                    frame=int(root["frame"]),
                    phase=str(root["phase"]),
                    tip_dist_m=float(root["tip_dist_m"]),
                    peg_ok=bool(root["peg_ok"]),
                    insert_ok=bool(root["insert_ok"]),
                    contact_total=int(root["contact_total"]),
                )
            )

    ep_split = atomic_episode_split(episode_ids)
    root_split = assign_roots_to_splits(all_roots, ep_split)
    leak = check_split_leakage(ep_split, root_split)

    # Aggregate
    sum_root_win = {f"H{h}": 0 for h in HORIZONS}
    sum_phase_win = {f"H{h}": 0 for h in HORIZONS}
    excl_tot = {"frame_gaps": 0, "early_terminate": 0, "label_invalid": 0, "no_contact_at_recorded": 0}
    n_with_transport = 0
    n_with_early = 0
    n_zero_roots = 0
    for r in rows1:
        for k in sum_root_win:
            sum_root_win[k] += int(r["root_anchored_windows"][k])
            sum_phase_win[k] += int(r["phase_contiguous_windows"][k])
        for k in excl_tot:
            excl_tot[k] += int(r["exclusions"][k])
        phases = {x["phase"] for x in r["roots"]}
        n_with_transport += int("transport" in phases)
        n_with_early += int("early_grasp" in phases)
        n_zero_roots += int(len(r["roots"]) == 0)

    # Merge deployment / privilege summaries (simple mean of per-ep means where n>0)
    def _merge_field(key_path: list[str]) -> dict[str, Any]:
        means = []
        miss = []
        ns = []
        meta = None
        for r in rows1:
            cur = r
            for k in key_path:
                cur = cur[k]
            if meta is None:
                meta = {k: cur[k] for k in cur if k in ("shape", "dtype", "units", "source")}
            if cur.get("n", 0) > 0 and "mean" in cur:
                means.append(np.asarray(cur["mean"], dtype=np.float64))
                ns.append(cur["n"])
            miss.append(float(cur.get("missing_rate") or 0.0))
        out = dict(meta or {})
        if means:
            out["mean_of_episode_means"] = np.mean(np.stack(means, axis=0), axis=0).tolist()
            out["episodes_with_data"] = len(means)
            out["frames_sum"] = int(sum(ns))
        out["mean_missing_rate"] = float(np.mean(miss)) if miss else None
        return out

    sample_info = None
    if not args.skip_sample:
        # up to 3 from test split
        sample_eps = list(ep_split["test"])[:3]
        if len(sample_eps) < 3:
            sample_eps = (list(ep_split["test"]) + list(ep_split["val"]) + list(ep_split["train"]))[:3]
        by_ep = {r["episode_index"]: r for r in rows1}
        sample_info = export_tmp_sample(sample_eps, by_ep, ep_split, args.sample_tmp_dir)

    overall = "feasibility_pass" if bit_exact and leak["ok"] and n_zero_roots < len(episode_ids) else "feasibility_fail"
    # Never claim Obs P0
    claims = {
        "observability_p0_pass": False,
        "geometry_held_out_available": False,
        "training_authorized": False,
        "full_eval_pack_authorized": False,
        "single_geometry_only": True,
    }

    manifest = {
        "protocol": PROTOCOL,
        "created_at": _utc(),
        "overall_verdict": overall,
        "n_episodes_scanned": len(episode_ids),
        "episode_ids": episode_ids,
        "checks": {
            "bit_exact_repeat": bit_exact,
            "digest_pass1": dig1,
            "digest_pass2": dig2,
            "split_leakage_ok": leak["ok"],
            "split_leakage_issues": leak["issues"],
            "WRITE_IMPLEMENTATION_ENABLED": False,
            "formal_pilot_out_root_exists": False,
            "training_dataset_written": False,
            "collect_new_episodes": False,
            "reopen_c0_c1_c1_1": False,
        },
        "roots_summary": {
            "n_roots_total": len(all_roots),
            "episodes_with_early_grasp": n_with_early,
            "episodes_with_transport": n_with_transport,
            "episodes_with_zero_roots": n_zero_roots,
        },
        "windows": {
            "root_anchored": sum_root_win,
            "phase_contiguous": sum_phase_win,
            "horizons": list(HORIZONS),
        },
        "exclusions_total": excl_tot,
        "split": {
            "method": "sha256(seed:episode) ranked; episode-atomic 70/15/15",
            "seed": 20260814,
            "episodes": ep_split,
            "roots": root_split,
            "counts": {k: len(v) for k, v in ep_split.items()},
        },
        "deployment_inputs": {
            "A_proprio_act44": _merge_field(["deployment_A_act44"]),
            "B_wrist_ft12": _merge_field(["deployment_B_ft12"]),
            "note": "B deployment input = A history + ft12 history; shapes are per-frame",
        },
        "privilege_label_distributions": {
            "o2h_translation_m": _merge_field(["privilege_labels", "o2h_translation_m"]),
            "o2h_rotvec_rad": _merge_field(["privilege_labels", "o2h_rotvec_rad"]),
            "finger_force_norm_N": _merge_field(["privilege_labels", "finger_force_norm_N"]),
            "contact_total_mean_across_eps": float(
                np.mean([r["privilege_labels"]["contact_total_mean"] for r in rows1 if r["privilege_labels"]["contact_total_mean"] is not None])
            )
            if rows1
            else None,
            "outcome_rates_mean": {
                k: float(
                    np.mean(
                        [
                            r["privilege_labels"]["outcome_rates"][k]
                            for r in rows1
                            if r["privilege_labels"]["outcome_rates"][k] is not None
                        ]
                    )
                )
                for k in ("tray_ok", "peg_ok", "insert_ok")
            },
        },
        "sample_pack_tmp": sample_info,
        "claims": claims,
        "per_episode": [
            {
                "episode_index": r["episode_index"],
                "roots": r["roots"],
                "root_anchored_windows": r["root_anchored_windows"],
                "phase_contiguous_windows": r["phase_contiguous_windows"],
                "exclusions": r["exclusions"],
                "scanned_frames": r["scanned_frames"],
            }
            for r in rows1
        ],
        "report": path_for_manifest(args.report, project_root=PROJECT_ROOT),
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Observability Dataset Feasibility (P0-Obs-D0)",
        "",
        f"- 日期：{manifest['created_at']}",
        f"- overall_verdict：**{overall}**",
        f"- episodes scanned：{len(episode_ids)}",
        f"- bit-exact repeat：{bit_exact}",
        f"- split leakage ok：{leak['ok']}",
        f"- roots：{len(all_roots)}（early_grasp eps={n_with_early}, transport eps={n_with_transport}, zero_roots={n_zero_roots}）",
        f"- root-anchored windows：{sum_root_win}",
        f"- phase-contiguous windows：{sum_phase_win}",
        f"- exclusions：{excl_tot}",
        f"- split counts：{manifest['split']['counts']}",
        "- **claims_observability_p0_pass=false**（单几何）",
        "- 未训练、未采集、未开写盘、未写正式 pilot、未重开 C0/C1/C1.1",
        f"- manifest：`{path_for_manifest(args.manifest, project_root=PROJECT_ROOT)}`",
        f"- /tmp sample：{sample_info}",
        "",
        "## Decision branch",
        "",
        "- 字段/覆盖足够 → 可审批**完整只读评测包**导出（另授权）",
        "- 缺失严重 → 先修输入/标签接口",
        "- 即使可导出 → 训练仍须单独授权",
        "",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "overall_verdict": overall,
                "bit_exact": bit_exact,
                "leak_ok": leak["ok"],
                "n_roots": len(all_roots),
                "windows_root": sum_root_win,
                "manifest": path_for_manifest(args.manifest, project_root=PROJECT_ROOT),
                "claims_observability_p0_pass": False,
            },
            indent=2,
        )
    )
    return 0 if overall.startswith("feasibility_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
