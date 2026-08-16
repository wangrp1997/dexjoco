#!/usr/bin/env python3
"""Readonly audit: is successful/recoverable handoff support present in existing data?"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEXJOCO_ROOT = PROJECT_ROOT.parent
EMBODIED = DEXJOCO_ROOT / "embodied_grasp_insertion"
REACH = DEXJOCO_ROOT.parent / "reach_insert_rl"
for _p in (str(PROJECT_ROOT), str(DEXJOCO_ROOT), str(DEXJOCO_ROOT / "dexjoco"), str(EMBODIED), str(REACH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.physics.grasp_metrics import (  # noqa: E402
    object_in_hand_pose,
    peg_hand_contact_counts,
    relative_pose_error,
)
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    make_full_env,
    replay_demo_one_step,
)
from reach_insert_rl.env.handoff_env import InsertHandoffEnv, load_manifest_entries  # noqa: E402
from reach_insert_rl.env.full_obs import privileged_full_features  # noqa: E402
from reach_insert_rl.env.obs import privileged_geom_features  # noqa: E402
from pose_insert.pre_insert import resolve_peg_lift_end_frame  # noqa: E402

PROTOCOL = "HandoffSupportRegionAudit"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _feat_pack(raw, o2h0=None) -> dict[str, Any]:
    feat = privileged_full_features(raw)
    o2h = object_in_hand_pose(raw)
    contact = peg_hand_contact_counts(raw)
    out = {
        "tip_m": float(feat["tip_dist"]),
        "lat_m": float(feat["lat_err"]),
        "along_m": float(feat["along"]),
        "axis_err_rad": float(feat["axis_err"]),
        "peg_ok": bool(raw._labeler.compute(raw).peg_ok) if hasattr(raw, "_labeler") else None,
        "contact_total": int(contact.total),
        "o2h_t": o2h.translation.tolist(),
        "o2h_rv": o2h.rotvec.tolist(),
    }
    if o2h0 is not None:
        dt, dr = relative_pose_error(o2h0, o2h)
        out["o2h_drift_trans_m"] = float(dt)
        out["o2h_drift_rot_rad"] = float(dr)
    return out


def scan_demo_episode(env, entry: dict, *, sidecar: Path) -> dict[str, Any]:
    """Single pass demo replay; record grasp / transport / handoff markers."""
    env.reset(entry=entry)
    assert env._actions is not None and env._spec is not None
    ep = int(env._spec.episode_index)
    peg_lift_end = int(resolve_peg_lift_end_frame(entry, sidecar))

    first_peg = None
    transport = None
    handoff = None
    o2h_first = None
    contact_first = None
    max_trans_drift = 0.0
    max_rot_drift = 0.0
    peg_loss_steps = 0
    n_steps = 0
    insert_ok_seen = False
    raw = env._raw

    while not env._done and int(env._t) < len(env._actions):
        t = int(env._t)
        outcome = env._labeler.compute(raw)
        feat = privileged_full_features(raw)
        o2h = object_in_hand_pose(raw)
        contact = peg_hand_contact_counts(raw)
        if outcome.peg_ok and first_peg is None:
            first_peg = {
                "frame": t,
                "tip_m": float(feat["tip_dist"]),
                "lat_m": float(feat["lat_err"]),
                "along_m": float(feat["along"]),
                "axis_err_rad": float(feat["axis_err"]),
                "contact_total": int(contact.total),
                "o2h_t": o2h.translation.tolist(),
                "o2h_rv": o2h.rotvec.tolist(),
            }
            o2h_first = o2h
            contact_first = int(contact.total)
        if (
            transport is None
            and first_peg is not None
            and outcome.peg_ok
            and t >= int(first_peg["frame"]) + 20
            and float(feat["tip_dist"]) >= 0.08
        ):
            dt, dr = relative_pose_error(o2h_first, o2h)
            transport = {
                "frame": t,
                "tip_m": float(feat["tip_dist"]),
                "lat_m": float(feat["lat_err"]),
                "along_m": float(feat["along"]),
                "axis_err_rad": float(feat["axis_err"]),
                "contact_total": int(contact.total),
                "o2h_drift_trans_m": float(dt),
                "o2h_drift_rot_rad": float(dr),
                "contact_retention": float(contact.total / max(contact_first, 1)),
            }
        if first_peg is not None and o2h_first is not None:
            dt, dr = relative_pose_error(o2h_first, o2h)
            max_trans_drift = max(max_trans_drift, float(dt))
            max_rot_drift = max(max_rot_drift, float(dr))
            if contact.total <= 0:
                peg_loss_steps += 1
        if handoff is None and t >= int(peg_lift_end):
            dt, dr = (0.0, 0.0)
            if o2h_first is not None:
                dt, dr = relative_pose_error(o2h_first, o2h)
            handoff = {
                "frame": t,
                "tip_m": float(feat["tip_dist"]),
                "lat_m": float(feat["lat_err"]),
                "along_m": float(feat["along"]),
                "axis_err_rad": float(feat["axis_err"]),
                "contact_total": int(contact.total),
                "o2h_t": o2h.translation.tolist(),
                "o2h_rv": o2h.rotvec.tolist(),
                "o2h_drift_trans_m": float(dt),
                "o2h_drift_rot_rad": float(dr),
                "contact_retention": float(
                    contact.total / max(contact_first, 1) if contact_first else 0.0
                ),
                "peg_ok": bool(outcome.peg_ok),
            }
        if outcome.insert_ok:
            insert_ok_seen = True
        try:
            replay_demo_one_step(env)
        except Exception as e:
            return {
                "episode_index": ep,
                "error": str(e),
                "peg_lift_end": peg_lift_end,
                "handoff": handoff,
                "first_peg_ok": first_peg,
                "transport": transport,
            }
        n_steps += 1
        # Stop soon after handoff: demos are successful trajectories; full insert replay is costly.
        if handoff is not None and t >= int(peg_lift_end) + 2:
            break
        if n_steps > len(env._actions) + 2:
            break

    return {
        "episode_index": ep,
        "peg_lift_end": int(peg_lift_end),
        "n_steps_scanned": n_steps,
        "first_peg_ok": first_peg,
        "transport": transport,
        "handoff": handoff,
        "transport_max_o2h_drift_trans_m": float(max_trans_drift),
        "transport_max_o2h_drift_rot_rad": float(max_rot_drift),
        "peg_contact_absent_steps": int(peg_loss_steps),
        "demo_insert_ok_seen": bool(insert_ok_seen),
        "demo_is_success_trajectory": True,
        "source": "demo_sidecar",
    }


def handoff_from_inserthandoff(env: InsertHandoffEnv, ep: int) -> dict[str, Any]:
    obs, info = env.reset(episode_index=int(ep))
    feat = privileged_geom_features(env._raw)
    o2h = object_in_hand_pose(env._raw)
    contact = peg_hand_contact_counts(env._raw)
    return {
        "episode_index": int(ep),
        "peg_lift_end": int(info.get("peg_lift_end", -1)),
        "tip_m": float(info.get("tip_socket_dist_m", feat["tip_dist"])),
        "lat_m": float(feat["lat_err"]),
        "along_m": float(feat["along"]),
        "axis_err_rad": float(feat["axis_err"]),
        "contact_total": int(contact.total),
        "o2h_t": o2h.translation.tolist(),
        "o2h_rv": o2h.rotvec.tolist(),
        "source": "inserthandoff_reset",
    }


def percentile_box(points: np.ndarray, lo: float, hi: float) -> dict[str, list[float]]:
    """Axis-aligned box from percentiles; points (N,2) tip,lat."""
    if points.size == 0:
        return {"lo": [0.0, 0.0], "hi": [0.0, 0.0], "empty": True}
    plo = np.percentile(points, lo, axis=0)
    phi = np.percentile(points, hi, axis=0)
    return {"lo": plo.tolist(), "hi": phi.tolist(), "empty": False}


def inside_box(pt: np.ndarray, box: dict) -> bool:
    if box.get("empty"):
        return False
    lo = np.asarray(box["lo"], dtype=np.float64)
    hi = np.asarray(box["hi"], dtype=np.float64)
    return bool(np.all(pt >= lo) and np.all(pt <= hi))


def knn_median(points: np.ndarray, k: int) -> float:
    if len(points) < k + 1:
        return float("inf")
    dmat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    knn = np.sort(dmat, axis=1)[:, :k]
    return float(np.median(knn.mean(axis=1)))


def judge(demo_rows, policy_rows, cfg) -> dict[str, Any]:
    demo_h = [r for r in demo_rows if r.get("handoff")]
    succ_pts = []
    for r in demo_h:
        h = r["handoff"]
        succ_pts.append([h["tip_m"], h["lat_m"]])
    for r in policy_rows:
        if r.get("insert_ok"):
            succ_pts.append([r["handoff"]["tip_m"], r["handoff"]["lat_m"]])
    succ_pts = np.asarray(succ_pts, dtype=np.float64).reshape(-1, 2)

    fail_pol = [r for r in policy_rows if not r.get("insert_ok")]
    fail_pts = np.asarray(
        [[r["handoff"]["tip_m"], r["handoff"]["lat_m"]] for r in fail_pol], dtype=np.float64
    ).reshape(-1, 2)

    box = percentile_box(
        succ_pts, float(cfg["success_percentile_lo"]), float(cfg["success_percentile_hi"])
    )
    knn = knn_median(succ_pts, int(cfg["nn_k"])) if len(succ_pts) else float("inf")

    n_fail_inside = int(sum(1 for p in fail_pts if inside_box(p, box))) if len(fail_pts) else 0
    frac_fail_inside = float(n_fail_inside / len(fail_pts)) if len(fail_pts) else None

    # transport / grasp stats from demos
    drifts = [r["transport_max_o2h_drift_trans_m"] for r in demo_rows if "transport_max_o2h_drift_trans_m" in r]
    rets = [
        r["handoff"]["contact_retention"]
        for r in demo_h
        if r["handoff"].get("contact_retention") is not None
    ]
    demo_insert = sum(1 for r in demo_rows if r.get("demo_insert_ok_seen"))

    n_succ = int(len(succ_pts))
    continuous = bool(np.isfinite(knn) and knn <= float(cfg["max_success_knn_median_m"]))
    enough = n_succ >= int(cfg["min_success_handoffs"])
    coverage_ok = (
        frac_fail_inside is not None and frac_fail_inside >= float(cfg["coverage_fail_inside_min"])
    )

    if not enough or not continuous:
        verdict = "cannot_confirm_support"
        summary = (
            "成功 handoff 样本不足或不连续；无法从现有数据确认可学习支持区域。"
            "应暂停主线或重做数据设计，而非继续换模型。"
        )
        branch = 3
    elif coverage_ok:
        verdict = "support_confirmed_coverage_ok"
        summary = (
            "成功 handoff 形成连续支持区，且多数策略失败仍落在区内："
            "覆盖不是主瓶颈，应审插入策略/接口，不先扩采。"
        )
        branch = 1
    else:
        verdict = "coverage_gap"
        summary = (
            "成功区域存在，但失败 handoff 大量落在区外：现有数据对可恢复 handoff 覆盖不足，"
            "应考虑重做数据生成。"
        )
        branch = 2

    return {
        "verdict": verdict,
        "branch": branch,
        "summary": summary,
        "n_demo_scanned": len(demo_rows),
        "n_demo_with_handoff": len(demo_h),
        "n_demo_insert_ok_seen": int(demo_insert),
        "n_success_handoff_points": n_succ,
        "n_policy_fail": len(fail_pol),
        "n_fail_inside_success_box": n_fail_inside,
        "frac_fail_inside_success_box": frac_fail_inside,
        "success_tip_lat_box": box,
        "success_knn_median_m": knn if np.isfinite(knn) else None,
        "continuous_enough": continuous,
        "enough_successes": enough,
        "coverage_ok": coverage_ok,
        "demo_transport_drift_trans_mean_m": float(np.mean(drifts)) if drifts else None,
        "demo_transport_drift_trans_p95_m": float(np.percentile(drifts, 95)) if drifts else None,
        "demo_handoff_contact_retention_mean": float(np.mean(rets)) if rets else None,
        "policy_succ_tip_mean_m": float(
            np.mean([r["handoff"]["tip_m"] for r in policy_rows if r.get("insert_ok")])
        )
        if any(r.get("insert_ok") for r in policy_rows)
        else None,
        "policy_fail_tip_mean_m": float(np.mean(fail_pts[:, 0])) if len(fail_pts) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "handoff_support_audit.yaml",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = PROJECT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = PROJECT_ROOT / cfg["manifest_path"]
    man_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / cfg["report_path"]

    sidecar = Path(cfg["sidecar_dir"])
    # --- PrivHI policy contrast via InsertHandoffEnv ---
    expand = json.loads(Path(cfg["privhi_expand15"]).read_text(encoding="utf-8"))
    holdout = json.loads(Path(cfg["privhi_holdout"]).read_text(encoding="utf-8"))
    # unique eps from expand (holdout subset)
    ep_meta = {int(e["ep"]): e for e in expand["episodes"]}
    for e in holdout["episodes"]:
        ep_meta.setdefault(int(e["ep"]), e)

    entries = load_manifest_entries(sidecar, episode_indices=sorted(ep_meta.keys()))
    henv = InsertHandoffEnv(entries, sidecar_dir=sidecar, seed=int(cfg["seed"]), use_force=True)
    policy_rows = []
    for ep, meta in sorted(ep_meta.items()):
        print(f"[privhi-handoff] ep{ep} insert_ok={meta.get('insert_ok')}", flush=True)
        try:
            h = handoff_from_inserthandoff(henv, ep)
        except Exception as ex:
            policy_rows.append(
                {
                    "episode_index": ep,
                    "insert_ok": bool(meta.get("insert_ok")),
                    "eval_tip0_mm": meta.get("tip0_mm"),
                    "fail_reason": meta.get("reason"),
                    "error": str(ex),
                    "handoff": None,
                }
            )
            continue
        policy_rows.append(
            {
                "episode_index": ep,
                "insert_ok": bool(meta.get("insert_ok")),
                "eval_tip0_mm": meta.get("tip0_mm"),
                "eval_lat_at_min_mm": meta.get("lat_at_min_mm"),
                "fail_reason": meta.get("reason"),
                "split": "holdout" if any(int(x["ep"]) == ep for x in holdout["episodes"]) else "expand15",
                "handoff": h,
            }
        )
    henv.close() if hasattr(henv, "close") else None

    # --- Demo support scan ---
    man = json.loads((sidecar / "manifest.json").read_text(encoding="utf-8"))
    all_eps = [int(e["episode_index"]) for e in man["episodes"]]
    all_eps = all_eps[: int(cfg["demo_max_episodes"])]
    fenv = make_full_env(all_eps, sidecar_dir=sidecar, seed=int(cfg["seed"]))
    demo_rows = []
    for ep in all_eps:
        entry = next(e for e in fenv.entries if int(e["episode_index"]) == ep)
        print(f"[demo-scan] ep{ep}", flush=True)
        try:
            row = scan_demo_episode(fenv, entry, sidecar=sidecar)
        except Exception as ex:
            row = {"episode_index": ep, "error": str(ex)}
        demo_rows.append(row)
        (out_dir / f"demo_ep{ep:03d}.json").write_text(
            json.dumps(row, indent=2, default=float), encoding="utf-8"
        )

    verdict = judge(demo_rows, policy_rows, cfg)
    manifest = {
        "protocol": PROTOCOL,
        "finished_at": _utc(),
        "config": cfg,
        "verdict": verdict,
        "policy_rows": policy_rows,
        "demo_index": [
            {
                "episode_index": r.get("episode_index"),
                "has_handoff": bool(r.get("handoff")),
                "demo_insert_ok_seen": r.get("demo_insert_ok_seen"),
                "handoff_tip_m": (r.get("handoff") or {}).get("tip_m"),
                "handoff_lat_m": (r.get("handoff") or {}).get("lat_m"),
                "transport_max_o2h_drift_trans_m": r.get("transport_max_o2h_drift_trans_m"),
                "error": r.get("error"),
            }
            for r in demo_rows
        ],
    }
    man_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, default=float), encoding="utf-8"
    )

    lines = [
        "# Handoff Support-Region Audit Result",
        "",
        f"- 完成：{_utc()}",
        f"- 判定：`{verdict['verdict']}`",
        f"- 分支：{verdict['branch']}",
        f"- 摘要：{verdict['summary']}",
        "",
        "## 关键数字",
        "",
        f"- demo 扫描：{verdict['n_demo_scanned']}（handoff {verdict['n_demo_with_handoff']}）",
        f"- demo 见 insert_ok：{verdict['n_demo_insert_ok_seen']}",
        f"- 成功 handoff 点数：{verdict['n_success_handoff_points']}",
        f"- 成功 tip-lat box：{verdict['success_tip_lat_box']}",
        f"- 成功 kNN median：{verdict['success_knn_median_m']}",
        f"- 策略失败数：{verdict['n_policy_fail']}；落在成功区内：{verdict['n_fail_inside_success_box']}（{verdict['frac_fail_inside_success_box']}）",
        f"- demo transport o2h drift mean/p95：{verdict['demo_transport_drift_trans_mean_m']} / {verdict['demo_transport_drift_trans_p95_m']}",
        f"- demo handoff contact retention mean：{verdict['demo_handoff_contact_retention_mean']}",
        "",
        "## 决策",
        "",
    ]
    if verdict["branch"] == 1:
        lines += ["- 支持可确认且覆盖失败 → **不先扩采**；审插入策略/阶段接口。", ""]
    elif verdict["branch"] == 2:
        lines += ["- **覆盖缺口** → 考虑重做数据生成（仍不宣称仿真不可解）。", ""]
    else:
        lines += ["- **无法确认支持** → 暂停插孔研究主线或先重做数据设计。", ""]

    report_path.write_text("\n".join(lines), encoding="utf-8")

    state = {
        "date": "2026-08-15",
        "phase": "handoff_support_audit_complete",
        "busy": False,
        "training_allowed": False,
        "collection_allowed": False,
        "candidate_b": "stopped_no_salvage",
        "handoff_support_audit": {
            "verdict": verdict["verdict"],
            "branch": verdict["branch"],
            "manifest": str(man_path.relative_to(PROJECT_ROOT)),
            "report": str(report_path.relative_to(PROJECT_ROOT)),
        },
        "next_action": {
            1: "audit_insert_policy_interface_no_collection",
            2: "redesign_data_generation_for_recoverable_handoff",
            3: "pause_or_redesign_data_support",
        }[verdict["branch"]],
    }
    (PROJECT_ROOT / "outputs" / "state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    print(json.dumps({"verdict": verdict["verdict"], "branch": verdict["branch"], "summary": verdict["summary"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
