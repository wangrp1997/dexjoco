#!/usr/bin/env python3
"""Audit Allegro right-finger action sign → physical flexion (P0-C1)."""

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
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.physics.grasp_metrics import peg_hand_contact_counts  # noqa: E402
from embodied_grasp_insertion.simulation.calibrated_interventions import (  # noqa: E402
    FINGER_TIP_FOR_JOINT,
    RIGHT_FINGER_ACTUATOR_NAMES,
    RIGHT_FINGER_IDX,
    RIGHT_FINGER_JOINT_NAMES,
)
from embodied_grasp_insertion.simulation.full_episode_snapshot import (  # noqa: E402
    FullEpisodeSnapshot,
)
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    make_full_env,
    replay_demo_to_frame,
    select_roots_for_episode,
)
from interaction_retarget.constants import PEG_BODY  # noqa: E402


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _effective_tip_xpos(raw, tip_body: str) -> np.ndarray:
    """World position for tip metrics.

    Allegro tip bodies often sit on the distal joint origin (zero local offset), so
    distal hinge rotation does not move tip xpos. Use a small offset along the tip
    body's local +z when tip coincides with its parent origin.
    """
    model, data = raw._model, raw._data
    tip_id = int(model.body(tip_body).id)
    tip = np.asarray(data.xpos[tip_id], dtype=np.float64).copy()
    parent = int(model.body_parentid[tip_id])
    if parent >= 0:
        parent_pos = np.asarray(data.xpos[parent], dtype=np.float64)
        if float(np.linalg.norm(tip - parent_pos)) < 1e-6:
            R = np.asarray(data.xmat[tip_id], dtype=np.float64).reshape(3, 3)
            tip = tip + R @ np.array([0.0, 0.0, 0.012], dtype=np.float64)
    return tip


def _tip_palm_peg_dists(raw, tip_body: str) -> tuple[float, float]:
    model, data = raw._model, raw._data
    tip = _effective_tip_xpos(raw, tip_body)
    palm = np.asarray(data.xpos[model.body("allegro_palm_right").id], dtype=np.float64)
    peg = np.asarray(data.xpos[model.body(PEG_BODY).id], dtype=np.float64)
    return float(np.linalg.norm(tip - palm)), float(np.linalg.norm(tip - peg))


def _joint_qpos(raw, joint_name: str) -> float:
    jid = int(raw._model.joint(joint_name).id)
    adr = int(raw._model.jnt_qposadr[jid])
    return float(raw._data.qpos[adr])


def _at_limit(raw, joint_name: str, q: float, eps: float = 1e-3) -> bool:
    jid = int(raw._model.joint(joint_name).id)
    if not raw._model.jnt_limited[jid]:
        return False
    lo, hi = raw._model.jnt_range[jid]
    return bool(q <= float(lo) + eps or q >= float(hi) - eps)


def _tip_in_palm(raw, tip_body: str) -> np.ndarray:
    model, data = raw._model, raw._data
    tip = _effective_tip_xpos(raw, tip_body)
    palm_id = int(model.body("allegro_palm_right").id)
    palm = np.asarray(data.xpos[palm_id], dtype=np.float64)
    R = np.asarray(data.xmat[palm_id], dtype=np.float64).reshape(3, 3)
    return R.T @ (tip - palm)


def _finger_curl_angle(raw, joint_name: str) -> float | None:
    """Angle at medial/distal hinge approximating finger curl (radians)."""
    # joint *j3 -> tip vs medial; *j2 -> distal vs proximal; *j1 -> medial vs proximal
    prefix = joint_name[:2]  # ff/mf/rf/th
    side = "right"
    model, data = raw._model, raw._data
    try:
        if joint_name[2] == "j" and joint_name[3] == "3":
            # Distal joint: tip is fixed on distal link, so medial-distal-tip angle is rigid.
            # Use proximal-medial-tip chain angle which changes when distal curls.
            b0 = f"{prefix}_proximal_{side}"
            b1 = f"{prefix}_medial_{side}"
            b2 = f"{prefix}_tip_{side}"
        elif joint_name[2] == "j" and joint_name[3] == "2":
            b0 = f"{prefix}_proximal_{side}"
            b1 = f"{prefix}_medial_{side}"
            b2 = f"{prefix}_distal_{side}"
        elif joint_name[2] == "j" and joint_name[3] == "1":
            b0 = f"{prefix}_proximal_{side}"
            b1 = f"{prefix}_medial_{side}"
            b2 = f"{prefix}_distal_{side}"
        else:
            return None
        p0 = np.asarray(data.xpos[model.body(b0).id], dtype=np.float64)
        p1 = np.asarray(data.xpos[model.body(b1).id], dtype=np.float64)
        if joint_name[2] == "j" and joint_name[3] == "3":
            p2 = _effective_tip_xpos(raw, b2)
        else:
            p2 = np.asarray(data.xpos[model.body(b2).id], dtype=np.float64)
    except Exception:
        return None
    v1 = p0 - p1
    v2 = p2 - p1
    n1 = np.linalg.norm(v1) + 1e-12
    n2 = np.linalg.norm(v2) + 1e-12
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.arccos(c))


def _pulse_one(
    env,
    snap: FullEpisodeSnapshot,
    *,
    joint_index: int,
    sign: int,
    pulse_norm: float,
    hold_steps: int,
    settle_steps: int = 2,
) -> dict[str, Any]:
    joint_name = RIGHT_FINGER_JOINT_NAMES[joint_index]
    tip = FINGER_TIP_FOR_JOINT[joint_name]
    snap.restore(env)
    # Settle with zero actions so residual contact dynamics do not bias baseline.
    for _ in range(int(settle_steps)):
        if env._done:
            break
        env.step(np.zeros(44))
    if env._done:
        return {
            "sign": int(sign),
            "error": "done_during_settle",
            "qpos_delta_rad": 0.0,
            "tip_palm_delta_m": 0.0,
            "tip_peg_delta_m": 0.0,
            "tip_palm_frame_delta": [0.0, 0.0, 0.0],
            "contact_total_delta": 0,
            "hit_limit_end": False,
            "terminated_early": True,
            "target_delta_rad": 0.0,
            "pulse_norm": float(pulse_norm),
            "qpos_start": _joint_qpos(env._raw, joint_name),
            "qpos_end": _joint_qpos(env._raw, joint_name),
        }

    q0 = _joint_qpos(env._raw, joint_name)
    tip_palm0, tip_peg0 = _tip_palm_peg_dists(env._raw, tip)
    tip_local0 = _tip_in_palm(env._raw, tip)
    curl0 = _finger_curl_angle(env._raw, joint_name)
    hold0 = float(env._hold44[RIGHT_FINGER_IDX[joint_index]])
    contact0 = peg_hand_contact_counts(env._raw).total

    action = np.zeros(44, dtype=np.float64)
    action[RIGHT_FINGER_IDX[joint_index]] = float(sign) * float(pulse_norm)
    seq = [action] + [np.zeros(44) for _ in range(int(hold_steps))]
    infos = []
    for a in seq:
        if env._done:
            break
        obs, rew, term, trunc, info = env.step(a)
        infos.append(
            {
                "terminated": bool(term),
                "truncated": bool(trunc),
                "peg_ok": bool(info.get("peg_ok")),
            }
        )
        if term or trunc:
            break

    q1 = _joint_qpos(env._raw, joint_name)
    tip_palm1, tip_peg1 = _tip_palm_peg_dists(env._raw, tip)
    tip_local1 = _tip_in_palm(env._raw, tip)
    curl1 = _finger_curl_angle(env._raw, joint_name)
    hold1 = float(env._hold44[RIGHT_FINGER_IDX[joint_index]])
    contact1 = peg_hand_contact_counts(env._raw).total
    return {
        "sign": int(sign),
        "pulse_norm": float(pulse_norm),
        "target_delta_rad": hold1 - hold0,
        "qpos_delta_rad": q1 - q0,
        "tip_palm_delta_m": tip_palm1 - tip_palm0,
        "tip_peg_delta_m": tip_peg1 - tip_peg0,
        "tip_palm_frame_delta": (tip_local1 - tip_local0).tolist(),
        "curl_angle_delta_rad": (
            None if curl0 is None or curl1 is None else float(curl1 - curl0)
        ),
        "contact_total_delta": int(contact1 - contact0),
        "hit_limit_end": _at_limit(env._raw, joint_name, q1),
        "qpos_start": q0,
        "qpos_end": q1,
        "terminated_early": bool(infos and (infos[-1]["terminated"] or infos[-1]["truncated"])),
    }


def _classify_joint(pos: dict[str, Any], neg: dict[str, Any]) -> dict[str, Any]:
    """Infer flexion direction from tip-palm (primary) and tip-peg / palm-frame (secondary)."""
    if pos.get("error") or neg.get("error"):
        return {
            "flexion_direction": None,
            "extension_direction": None,
            "confidence": "fail",
            "ambiguous_reason": f"pulse_error:{pos.get('error')},{neg.get('error')}",
            "positive_effect": pos,
            "negative_effect": neg,
        }

    d_palm_pos = float(pos["tip_palm_delta_m"])
    d_palm_neg = float(neg["tip_palm_delta_m"])
    d_peg_pos = float(pos["tip_peg_delta_m"])
    d_peg_neg = float(neg["tip_peg_delta_m"])
    q_pos = float(pos["qpos_delta_rad"])
    q_neg = float(neg["qpos_delta_rad"])
    # Palm-frame tip radius change (xy magnitude): flexion usually pulls tip inward.
    r0_pos = float(np.linalg.norm(pos["tip_palm_frame_delta"][:2]))
    # Use signed change in distance-to-palm-origin in palm frame.
    # Recompute from deltas approximately via tip_palm metric already; add z approach.
    z_pos = float(pos["tip_palm_frame_delta"][2])
    z_neg = float(neg["tip_palm_frame_delta"][2])

    if abs(q_pos) < 1e-4 and abs(q_neg) < 1e-4:
        return {
            "flexion_direction": None,
            "extension_direction": None,
            "confidence": "fail",
            "ambiguous_reason": "qpos_unchanged_under_pulse",
            "positive_effect": pos,
            "negative_effect": neg,
        }
    # Require opposite-signed qpos motion with at least one side meaningful.
    if q_pos * q_neg > 0 and min(abs(q_pos), abs(q_neg)) > 5e-4:
        return {
            "flexion_direction": None,
            "extension_direction": None,
            "confidence": "fail",
            "ambiguous_reason": "qpos_same_sign_for_opposite_pulses",
            "positive_effect": pos,
            "negative_effect": neg,
        }

    palm_margin = 5e-5
    peg_margin = 5e-5
    z_margin = 5e-5

    palm_better_pos = d_palm_pos < d_palm_neg - palm_margin
    palm_better_neg = d_palm_neg < d_palm_pos - palm_margin
    peg_better_pos = d_peg_pos < d_peg_neg - peg_margin
    peg_better_neg = d_peg_neg < d_peg_pos - peg_margin
    # More negative palm-local z ≈ tip moves toward palm underside / grasp for many joints.
    z_better_pos = z_pos < z_neg - z_margin
    z_better_neg = z_neg < z_pos - z_margin

    flex = None
    conf = "low"
    reason = None
    if palm_better_pos and not palm_better_neg:
        flex = "positive"
        conf = "high" if abs(d_palm_pos - d_palm_neg) > 2e-4 else "medium"
    elif palm_better_neg and not palm_better_pos:
        flex = "negative"
        conf = "high" if abs(d_palm_pos - d_palm_neg) > 2e-4 else "medium"
    elif peg_better_pos and not peg_better_neg:
        flex = "positive"
        conf = "medium"
        reason = "used_tip_peg_because_tip_palm_ambiguous"
    elif peg_better_neg and not peg_better_pos:
        flex = "negative"
        conf = "medium"
        reason = "used_tip_peg_because_tip_palm_ambiguous"
    elif z_better_pos and not z_better_neg:
        flex = "positive"
        conf = "medium"
        reason = "used_palm_frame_z_because_tip_metrics_ambiguous"
    elif z_better_neg and not z_better_pos:
        flex = "negative"
        conf = "medium"
        reason = "used_palm_frame_z_because_tip_metrics_ambiguous"
    else:
        # Differential tip effect (+ minus -) to cancel common-mode settle drift.
        d_palm = d_palm_pos - d_palm_neg
        d_peg = d_peg_pos - d_peg_neg
        d_z = z_pos - z_neg
        c_pos = pos.get("curl_angle_delta_rad")
        c_neg = neg.get("curl_angle_delta_rad")
        d_curl = None if c_pos is None or c_neg is None else float(c_pos - c_neg)
        if abs(d_palm) > 2e-6:
            flex = "positive" if d_palm < 0 else "negative"
            conf = "low"
            reason = "used_differential_tip_palm_cancel_common_mode"
        elif abs(d_peg) > 2e-6:
            flex = "positive" if d_peg < 0 else "negative"
            conf = "low"
            reason = "used_differential_tip_peg_cancel_common_mode"
        elif d_curl is not None and abs(d_curl) > 2e-6:
            flex = "positive" if d_curl > 0 else "negative"
            conf = "low"
            reason = "used_differential_curl_cancel_common_mode"
        elif abs(d_z) > 2e-6:
            flex = "positive" if d_z < 0 else "negative"
            conf = "low"
            reason = "used_differential_palm_z_cancel_common_mode"
        else:
            return {
                "flexion_direction": None,
                "extension_direction": None,
                "confidence": "fail",
                "ambiguous_reason": "tip_palm_and_tip_peg_effects_ambiguous",
                "positive_effect": pos,
                "negative_effect": neg,
                "diagnostics": {
                    "d_palm_pos": d_palm_pos,
                    "d_palm_neg": d_palm_neg,
                    "d_peg_pos": d_peg_pos,
                    "d_peg_neg": d_peg_neg,
                    "z_pos": z_pos,
                    "z_neg": z_neg,
                    "curl_pos": c_pos,
                    "curl_neg": c_neg,
                    "diff_palm": d_palm,
                    "diff_peg": d_peg,
                    "diff_curl": d_curl,
                },
            }

    if pos.get("terminated_early") or neg.get("terminated_early"):
        conf = "low"
        reason = (reason or "") + ";episode_terminated_during_pulse"

    if pos.get("hit_limit_end") or neg.get("hit_limit_end"):
        conf = "low" if conf == "high" else conf
        reason = (reason or "") + ";hit_joint_limit"

    return {
        "flexion_direction": flex,
        "extension_direction": "negative" if flex == "positive" else "positive",
        "confidence": conf,
        "ambiguous_reason": reason,
        "positive_effect": pos,
        "negative_effect": neg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--sidecar-dir", type=str, default="/mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly")
    parser.add_argument("--pulse-norm", type=float, default=0.25)
    parser.add_argument("--hold-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-manifest",
        type=str,
        default=str(PROJECT_ROOT / "data/manifests/finger_actuation_semantics_v1.json"),
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(PROJECT_ROOT / "docs/FINGER_ACTUATION_SEMANTICS.md"),
    )
    args = parser.parse_args()

    env = make_full_env([args.episode_id], sidecar_dir=Path(args.sidecar_dir), seed=args.seed)
    snapshots: list[dict[str, Any]] = []
    try:
        env.reset(episode_index=args.episode_id)
        # Snapshot A: shortly after reset (pre-grasp / early), and B: early grasp if available.
        snap_early = FullEpisodeSnapshot.capture(env)
        snapshots.append({"label": "t0_pre_or_start", "frame": int(env._t), "snap": snap_early})

        roots = select_roots_for_episode(env)
        grasp_roots = [r for r in roots if r.phase == "early_grasp"]
        if grasp_roots:
            env.reset(episode_index=args.episode_id)
            replay_demo_to_frame(env, int(grasp_roots[0].frame))
            snap_g = FullEpisodeSnapshot.capture(env)
            snapshots.append(
                {
                    "label": "early_grasp",
                    "frame": int(grasp_roots[0].frame),
                    "snap": snap_g,
                }
            )
        # Prefer grasp snapshot for tip-peg signal; fall back to early.
        primary = snapshots[-1]

        rows = []
        for ji in range(16):
            best = None
            attempts = []
            for snap_info in snapshots:
                for pulse in (args.pulse_norm, min(0.45, args.pulse_norm * 1.4)):
                    pos = _pulse_one(
                        env,
                        snap_info["snap"],
                        joint_index=ji,
                        sign=+1,
                        pulse_norm=pulse,
                        hold_steps=args.hold_steps,
                    )
                    neg = _pulse_one(
                        env,
                        snap_info["snap"],
                        joint_index=ji,
                        sign=-1,
                        pulse_norm=pulse,
                        hold_steps=args.hold_steps,
                    )
                    cls = _classify_joint(pos, neg)
                    attempts.append(
                        {
                            "snapshot": snap_info["label"],
                            "frame": snap_info["frame"],
                            "pulse_norm": pulse,
                            **cls,
                        }
                    )
                    if cls["flexion_direction"] in ("positive", "negative"):
                        rank = {"high": 3, "medium": 2, "low": 1}.get(cls["confidence"], 0)
                        if best is None or rank > best["_rank"]:
                            best = {**cls, "_rank": rank, "snapshot_used": snap_info["label"], "pulse_used": pulse}
                if best is not None and best["_rank"] >= 2:
                    break
            if best is None:
                # Keep last attempt diagnostics.
                last = attempts[-1]
                rows.append(
                    {
                        "joint_index": ji,
                        "actuator_name": RIGHT_FINGER_ACTUATOR_NAMES[ji],
                        "joint_name": RIGHT_FINGER_JOINT_NAMES[ji],
                        "tip_body": FINGER_TIP_FOR_JOINT[RIGHT_FINGER_JOINT_NAMES[ji]],
                        "attempts": attempts,
                        **{k: last[k] for k in last if k not in {"snapshot", "frame", "pulse_norm"}},
                    }
                )
            else:
                best.pop("_rank", None)
                rows.append(
                    {
                        "joint_index": ji,
                        "actuator_name": RIGHT_FINGER_ACTUATOR_NAMES[ji],
                        "joint_name": RIGHT_FINGER_JOINT_NAMES[ji],
                        "tip_body": FINGER_TIP_FOR_JOINT[RIGHT_FINGER_JOINT_NAMES[ji]],
                        "attempts_count": len(attempts),
                        **best,
                    }
                )

        n_ok = sum(1 for r in rows if r["flexion_direction"] in ("positive", "negative"))
        n_high = sum(1 for r in rows if r["confidence"] in ("high", "medium") and r["flexion_direction"])
        # Reliable = all 16 determined with at least medium confidence (not noise-level differentials).
        calibration_pass = n_ok == 16 and n_high == 16

        manifest = {
            "name": "finger_actuation_semantics_v1",
            "created_at": _utc(),
            "episode_id": args.episode_id,
            "primary_snapshot": {
                "label": primary["label"],
                "frame": primary["frame"],
            },
            "pulse_norm": args.pulse_norm,
            "hold_steps": args.hold_steps,
            "finger_scale": float(env.finger_scale),
            "expected_target_delta_rad": float(args.pulse_norm) * float(env.finger_scale),
            "right_joints": rows,
            "left_hand_schema_only": {
                "note": "left actuators exist symmetrically; peg interventions use right only",
                "n_left_finger_action_dims": 16,
            },
            "summary": {
                "n_determined": n_ok,
                "n_high_or_medium": n_high,
                "calibration_pass": calibration_pass,
                "verdict": "pass" if calibration_pass else "calibration_fail",
            },
        }
    finally:
        env.close()

    out = Path(args.output_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Finger Actuation Semantics (P0-C1)",
        "",
        f"- 日期：{manifest['created_at']}",
        f"- episode：{args.episode_id}",
        f"- snapshot：{primary['label']} frame={primary['frame']}",
        f"- pulse_norm={args.pulse_norm} → target Δ≈{manifest['expected_target_delta_rad']:.4f} rad",
        f"- 判定：**{manifest['summary']['verdict']}**（{n_ok}/16 关节确定 flexion）",
        "",
        "| idx | joint | flexion | confidence | tip_palm(+)| tip_palm(-)| qΔ(+)| qΔ(-) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['joint_index']} | `{r['joint_name']}` | {r['flexion_direction']} | "
            f"{r['confidence']} | {r['positive_effect']['tip_palm_delta_m']:.5f} | "
            f"{r['negative_effect']['tip_palm_delta_m']:.5f} | "
            f"{r['positive_effect']['qpos_delta_rad']:.5f} | "
            f"{r['negative_effect']['qpos_delta_rad']:.5f} |"
        )
    lines += [
        "",
        "## 规则",
        "",
        "- 未可靠判断 flexion/extension 时不得构造统一 close。",
        "- 禁止写 qpos / 名称猜符号。",
        "- 左手本轮仅 schema；peg 干预只用右手。",
        "",
    ]
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": manifest["summary"]["verdict"],
                "n_determined": n_ok,
                "manifest": str(out),
            },
            ensure_ascii=False,
        )
    )
    if not calibration_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
