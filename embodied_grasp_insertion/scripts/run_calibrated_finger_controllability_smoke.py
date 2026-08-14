#!/usr/bin/env python3
"""P0-C1 calibrated finger controllability matched smoke (no training)."""

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

from embodied_grasp_insertion.physics.grasp_metrics import (  # noqa: E402
    compute_step_metrics,
    control_dt_seconds,
    object_in_hand_pose,
    peg_hand_contact_counts,
    summarize_rollout_metrics_v2,
)
from embodied_grasp_insertion.simulation.calibrated_interventions import (  # noqa: E402
    LEFT_FINGER_IDX,
    RIGHT_FINGER_IDX,
    WRIST_IDX,
    assert_left_fingers_zero,
    build_calibrated_right_offset,
    build_right_demo_replay_actions,
    load_semantics,
    target_offset_to_pulse_actions,
)
from embodied_grasp_insertion.simulation.full_episode_snapshot import (  # noqa: E402
    FullEpisodeSnapshot,
)
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    action_wrist_equal,
    build_wrist_sequence,
    load_yaml,
    make_full_env,
    replay_demo_to_frame,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pad44(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).ravel()
    if a.size >= 44:
        return a[:44].copy()
    out = np.zeros(44, dtype=np.float64)
    out[: a.size] = a
    return out


def _verdict(branches: list[dict[str, Any]]) -> dict[str, Any]:
    by_root: dict[str, list[dict[str, Any]]] = {}
    for b in branches:
        if not b.get("fairness_passed"):
            continue
        # Only unstable roots count for promising.
        if b.get("root_kind") != "unstable":
            continue
        by_root.setdefault(f"{b['context']}::{b['root_id']}", []).append(b)

    if any(not b.get("fairness_passed") for b in branches):
        return {
            "label": "infrastructure_fail",
            "allow_extended_controllability_p0": False,
            "reason": "fairness failure",
        }
    if not by_root:
        return {
            "label": "infrastructure_fail",
            "allow_extended_controllability_p0": False,
            "reason": "no valid unstable matched roots",
        }

    stabilizing = []
    harmful = []
    effect = []
    for rid, blist in by_root.items():
        hold = next((x for x in blist if x["intervention"] == "hold"), None)
        rand = next((x for x in blist if x["intervention"] == "random_matched"), None)
        if hold is None:
            continue
        hs = hold["metrics"]
        improved = []
        worsened = []
        any_eff = False
        for x in blist:
            if x["intervention"] == "hold":
                continue
            ms = x["metrics"]
            # Independent thresholds (no mixed-unit sum).
            better_drift = ms["trans_drift_max_m"] < hs["trans_drift_max_m"] - 1e-3 or (
                ms["rot_drift_max_rad"] < hs["rot_drift_max_rad"] - 5e-3
            )
            no_drift_regression = (
                ms["trans_drift_max_m"] <= hs["trans_drift_max_m"] + 1e-3
                and ms["rot_drift_max_rad"] <= hs["rot_drift_max_rad"] + 5e-3
            )
            better_ret = (
                ms["contact_retention_vs_root_mean"] > hs["contact_retention_vs_root_mean"] + 0.05
            )
            better_loss = ms["contact_loss_steps"] < hs["contact_loss_steps"]
            peg_not_worse = bool(ms["terminal_peg_ok"]) >= bool(hs["terminal_peg_ok"])
            drop_not_worse = bool(ms["object_dropped_proxy"]) <= bool(hs["object_dropped_proxy"])

            worse = (
                (not ms["terminal_peg_ok"] and hs["terminal_peg_ok"])
                or (ms["object_dropped_proxy"] and not hs["object_dropped_proxy"])
                or (
                    ms["trans_drift_max_m"] > hs["trans_drift_max_m"] + 1e-3
                    and ms["contact_retention_vs_root_mean"] + 0.05
                    < hs["contact_retention_vs_root_mean"]
                )
            )
            delta = (
                abs(ms["trans_drift_max_m"] - hs["trans_drift_max_m"])
                + abs(ms["contact_retention_vs_root_mean"] - hs["contact_retention_vs_root_mean"])
            )
            if delta > 1e-4 or ms["terminal_peg_ok"] != hs["terminal_peg_ok"]:
                any_eff = True

            random_beats_hold = False
            if rand is not None:
                rs = rand["metrics"]
                random_beats_hold = (
                    (
                        rs["trans_drift_max_m"] < hs["trans_drift_max_m"] - 1e-3
                        and rs["rot_drift_max_rad"] <= hs["rot_drift_max_rad"] + 5e-3
                    )
                    or (
                        rs["rot_drift_max_rad"] < hs["rot_drift_max_rad"] - 5e-3
                        and rs["trans_drift_max_m"] <= hs["trans_drift_max_m"] + 1e-3
                    )
                    or rs["contact_retention_vs_root_mean"]
                    > hs["contact_retention_vs_root_mean"] + 0.05
                    or rs["contact_loss_steps"] < hs["contact_loss_steps"]
                )

            better = (
                peg_not_worse
                and drop_not_worse
                and no_drift_regression
                and (better_drift or better_ret or better_loss)
                and not (random_beats_hold and x["intervention"] != "random_matched")
            )
            if better:
                improved.append(x["intervention"])
            if worse:
                worsened.append(x["intervention"])
        if any_eff:
            effect.append(rid)
        if improved:
            stabilizing.append({"root_id": rid, "improved": improved})
        if worsened and not improved:
            harmful.append({"root_id": rid, "worsened": worsened})

    n_roots = len({rid.split("::", 1)[-1] for rid in by_root})
    # Count unique physical roots (ignore context) per intervention.
    from collections import defaultdict

    inter_roots: dict[str, set[str]] = defaultdict(set)
    phys_to_eps: dict[str, int] = {}
    for b in branches:
        if b.get("root_kind") != "unstable":
            continue
        rid = str(b.get("root_id", ""))
        phys = rid.split("::")[-1] if "::" in rid else rid
        if "episode_index" in b:
            phys_to_eps[phys] = int(b["episode_index"])
    for s in stabilizing:
        phys = s["root_id"].split("::", 1)[-1]
        for it in s["improved"]:
            if it == "random_matched":
                continue
            inter_roots[it].add(phys)
    inter_counts = {k: len(v) for k, v in inter_roots.items()}
    best_inter, best_n = (
        max(inter_counts.items(), key=lambda kv: kv[1]) if inter_counts else (None, 0)
    )
    best_eps = (
        len({phys_to_eps[p] for p in inter_roots[best_inter] if p in phys_to_eps})
        if best_inter
        else 0
    )

    if best_n >= 3 and n_roots >= 4 and best_eps >= 3:
        return {
            "label": "promising",
            "allow_extended_controllability_p0": True,
            "reason": (
                f"{best_inter} improved {best_n} unstable physical roots "
                f"across {best_eps} episodes (screened_roots={n_roots})"
            ),
            "stabilizing": stabilizing,
        }
    if len(stabilizing) == 0 and len(harmful) >= 2 and len(effect) >= 2:
        return {
            "label": "harmful_only",
            "allow_extended_controllability_p0": False,
            "reason": "calibrated interventions change grasp metrics but do not stabilize vs hold/random",
            "harmful": harmful,
        }
    if len(effect) == 0:
        return {
            "label": "no_effect",
            "allow_extended_controllability_p0": False,
            "reason": "no repeatable calibrated finger effect on unstable roots",
        }
    return {
        "label": "no_effect",
        "allow_extended_controllability_p0": False,
        "reason": (
            f"effects present but below promising bar "
            f"(best={best_inter}:{best_n}roots/{best_eps}eps, "
            f"screened_roots={n_roots}, stabilizing_rows={len(stabilizing)})"
        ),
        "stabilizing": stabilizing,
        "harmful": harmful,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs/finger_controllability_calibrated_smoke.yaml"),
    )
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))

    sem_path = PROJECT_ROOT / cfg["semantics_manifest"]
    sem = json.loads(sem_path.read_text(encoding="utf-8"))
    if not sem.get("summary", {}).get("calibration_pass"):
        raise SystemExit("calibration_fail: refusing to run P0-C1 smoke")
    semantics = load_semantics(sem)

    roots_path = PROJECT_ROOT / cfg["unstable_roots_manifest"]
    roots_man = json.loads(roots_path.read_text(encoding="utf-8"))
    root_list = list(roots_man.get("unstable_roots") or [])[
        : int(cfg.get("max_unstable_roots_for_smoke", 8))
    ] + list(roots_man.get("stable_control_roots") or [])[
        : int(cfg.get("max_stable_controls_for_smoke", 3))
    ]
    if len(roots_man.get("unstable_roots") or []) < 1:
        raise SystemExit("no unstable roots to test")

    horizon = int(cfg["horizon"])
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    sidecar = Path(cfg["sidecar_dir"])
    mild = _pad44(cfg.get("mild_transport_delta44", np.zeros(44)))
    out_dir = PROJECT_ROOT / cfg["output_dir"]
    branch_dir = out_dir / "branches"
    branch_dir.mkdir(parents=True, exist_ok=True)

    # busy
    state_path = PROJECT_ROOT / "outputs" / "state.json"
    if state_path.exists():
        st = json.loads(state_path.read_text())
        st["busy"] = True
        st["phase"] = "finger_controllability_calibrated_smoke"
        state_path.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n")

    branches: list[dict[str, Any]] = []
    # Group roots by episode for env reuse.
    by_ep: dict[int, list[dict[str, Any]]] = {}
    for r in root_list:
        by_ep.setdefault(int(r["episode_index"]), []).append(r)

    for ep, roots in by_ep.items():
        env = make_full_env([ep], sidecar_dir=sidecar, seed=seed)
        try:
            dt = control_dt_seconds(env)
            for root in roots:
                kind = (
                    "unstable"
                    if root in (roots_man.get("unstable_roots") or [])
                    else "stable_control"
                )
                # membership by frame id safer:
                kind = (
                    "unstable"
                    if any(
                        int(u["episode_index"]) == ep and int(u["frame"]) == int(root["frame"])
                        for u in (roots_man.get("unstable_roots") or [])
                    )
                    else "stable_control"
                )
                env.reset(episode_index=ep)
                replay_demo_to_frame(env, int(root["frame"]))
                snap = FullEpisodeSnapshot.capture(env)
                root_o2h = object_in_hand_pose(env._raw)
                root_contact = peg_hand_contact_counts(env._raw)
                root_z = float(
                    env._raw._data.xpos[env._raw._model.body("industreal_round_peg_8mm").id][2]
                )
                root_id = f"ep{ep:03d}_f{int(root['frame']):04d}_{root['phase']}"

                for context in cfg["contexts"]:
                    if context == "wrist_hold":
                        wrist_seq = build_wrist_sequence(
                            source="hold", horizon=horizon, mild_transport_delta=mild
                        )
                    else:
                        wrist_seq = build_wrist_sequence(
                            source="mild_transport",
                            horizon=horizon,
                            mild_transport_delta=mild,
                        )

                    # Reference restore for fairness.
                    snap.restore(env)
                    q0 = np.asarray(env._raw._data.qpos, dtype=np.float64).copy()
                    obs0 = np.asarray(env._obs(), dtype=np.float64).copy()
                    c0 = int(peg_hand_contact_counts(env._raw).total)

                    close_low_off = build_calibrated_right_offset(
                        semantics,
                        mode="calibrated_close_low",
                        low_rad=float(cfg["close_low_rad"]),
                        medium_rad=float(cfg["close_medium_rad"]),
                        rng=rng,
                    )
                    close_low_l2 = float(np.linalg.norm(close_low_off))

                    for intervention in cfg["interventions"]:
                        snap.restore(env)
                        if intervention == "hold":
                            actions = wrist_seq.copy()
                            meta = {"realized_l2": 0.0, "realized_right_offset_rad": [0.0] * 16}
                        elif intervention == "right_demo_replay":
                            actions, meta = build_right_demo_replay_actions(
                                env,
                                root_frame=int(root["frame"]),
                                horizon=horizon,
                                wrist_seq=wrist_seq,
                            )
                        elif intervention == "random_matched":
                            off = build_calibrated_right_offset(
                                semantics,
                                mode="random_matched",
                                low_rad=float(cfg["close_low_rad"]),
                                medium_rad=float(cfg["close_medium_rad"]),
                                rng=rng,
                            )
                            # Match L2 to close_low.
                            n = float(np.linalg.norm(off)) + 1e-12
                            off = off / n * close_low_l2
                            actions, meta = target_offset_to_pulse_actions(
                                env,
                                right_offset_rad=off,
                                horizon=horizon,
                                pulse_steps=int(cfg["pulse_steps"]),
                                wrist_seq=wrist_seq,
                            )
                        else:
                            mode = intervention
                            off = build_calibrated_right_offset(
                                semantics,
                                mode=mode,
                                low_rad=float(cfg["close_low_rad"]),
                                medium_rad=float(cfg["close_medium_rad"]),
                                rng=rng,
                            )
                            actions, meta = target_offset_to_pulse_actions(
                                env,
                                right_offset_rad=off,
                                horizon=horizon,
                                pulse_steps=int(cfg["pulse_steps"]),
                                wrist_seq=wrist_seq,
                            )

                        assert_left_fingers_zero(actions)
                        # Fairness checks
                        q1 = np.asarray(env._raw._data.qpos, dtype=np.float64)
                        obs1 = np.asarray(env._obs(), dtype=np.float64)
                        c1 = int(peg_hand_contact_counts(env._raw).total)
                        fair = (
                            float(np.max(np.abs(q1 - q0))) <= float(cfg["fairness"]["init_qpos_atol"])
                            and float(np.max(np.abs(obs1 - obs0)))
                            <= float(cfg["fairness"]["init_obs_atol"])
                            and c1 == c0
                            and action_wrist_equal(actions, wrist_seq, atol=0.0)
                            and np.allclose(actions[:, LEFT_FINGER_IDX], 0.0)
                        )
                        # Budget match for open_low/random vs close_low
                        if intervention in ("calibrated_open_low", "random_matched"):
                            fair = fair and abs(float(meta["realized_l2"]) - close_low_l2) <= float(
                                cfg["fairness"]["budget_l2_atol"]
                            ) + 1e-4

                        # Rollout
                        steps = []
                        prev = root_o2h
                        executed = []
                        term_reason = "horizon_end"
                        for a in actions:
                            if env._done:
                                break
                            _, _, term, trunc, info = env.step(a)
                            m = compute_step_metrics(
                                env, root_o2h=root_o2h, prev_o2h=prev, dt=dt
                            )
                            steps.append(m)
                            prev = m.object_in_hand
                            executed.append(np.asarray(a, dtype=np.float64).copy())
                            if term or trunc:
                                term_reason = info.get("fail_reason") or (
                                    "terminated" if term else "truncated"
                                )
                                break
                        summary = summarize_rollout_metrics_v2(
                            steps,
                            root_o2h=root_o2h,
                            root_contact=root_contact,
                            control_dt_s=dt,
                            root_peg_world_z=root_z,
                        )
                        exec_a = (
                            np.asarray(executed, dtype=np.float64)
                            if executed
                            else np.zeros((0, 44))
                        )
                        if exec_a.shape[0]:
                            fair = fair and action_wrist_equal(
                                actions[: exec_a.shape[0]], exec_a, atol=0.0
                            )

                        branch_id = f"{root_id}__{context}__{intervention}"
                        npz = branch_dir / f"{branch_id}.npz"
                        np.savez_compressed(
                            npz,
                            actions=actions.astype(np.float32),
                            executed=exec_a.astype(np.float32),
                        )
                        branches.append(
                            {
                                "branch_id": branch_id,
                                "root_id": root_id,
                                "root_kind": kind,
                                "episode_index": ep,
                                "root_frame": int(root["frame"]),
                                "root_phase": root["phase"],
                                "context": context,
                                "intervention": intervention,
                                "seed": seed,
                                "horizon": horizon,
                                "n_steps_executed": len(steps),
                                "wrist_action_source": context,
                                "finger_action_source": intervention,
                                "action_meta": meta,
                                "metrics": summary,
                                "termination_reason": term_reason,
                                "fairness_passed": bool(fair),
                                "output_path": str(npz),
                            }
                        )
        finally:
            env.close()

    verdict = _verdict(branches)
    summary = {
        "created_at": _utc(),
        "verdict": verdict,
        "n_branches": len(branches),
        "fairness_pass_rate": float(np.mean([b["fairness_passed"] for b in branches]))
        if branches
        else 0.0,
        "n_unstable_roots": len(roots_man.get("unstable_roots") or []),
        "n_stable_controls": len(roots_man.get("stable_control_roots") or []),
        "contexts": list(cfg["contexts"]),
        "interventions": list(cfg["interventions"]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    man = {
        "name": "finger_controllability_calibrated_smoke_v1",
        "created_at": summary["created_at"],
        "config": cfg,
        "semantics": str(sem_path),
        "unstable_roots_manifest": str(roots_path),
        "branches": branches,
        "verdict": verdict,
    }
    man_path = PROJECT_ROOT / cfg["manifest_path"]
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = [
        "# Finger Controllability Calibrated Smoke (P0-C1)",
        "",
        f"- 日期：{summary['created_at']}",
        f"- 结论：**{verdict['label']}**",
        f"- 扩展 Controllability P0：{verdict.get('allow_extended_controllability_p0')}",
        f"- Observability / Semantic / policy：**仍禁止**",
        f"- fairness：{summary['fairness_pass_rate']}",
        f"- unstable roots：{summary['n_unstable_roots']}；stable controls：{summary['n_stable_controls']}",
        f"- reason：{verdict.get('reason')}",
        "",
        "P0-C0 数据未覆盖。本轮使用校准右手 target-offset pulse + unstable roots。",
        "",
    ]
    (PROJECT_ROOT / cfg["report_path"]).write_text("\n".join(report) + "\n", encoding="utf-8")

    # state update
    st = json.loads(state_path.read_text()) if state_path.exists() else {}
    st["busy"] = False
    st["phase"] = "finger_controllability_calibrated_smoke"
    st["updated_at"] = summary["created_at"]
    st["p0c1"] = {
        "verdict": verdict,
        "calibration_pass": True,
        "allow_extended_controllability_p0": verdict.get("allow_extended_controllability_p0"),
        "allow_observability_p0": False,
        "allow_semantic_p0": False,
        "allow_policy_training": False,
        "summary_path": str(out_dir / "summary.json"),
    }
    hist = st.get("history") or []
    hist.append(
        {
            "date": summary["created_at"][:10],
            "event": "p0c1_calibrated_finger_smoke",
            "verdict": verdict["label"],
        }
    )
    st["history"] = hist
    state_path.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict["label"], **{k: summary[k] for k in ('n_branches','fairness_pass_rate')}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
