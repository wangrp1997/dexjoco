#!/usr/bin/env python3
"""P0-C1.1 calibrated finger smoke with feasible matched budgets (no training)."""

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
    assert_left_fingers_zero,
    build_calibrated_right_offset,
    build_right_demo_replay_actions,
    load_semantics,
    project_matched_feasible_offsets,
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
# Import shared verdict helper without package install.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_c1_smoke",
    PROJECT_ROOT / "scripts" / "run_calibrated_finger_controllability_smoke.py",
)
_c1 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_c1)
_verdict = _c1._verdict


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pad44(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).ravel()
    out = np.zeros(44, dtype=np.float64)
    out[: min(44, a.size)] = a[:44]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs/finger_controllability_c1_1.yaml"),
    )
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))

    roots_path = PROJECT_ROOT / cfg["unstable_roots_manifest"]
    roots_man = json.loads(roots_path.read_text(encoding="utf-8"))
    gate = roots_man.get("screening_gate") or {}
    if not gate.get("passed"):
        raise SystemExit(f"screening_fail: refusing smoke ({gate})")

    load_path = PROJECT_ROOT / cfg.get("transport_load_manifest", "")
    load_profile = str(cfg.get("wrist_load_profile", "constant"))
    if load_path.exists():
        load_man = json.loads(load_path.read_text(encoding="utf-8"))
        if not load_man.get("selection_ok"):
            raise SystemExit("screening_fail: transport load selection_ok=false")
        mild = _pad44(load_man["selected_delta44"])
        load_profile = str(load_man.get("selected_profile") or load_profile)
    else:
        mild = _pad44(cfg.get("mild_transport_delta44", np.zeros(44)))

    sem_path = PROJECT_ROOT / cfg["semantics_manifest"]
    sem = json.loads(sem_path.read_text(encoding="utf-8"))
    if not sem.get("summary", {}).get("calibration_pass"):
        raise SystemExit("calibration_fail")
    semantics = load_semantics(sem)

    root_list = list(roots_man.get("unstable_roots") or [])[
        : int(cfg.get("max_unstable_roots_for_smoke", 8))
    ] + list(roots_man.get("stable_control_roots") or [])[
        : int(cfg.get("max_stable_controls_for_smoke", 3))
    ]

    horizon = int(cfg["horizon"])
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    sidecar = Path(cfg["sidecar_dir"])
    out_dir = PROJECT_ROOT / cfg["output_dir"]
    branch_dir = out_dir / "branches"
    branch_dir.mkdir(parents=True, exist_ok=True)
    fair_cfg = cfg.get("fairness", {})
    require_clip0 = bool(fair_cfg.get("require_clip_count_zero", True))

    state_path = PROJECT_ROOT / "outputs" / "state.json"
    if state_path.exists():
        st = json.loads(state_path.read_text())
        st["busy"] = True
        st["phase"] = "finger_controllability_c1_1_smoke"
        st["updated_at"] = _utc()
        state_path.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n")

    branches: list[dict[str, Any]] = []
    by_ep: dict[int, list[dict[str, Any]]] = {}
    for r in root_list:
        by_ep.setdefault(int(r["episode_index"]), []).append(r)

    matched_modes = ("calibrated_close_low", "calibrated_open_low", "random_matched")

    try:
        for ep, roots in by_ep.items():
            env = make_full_env([ep], sidecar_dir=sidecar, seed=seed)
            try:
                dt = control_dt_seconds(env)
                for root in roots:
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
                                profile=load_profile,
                            )

                        snap.restore(env)
                        q0 = np.asarray(env._raw._data.qpos, dtype=np.float64).copy()
                        obs0 = np.asarray(env._obs(), dtype=np.float64).copy()
                        c0 = int(peg_hand_contact_counts(env._raw).total)

                        raw_offsets = {
                            "calibrated_close_low": build_calibrated_right_offset(
                                semantics,
                                mode="calibrated_close_low",
                                low_rad=float(cfg["close_low_rad"]),
                                medium_rad=float(cfg.get("close_medium_rad", 0.04)),
                                rng=rng,
                            ),
                            "calibrated_open_low": build_calibrated_right_offset(
                                semantics,
                                mode="calibrated_open_low",
                                low_rad=float(cfg["open_low_rad"]),
                                medium_rad=float(cfg.get("close_medium_rad", 0.04)),
                                rng=rng,
                            ),
                            "random_matched": build_calibrated_right_offset(
                                semantics,
                                mode="random_matched",
                                low_rad=float(cfg["close_low_rad"]),
                                medium_rad=float(cfg.get("close_medium_rad", 0.04)),
                                rng=rng,
                            ),
                        }
                        projected, proj_meta = project_matched_feasible_offsets(env, raw_offsets)
                        shared_l2 = float(proj_meta["shared_l2"])

                        for intervention in cfg["interventions"]:
                            snap.restore(env)
                            if intervention == "hold":
                                actions = wrist_seq.copy()
                                meta = {
                                    "realized_l2": 0.0,
                                    "realized_right_offset_rad": [0.0] * 16,
                                    "clip_count": 0,
                                }
                            elif intervention == "right_demo_replay":
                                actions, meta = build_right_demo_replay_actions(
                                    env,
                                    root_frame=int(root["frame"]),
                                    horizon=horizon,
                                    wrist_seq=wrist_seq,
                                )
                                meta["clip_count"] = 0
                            elif intervention in matched_modes:
                                actions, meta = target_offset_to_pulse_actions(
                                    env,
                                    right_offset_rad=projected[intervention],
                                    horizon=horizon,
                                    pulse_steps=int(cfg["pulse_steps"]),
                                    wrist_seq=wrist_seq,
                                    allow_clip=False,
                                )
                                meta["budget_projection"] = proj_meta
                            else:
                                raise RuntimeError(f"paused/unknown intervention: {intervention}")

                            assert_left_fingers_zero(actions)
                            q1 = np.asarray(env._raw._data.qpos, dtype=np.float64)
                            obs1 = np.asarray(env._obs(), dtype=np.float64)
                            c1 = int(peg_hand_contact_counts(env._raw).total)
                            fair = (
                                float(np.max(np.abs(q1 - q0)))
                                <= float(fair_cfg.get("init_qpos_atol", 1e-8))
                                and float(np.max(np.abs(obs1 - obs0)))
                                <= float(fair_cfg.get("init_obs_atol", 1e-5))
                                and c1 == c0
                                and action_wrist_equal(actions, wrist_seq, atol=0.0)
                                and np.allclose(actions[:, LEFT_FINGER_IDX], 0.0)
                            )
                            if intervention in matched_modes:
                                fair = fair and abs(float(meta["realized_l2"]) - shared_l2) <= float(
                                    fair_cfg.get("budget_l2_atol", 1e-6)
                                )
                                if require_clip0:
                                    fair = fair and int(meta.get("clip_count", 0)) == 0

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
                                    "unstable_reasons": root.get("unstable_reasons"),
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

        # Strict: any fairness fail => infrastructure_fail via _verdict
        verdict = _verdict(branches)
        if any(not b["fairness_passed"] for b in branches):
            verdict = {
                "label": "infrastructure_fail",
                "allow_extended_controllability_p0": False,
                "reason": "fairness/clip/budget failure under P0-C1.1 strict matching",
            }
        summary = {
            "created_at": _utc(),
            "protocol": "P0-C1.1",
            "verdict": verdict,
            "n_branches": len(branches),
            "fairness_pass_rate": float(np.mean([b["fairness_passed"] for b in branches]))
            if branches
            else 0.0,
            "n_unstable_roots": len(roots_man.get("unstable_roots") or []),
            "n_stable_controls": len(roots_man.get("stable_control_roots") or []),
            "load_scale": (json.loads(load_path.read_text()).get("selected_scale") if load_path.exists() else None),
            "contexts": list(cfg["contexts"]),
            "interventions": list(cfg["interventions"]),
            "allow_extended_controllability_p0": False,
            "allow_observability_p0": False,
            "allow_semantic_p0": False,
            "allow_policy_training": False,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        man = {
            "name": "finger_controllability_calibrated_smoke_v1_1",
            "created_at": summary["created_at"],
            "config": cfg,
            "branches": branches,
            "verdict": verdict,
            "screening_gate": gate,
        }
        man_path = PROJECT_ROOT / cfg["manifest_path"]
        man_path.parent.mkdir(parents=True, exist_ok=True)
        man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report = [
            "# Finger Controllability Calibrated Smoke v1.1 (P0-C1.1)",
            "",
            f"- 日期：{summary['created_at']}",
            f"- 结论：**{verdict['label']}**",
            f"- 扩展 Controllability P0：false",
            f"- Observability / Semantic / policy：仍禁止",
            f"- fairness：{summary['fairness_pass_rate']}",
            f"- unstable/stable：{summary['n_unstable_roots']}/{summary['n_stable_controls']}",
            f"- load_scale：{summary['load_scale']}",
            f"- reason：{verdict.get('reason')}",
            "",
            "未覆盖 P0-C0 / P0-C1 v1；close_medium 已暂停。",
            "",
        ]
        (PROJECT_ROOT / cfg["report_path"]).write_text("\n".join(report) + "\n", encoding="utf-8")
    finally:
        if state_path.exists():
            st = json.loads(state_path.read_text())
            st["busy"] = False
            st["phase"] = "finger_controllability_c1_1"
            st["updated_at"] = _utc()
            st["p0c1_1"] = {
                "verdict": locals().get("verdict", {"label": "interrupted"}),
                "allow_extended_controllability_p0": False,
                "allow_observability_p0": False,
                "allow_semantic_p0": False,
                "allow_policy_training": False,
            }
            # Keep historical P0-C1 as no_effect.
            hist = st.get("history") or []
            hist.append(
                {
                    "date": _utc()[:10],
                    "event": "p0c1_1_smoke",
                    "verdict": (locals().get("verdict") or {}).get("label"),
                }
            )
            st["history"] = hist
            state_path.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps({"verdict": verdict, "summary": str(out_dir / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
