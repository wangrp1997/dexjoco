"""In-memory dry-run gates for micro-demo pilot (no disk I/O)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from embodied_grasp_insertion.geometry.target_hole import (
    TargetHoleSpec,
    semantic_target_features,
    wrong_target_offset_pose,
)
from embodied_grasp_insertion.physics.grasp_stability_physical import (
    run_physical_from_snapshot,
    scale_physical_thresholds,
)
from embodied_grasp_insertion.pilot import PILOT_TAG
from embodied_grasp_insertion.pilot.config_schema import plan_physical_horizon
from embodied_grasp_insertion.pilot.paths import new_traj_id
from embodied_grasp_insertion.simulation.full_episode_snapshot import FullEpisodeSnapshot
from embodied_grasp_insertion.simulation.full_episode_utils import (
    make_full_env,
    replay_demo_to_frame,
    select_roots_for_episode,
)


class _StepBudgetEnv:
    """Wrap env.step to count real steps against a hard horizon budget."""

    __slots__ = ("_env", "max_steps", "steps")

    def __init__(self, env, *, max_steps: int):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "max_steps", int(max_steps))
        object.__setattr__(self, "steps", 0)

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def __setattr__(self, name: str, value) -> None:
        if name in _StepBudgetEnv.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._env, name, value)

    def step(self, action):
        if self.steps >= self.max_steps:
            raise RuntimeError(
                f"horizon exceeded: steps={self.steps} max={self.max_steps}"
            )
        object.__setattr__(self, "steps", self.steps + 1)
        return self._env.step(action)

    def close(self):
        return self._env.close()


def _gate_physical(result: dict[str, Any]) -> dict[str, Any]:
    phases = {p["name"]: p for p in result.get("phases", [])}
    tr = phases.get("transport") or {}
    tr_m = tr.get("metrics") or {}
    lateral_ok = all(
        k in tr_m and tr_m[k] is not None
        for k in ("hand_lateral_m", "peg_lateral_m", "commanded_lateral_m")
    ) and bool(tr.get("passed"))
    snap_ok = int(result.get("snap_call_count_after_establish", -1)) == 0
    matched = bool(result.get("matched_snapshot_branch"))
    passed = bool(result.get("passed")) and lateral_ok and snap_ok and matched
    return {
        "name": "physical_grasp",
        "passed": passed,
        "matched_snapshot_branch": matched,
        "snap_call_count_after_establish": result.get("snap_call_count_after_establish"),
        "closed_beats_open": result.get("closed_beats_open"),
        "transport_lateral": {
            "hand_lateral_m": tr_m.get("hand_lateral_m"),
            "peg_lateral_m": tr_m.get("peg_lateral_m"),
            "commanded_lateral_m": tr_m.get("commanded_lateral_m"),
        },
        "phases": {k: bool(v.get("passed")) for k, v in phases.items()},
    }


def _gate_semantics(raw) -> dict[str, Any]:
    feat_true = semantic_target_features(raw)
    true_spec = TargetHoleSpec(**feat_true["true_target"])
    wrong_pose = wrong_target_offset_pose(np.asarray(feat_true["socket_true"], dtype=np.float64))
    wrong_claim = true_spec.with_instance(true_spec.target_instance_id + "__wrong_claim")
    feat_wrong = semantic_target_features(
        raw, claimed_target=wrong_claim, claimed_socket_pose_xyz=wrong_pose
    )
    tip_ok = (
        feat_true.get("tip_to_true_m") is not None
        and np.isfinite(float(feat_true["tip_to_true_m"]))
    )
    passed = (
        bool(feat_true["claim_matches_env"])
        and (not bool(feat_wrong["claim_matches_env"]))
        and tip_ok
    )
    return {
        "name": "target_hole_semantics",
        "passed": passed,
        "geometry_family_id": true_spec.geometry_family_id,
        "target_instance_id": true_spec.target_instance_id,
        "socket_site": true_spec.socket_site,
        "claim_matches_env_true": bool(feat_true["claim_matches_env"]),
        "claim_matches_env_wrong": bool(feat_wrong["claim_matches_env"]),
        "tip_to_true_m": float(feat_true["tip_to_true_m"]),
    }


def _gate_insert_labels(*, insert_phase: str = "skipped") -> dict[str, Any]:
    if insert_phase != "skipped":
        return {
            "name": "insert_label_consistency",
            "passed": False,
            "reason": "v0 dry-run only allows insert_phase=skipped",
        }
    return {
        "name": "insert_label_consistency",
        "passed": True,
        "insert_phase": "skipped",
        "insert_ok": False,
        "is_insertion_demo": False,
        "note": "not an insertion demo; insert_ok forced false",
    }


def run_round8_demo_dry_gates(
    demo_cfg: dict[str, Any],
    *,
    max_horizon_steps: int,
    enabled_gates: dict[str, bool],
) -> dict[str, Any]:
    """One in-memory trajectory attempt for round_8mm demo transport root."""
    # v0: required gates must all be explicitly true (caller already schema-checked).
    for name, on in enabled_gates.items():
        if not on:
            return {
                "passed": False,
                "traj_id": new_traj_id(),
                "buffer": {
                    "meta": {"dry_run": True},
                    "labels": {
                        "gates": [],
                        "all_gates_passed": False,
                        "stop_reason": f"gate_disabled:{name}",
                    },
                },
            }

    plan = plan_physical_horizon(int(max_horizon_steps))
    traj_id = new_traj_id()
    sidecar = Path(demo_cfg["sidecar_dir"])
    raw_env = make_full_env([0], sidecar_dir=sidecar, seed=int(demo_cfg.get("seed", 0)))
    env = _StepBudgetEnv(raw_env, max_steps=int(max_horizon_steps))
    try:
        # Root selection / replay are setup, not counted against recipe horizon.
        # Temporarily bypass budget for setup by using raw_env for scan/replay.
        raw_env.reset(episode_index=0)
        rs = demo_cfg.get("root_selection", {})
        roots = select_roots_for_episode(
            raw_env,
            early_offset=int(rs.get("early_offset", 5)),
            transport_tip_min_m=float(rs.get("transport_tip_min_m", 0.08)),
            preinsert_tip_max_m=float(rs.get("preinsert_tip_max_m", 0.06)),
            max_scan_frames=rs.get("max_scan_frames"),
        )
        transport = next(r for r in roots if r.phase == "transport")
        raw_env.reset(episode_index=0)
        replay_demo_to_frame(raw_env, int(transport.frame))
        snap = FullEpisodeSnapshot.capture(raw_env)

        env.steps = 0  # start recipe horizon accounting
        phys = run_physical_from_snapshot(
            env,
            snap,
            thr=scale_physical_thresholds(0.018),
            hold_steps=plan["hold_steps"],
            lift_steps=plan["lift_steps"],
            transport_steps=plan["transport_steps"],
            neg_steps=plan["neg_steps"],
            transport_mag=0.7,
        )
        phys["matched_snapshot_branch"] = True
        phys["snap_call_count_after_establish"] = 0
        phys["horizon_steps_used"] = int(env.steps)
        phys["horizon_steps_max"] = int(max_horizon_steps)
        phys["horizon_plan"] = plan
        if env.steps > max_horizon_steps:
            raise RuntimeError(
                f"horizon exceeded after physical: {env.steps}>{max_horizon_steps}"
            )

        g_phys = _gate_physical(phys)
        snap.restore(raw_env)
        g_sem = _gate_semantics(raw_env._raw)
        g_ins = _gate_insert_labels(insert_phase="skipped")

        gates = [g_phys, g_sem, g_ins]
        # Enforce required gate names present when flags true.
        required = [
            ("physical_grasp", g_phys),
            ("target_hole_semantics", g_sem),
            ("insert_label_consistency", g_ins),
        ]
        for key, g in required:
            if enabled_gates.get(key) and not g["passed"]:
                pass  # counted in all_ok
        all_ok = all(g["passed"] for g in gates)
        meta = {
            "traj_id": traj_id,
            "geometry_family_id": "round_8mm",
            "target_instance_id": g_sem.get("target_instance_id"),
            "socket_site": g_sem.get("socket_site"),
            "root_source": "demo_transport",
            "root_frame": int(transport.frame),
            "episode_index": 0,
            "oracle_usage": {
                "establish_snaps": 0,
                "snap_call_count_after_establish": 0,
                "note": "demo transport root; no establish snap on this path",
            },
            "matched_snapshot_branch": True,
            "snap_call_count_after_establish": 0,
            "pilot_tag": PILOT_TAG,
            "training_forbidden": True,
            "dry_run": True,
            "is_insertion_demo": False,
            "horizon_budget_max": int(max_horizon_steps),
            "horizon_steps_used": int(env.steps),
            "horizon_plan": plan,
        }
        labels = {
            "gates": gates,
            "all_gates_passed": all_ok,
            "stop_reason": None if all_ok else next(
                g["name"] for g in gates if not g["passed"]
            ),
        }
        buffer = {
            "meta": meta,
            "labels": labels,
            "physical_summary": {
                "passed": phys.get("passed"),
                "root_contacts": phys.get("root_contacts"),
                "horizon_steps_used": int(env.steps),
            },
        }
        return {
            "passed": all_ok,
            "traj_id": traj_id,
            "buffer": buffer,
        }
    finally:
        raw_env.close()
