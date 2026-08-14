#!/usr/bin/env python3
"""P0-S0.3: target-hole semantics gate smoke.

Checks:
- reset info exposes geometry_family_id + target socket/peg frames
- different families change semantic label vectors
- correct vs wrong target distances discriminate
- labeler / insert geometry resolve without hardcoded 8mm-only names

Does NOT claim policy knows the hole. No collection / no training.
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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEXJOCO_ROOT = PROJECT_ROOT.parent
for _p in (str(PROJECT_ROOT), str(DEXJOCO_ROOT), str(DEXJOCO_ROOT / "dexjoco")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from dexjoco.sim.envs.panda_bimanual_assembly_env import (  # noqa: E402
    PandaBimanualAssemblyGymEnv,
)
from embodied_grasp_insertion.geometry.family_spec import from_dict  # noqa: E402
from embodied_grasp_insertion.geometry.formal_xml_builder import (  # noqa: E402
    write_formal_family_assets,
)
from embodied_grasp_insertion.geometry.target_hole import (  # noqa: E402
    TargetHoleSpec,
    make_family_labeler,
    semantic_target_features,
    wrong_target_offset_pose,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


REQUIRED_INFO_KEYS = (
    "geometry_family_id",
    "target_instance_id",
    "target_socket_body",
    "target_socket_site",
    "target_peg_body",
    "target_socket_site_pose",
    "tip_to_target_socket_m",
)


def smoke_family(spec, *, seed: int = 0) -> dict[str, Any]:
    write_formal_family_assets(spec, overwrite=False)
    env = PandaBimanualAssemblyGymEnv(
        geometry_family=spec.family_id,
        image_obs=False,
        randomize=False,
        hz=0,
        seed=seed,
    )
    try:
        obs, info = env.reset(seed=seed)
    except TypeError:
        out = env.reset()
        obs, info = (out if isinstance(out, tuple) else (out, {}))

    row: dict[str, Any] = {"family_id": spec.family_id}
    missing = [k for k in REQUIRED_INFO_KEYS if k not in info]
    info_ok = len(missing) == 0 and info.get("geometry_family_id") == spec.family_id
    row["info_keys_ok"] = info_ok
    row["missing_info_keys"] = missing
    row["info"] = {k: info.get(k) for k in REQUIRED_INFO_KEYS}

    sock = np.asarray(env.data.site_xpos[env._socket_site_id], dtype=np.float64)
    wrong_pose = wrong_target_offset_pose(sock, offset_m=0.12)
    # Wrong claim must go through semantic_target_features(..., claimed_target=...).
    true_spec = TargetHoleSpec.from_family(spec.family_id)
    wrong_claim = true_spec.with_instance(
        f"{true_spec.target_instance_id}__synthetic_offset_y",
        socket_site=f"{true_spec.socket_site}__synthetic",
    )
    feat_true = semantic_target_features(env, claimed_target=true_spec)
    feat_wrong = semantic_target_features(
        env,
        claimed_target=wrong_claim,
        claimed_socket_pose_xyz=wrong_pose,
    )
    d_true = float(feat_true["tip_to_true_m"])
    d_wrong = float(feat_wrong["tip_to_claimed_m"])
    discriminate_ok = abs(d_wrong - d_true) > 0.05
    row["tip_to_true_m"] = d_true
    row["tip_to_wrong_claimed_m"] = d_wrong
    row["wrong_target_discriminates"] = discriminate_ok
    row["wrong_via_semantic_api"] = True
    row["label_vector"] = feat_true["label_vector"]
    row["claim_matches_env_true"] = bool(feat_true["claim_matches_env"])
    row["claim_matches_env_wrong"] = bool(feat_wrong["claim_matches_env"])
    row["target_instance_id"] = true_spec.target_instance_id
    row["claim_matches_env"] = bool(feat_true["claim_matches_env"])
    # Wrong claim must not match env instance.
    instance_gate = bool(feat_true["claim_matches_env"]) and (not feat_wrong["claim_matches_env"])
    row["instance_identity_ok"] = instance_gate

    # Labeler resolves this family's names (no hardcoded 8mm-only).
    try:
        labeler = make_family_labeler(env, family_id=spec.family_id)
        labeler.reset_reference(env)
        outcome = labeler.compute(env)
        row["labeler_ok"] = True
        row["labeler_family"] = labeler.geometry_family
        row["outcome_at_reset"] = {
            "tray_ok": outcome.tray_ok,
            "peg_ok": outcome.peg_ok,
            "insert_ok": outcome.insert_ok,
        }
        # At table reset, insert_ok should be false (no bottom contact yet).
        row["insert_ok_false_at_reset"] = outcome.insert_ok is False
    except Exception as e:
        row["labeler_ok"] = False
        row["labeler_error"] = str(e)
        row["insert_ok_false_at_reset"] = False

    # Non-default families must resolve to their own body names (not stuck on round_8mm).
    names_ok = env._geom_names.family_id == spec.family_id
    if spec.family_id != "round_8mm":
        names_ok = bool(
            names_ok
            and env._geom_names.peg_body != "industreal_round_peg_8mm"
            and f"{spec.nominal_size_mm}mm" in env._geom_names.peg_body
            and spec.section in env._geom_names.peg_body
        )
    row["names_ok"] = bool(names_ok)
    row["passed"] = bool(
        info_ok
        and discriminate_ok
        and instance_gate
        and row.get("labeler_ok")
        and row.get("insert_ok_false_at_reset")
        and names_ok
        and feat_true["claim_matches_env"]
    )
    env.close()
    return row


def cross_family_negative(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Different families must produce different semantic labels / target poses."""
    if len(rows) < 2:
        return {"passed": False, "reason": "need>=2 families"}
    a, b = rows[0], rows[1]
    la, lb = a.get("label_vector"), b.get("label_vector")
    pose_a = a.get("info", {}).get("target_socket_site_pose")
    pose_b = b.get("info", {}).get("target_socket_site_pose")
    family_diff = a["family_id"] != b["family_id"]
    label_diff = la != lb
    # Poses may coincide by chance after sampling; require family id in label differs.
    ok = bool(family_diff and label_diff and la[0] != lb[0])
    return {
        "passed": ok,
        "family_a": a["family_id"],
        "family_b": b["family_id"],
        "label_a0": None if not la else la[0],
        "label_b0": None if not lb else lb[0],
        "pose_a": pose_a,
        "pose_b": pose_b,
    }


def verdict(rows: list[dict[str, Any]], cross: dict[str, Any]) -> dict[str, Any]:
    n = len(rows)
    n_ok = sum(1 for r in rows if r.get("passed"))
    if n_ok == n and n >= 4 and cross.get("passed"):
        label = "pass"
        reason = f"all {n} families target-hole semantics ok; cross-family neg ok"
    elif n_ok >= 1:
        label = "partial"
        reason = f"family_ok={n_ok}/{n}, cross={cross.get('passed')}"
    else:
        label = "fail"
        reason = f"family_ok={n_ok}/{n}, cross={cross.get('passed')}"
    return {
        "label": label,
        "reason": reason,
        "n_families_passed": n_ok,
        "n_families": n,
        "allow_policy_training": False,
        "allow_full_collection": False,
        "allow_semantic_p0": False,
        "claims_policy_knows_hole": False,
        "next_if_pass": "grasp_stability_gate (still no train/collect)",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families-yaml",
        default=str(PROJECT_ROOT / "configs/geometry_families.yaml"),
    )
    parser.add_argument(
        "--families",
        default="round_8mm,round_16mm,rectangular_8mm,rectangular_16mm",
        help="comma-separated family ids (smoke subset; not all 8 required)",
    )
    parser.add_argument(
        "--out-manifest",
        default=str(PROJECT_ROOT / "data/manifests/target_hole_semantics_smoke_v1.json"),
    )
    parser.add_argument(
        "--out-report",
        default=str(PROJECT_ROOT / "docs/TARGET_HOLE_SEMANTICS_SMOKE.md"),
    )
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.families_yaml).read_text(encoding="utf-8"))
    by_id = {d["family_id"]: from_dict(d) for d in raw["families"]}
    wanted = [x.strip() for x in args.families.split(",") if x.strip()]
    specs = [by_id[fid] for fid in wanted]

    rows = []
    for i, spec in enumerate(specs):
        print(f"[target-hole] {spec.family_id}", flush=True)
        rows.append(smoke_family(spec, seed=i))

    cross = cross_family_negative(rows)
    v = verdict(rows, cross)
    man = {
        "name": "target_hole_semantics_smoke_v1",
        "created_at": _utc(),
        "protocol": "P0-S0.3",
        "verdict": v,
        "note": (
            "Single-target hole metadata/plumbing pass only. "
            "Exposes family/target_instance_id/socket in reset info; wrong-target via "
            "semantic_target_features(claimed_target=...). Does NOT prove policy knows hole."
        ),
        "scope": "single_target_hole_semantics_metadata_plumbing",
        "claims_policy_knows_hole": False,
        "families": rows,
        "cross_family_negative": cross,
    }
    Path(args.out_manifest).write_text(
        json.dumps(_jsonable(man), indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "# Target Hole Semantics Smoke (P0-S0.3)",
        "",
        f"- 日期：{man['created_at']}",
        f"- 结论：**{v['label']}**（单目标孔 metadata/plumbing；非策略知孔）",
        f"- reason：{v['reason']}",
        "- 不声称策略已知目标孔；不采集 / 不训练",
        "",
        "## Families",
    ]
    for r in rows:
        lines.append(
            f"- `{r['family_id']}`: passed={r.get('passed')} "
            f"info={r.get('info_keys_ok')} wrong_disc={r.get('wrong_target_discriminates')} "
            f"labeler={r.get('labeler_ok')} d_true={r.get('tip_to_true_m'):.3f} "
            f"d_wrong={r.get('tip_to_wrong_claimed_m'):.3f}"
        )
    lines += [
        "",
        "## Cross-family negative",
        f"- passed={cross.get('passed')} {cross.get('family_a')} vs {cross.get('family_b')}",
        "",
        "下一步：抓取稳定性门。",
        "",
    ]
    Path(args.out_report).write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": v, "manifest": args.out_manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
