#!/usr/bin/env python3
"""P0-S0.3b: same-family multi-socket instance semantics smoke.

Checks (metadata/plumbing only):
- dual-socket arena has two sites with distinct target_instance_id / socket_site
- poses differ in-scene
- claim_matches_env distinguishes instances even when family_id matches
- wrong claimed instance goes through semantic_target_features(claimed_target=...)
  using the *real* secondary site pose (not synthetic Y-offset)

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
    write_dual_socket_family_assets,
)
from embodied_grasp_insertion.geometry.target_hole import (  # noqa: E402
    TargetHoleSpec,
    claim_matches_env,
    list_same_family_instances,
    semantic_target_features,
    socket_site_pose_xyz,
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


def smoke_family(spec) -> dict[str, Any]:
    assets = write_dual_socket_family_assets(spec, secondary_key="b", overwrite=True)
    primary = TargetHoleSpec.from_family_instance(spec.family_id, "primary")
    secondary = TargetHoleSpec.from_family_instance(spec.family_id, "b")
    instances = list_same_family_instances(spec.family_id, keys=("primary", "b"))

    env = PandaBimanualAssemblyGymEnv(
        geometry_family=spec.family_id,
        xml_path=assets["dual_arena_xml"],
        image_obs=False,
        randomize=False,
        hz=0,
        seed=0,
    )
    try:
        try:
            _, info = env.reset(seed=0)
        except TypeError:
            out = env.reset()
            _, info = (out if isinstance(out, tuple) else (out, {}))

        pose_a = socket_site_pose_xyz(env, primary.socket_site)
        pose_b = socket_site_pose_xyz(env, secondary.socket_site)
        pose_sep = float(np.linalg.norm(pose_a - pose_b))

        same_family = primary.geometry_family_id == secondary.geometry_family_id
        id_distinct = primary.target_instance_id != secondary.target_instance_id
        site_distinct = primary.socket_site != secondary.socket_site
        pose_distinct = pose_sep > 0.05

        true_env = TargetHoleSpec.from_names(env._geom_names)
        match_primary = claim_matches_env(primary, true_env)
        match_secondary = claim_matches_env(secondary, true_env)

        feat_true = semantic_target_features(env, claimed_target=primary)
        feat_wrong = semantic_target_features(env, claimed_target=secondary)
        d_true = float(feat_true["tip_to_claimed_m"])
        d_wrong = float(feat_wrong["tip_to_claimed_m"])
        # Wrong instance must resolve a real site pose and differ from true tip distance.
        wrong_resolved = feat_wrong.get("socket_claimed") is not None
        discriminate = abs(d_wrong - d_true) > 0.03 and (not bool(feat_wrong["claim_matches_env"]))

        info_ok = (
            info.get("geometry_family_id") == spec.family_id
            and info.get("target_instance_id") == primary.target_instance_id
            and info.get("target_socket_site") == primary.socket_site
        )

        passed = bool(
            same_family
            and id_distinct
            and site_distinct
            and pose_distinct
            and match_primary
            and (not match_secondary)
            and wrong_resolved
            and discriminate
            and info_ok
            and len(instances) == 2
        )
        return {
            "family_id": spec.family_id,
            "passed": passed,
            "assets": assets,
            "primary": primary.to_dict(),
            "secondary": secondary.to_dict(),
            "pose_a": pose_a.tolist(),
            "pose_b": pose_b.tolist(),
            "pose_sep_m": pose_sep,
            "same_family": same_family,
            "id_distinct": id_distinct,
            "site_distinct": site_distinct,
            "pose_distinct": pose_distinct,
            "match_primary": match_primary,
            "match_secondary": match_secondary,
            "tip_to_primary_m": d_true,
            "tip_to_secondary_m": d_wrong,
            "wrong_claim_matches": bool(feat_wrong["claim_matches_env"]),
            "wrong_site_resolved": wrong_resolved,
            "info_ok": info_ok,
            "info_target_instance_id": info.get("target_instance_id"),
        }
    finally:
        env.close()


def verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_ok = sum(1 for r in rows if r.get("passed"))
    if n_ok == n and n >= 1:
        label, reason = "pass", f"dual_socket_families_ok={n_ok}/{n}"
    elif n_ok >= 1:
        label, reason = "partial", f"dual_socket_families_ok={n_ok}/{n}"
    else:
        label, reason = "fail", f"dual_socket_families_ok={n_ok}/{n}"
    return {
        "label": label,
        "reason": reason,
        "n_families_passed": n_ok,
        "n_families": n,
        "allow_policy_training": False,
        "allow_full_collection": False,
        "allow_semantic_p0": False,
        "claims_policy_knows_hole": False,
        "scope": "same_family_multi_socket_instance_metadata_plumbing",
        "next_if_pass": "S0.4c multi-family physical grasp roots / still no train-collect",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families-yaml",
        default=str(PROJECT_ROOT / "configs/geometry_families.yaml"),
    )
    parser.add_argument("--families", default="round_8mm,rectangular_8mm")
    parser.add_argument(
        "--out-manifest",
        default=str(PROJECT_ROOT / "data/manifests/target_hole_multi_instance_smoke_v1.json"),
    )
    parser.add_argument(
        "--out-report",
        default=str(PROJECT_ROOT / "docs/TARGET_HOLE_MULTI_INSTANCE_SMOKE.md"),
    )
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.families_yaml).read_text(encoding="utf-8"))
    by_id = {d["family_id"]: from_dict(d) for d in raw["families"]}
    wanted = [x.strip() for x in args.families.split(",") if x.strip()]
    rows = []
    for fid in wanted:
        print(f"[s0.3b] {fid}", flush=True)
        rows.append(smoke_family(by_id[fid]))

    v = verdict(rows)
    man = {
        "name": "target_hole_multi_instance_smoke_v1",
        "created_at": _utc(),
        "protocol": "P0-S0.3b",
        "verdict": v,
        "note": (
            "Same-family dual-socket metadata/plumbing. Distinguishes target_instance_id, "
            "socket_site, and in-scene poses; wrong claim uses real secondary site via "
            "semantic_target_features(claimed_target=...). Not policy hole cognition."
        ),
        "families": rows,
    }
    Path(args.out_manifest).write_text(
        json.dumps(_jsonable(man), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Target Hole Multi-Instance Smoke (P0-S0.3b)",
        "",
        f"- 日期：{man['created_at']}",
        f"- 结论：**{v['label']}**",
        f"- reason：{v['reason']}",
        "- 范围：同 family 双 socket 的 instance_id / site / pose 区分（metadata/plumbing）",
        "- 不声称策略知孔；不采集 / 不训练",
        "",
        "## Families",
    ]
    for r in rows:
        lines.append(
            f"- `{r['family_id']}`: passed={r.get('passed')} "
            f"id_distinct={r.get('id_distinct')} site_distinct={r.get('site_distinct')} "
            f"pose_sep={r.get('pose_sep_m'):.3f}m match_pri={r.get('match_primary')} "
            f"match_sec={r.get('match_secondary')} disc={r.get('wrong_site_resolved')}"
        )
    lines += ["", "下一步：S0.4c 多族物理抓取；仍禁采集/训练。", ""]
    Path(args.out_report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": v, "manifest": args.out_manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
