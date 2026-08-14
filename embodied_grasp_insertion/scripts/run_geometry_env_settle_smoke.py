#!/usr/bin/env python3
"""P0-S0.1: per-family 1–2 reset/settle env smoke (no training, no collection)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.geometry.family_spec import from_dict  # noqa: E402
from embodied_grasp_insertion.geometry.labeler_adapter import (  # noqa: E402
    lookup_ids,
    set_free_pose,
    tip_to_socket_site,
)
from embodied_grasp_insertion.geometry.xml_builder import write_family_xml  # noqa: E402


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finite(data: mujoco.MjData) -> bool:
    return bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())


def settle_once(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec,
    *,
    lateral_m: float,
    hover_m: float,
    steps: int,
) -> dict[str, Any]:
    ids = lookup_ids(model, spec)
    # Reset free poses: socket near origin, peg hovering above socket site.
    set_free_pose(
        model, data, ids.socket_joint, np.array([0.0, 0.0, 0.05]), np.array([1.0, 0, 0, 0])
    )
    mujoco.mj_forward(model, data)
    sock = np.asarray(data.site_xpos[ids.socket_site], dtype=np.float64).copy()
    tip_local = float(spec.collision["peg_tip_site_z_m"])
    tip_target = sock + np.array([lateral_m, 0.0, hover_m])
    peg_pos = tip_target - np.array([0.0, 0.0, tip_local])
    set_free_pose(model, data, ids.peg_joint, peg_pos, np.array([1.0, 0, 0, 0]))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    tip0 = tip_to_socket_site(data, ids).copy()
    ncon0 = int(data.ncon)
    exploded = False
    for _ in range(int(steps)):
        mujoco.mj_step(model, data)
        if not _finite(data):
            exploded = True
            break
        if float(np.linalg.norm(data.qvel)) > 50.0:
            exploded = True
            break

    tip1 = tip_to_socket_site(data, ids)
    peg_z = float(data.xpos[ids.peg_body, 2])
    sock_z = float(data.xpos[ids.socket_body, 2])
    ok = (
        not exploded
        and _finite(data)
        and peg_z > -0.05
        and abs(peg_z - sock_z) < 1.0
        and float(np.linalg.norm(tip1[:2])) < 0.25
    )
    return {
        "lateral_m": float(lateral_m),
        "hover_m": float(hover_m),
        "steps": int(steps),
        "exploded": bool(exploded),
        "finite": _finite(data),
        "ncon_start": ncon0,
        "ncon_end": int(data.ncon),
        "tip_delta_start": tip0.tolist(),
        "tip_delta_end": tip1.tolist(),
        "peg_z": peg_z,
        "socket_z": sock_z,
        "passed": bool(ok),
    }


def verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_fam = len(rows)
    n_pass = sum(1 for r in rows if r.get("family_passed"))
    n_trials = sum(len(r.get("trials") or []) for r in rows)
    n_trial_pass = sum(
        1 for r in rows for t in (r.get("trials") or []) if t.get("passed")
    )
    sections = sorted({r["section"] for r in rows if r.get("family_passed")})
    if n_pass >= 6 and "round" in sections and "rectangular" in sections:
        label = "pass"
        reason = f"{n_pass}/{n_fam} families settle-stable; trials {n_trial_pass}/{n_trials}"
    elif n_pass >= 3:
        label = "partial"
        reason = f"only {n_pass}/{n_fam} families stable"
    else:
        label = "fail"
        reason = f"unstable settle: {n_pass}/{n_fam}"
    return {
        "label": label,
        "reason": reason,
        "n_families_passed": n_pass,
        "n_families": n_fam,
        "n_trials_passed": n_trial_pass,
        "n_trials": n_trials,
        "allow_policy_training": False,
        "allow_full_collection": False,
        "allow_semantic_p0": False,
        "allow_formal_arena_edit": label == "pass",
        "next_if_pass": "optional parameterized arena plumbing design (still no mass collection)",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families-yaml",
        default=str(PROJECT_ROOT / "configs/geometry_families.yaml"),
    )
    parser.add_argument(
        "--xml-dir",
        default=str(PROJECT_ROOT / "outputs/geometry_xml_tmp"),
    )
    parser.add_argument("--settle-steps", type=int, default=200)
    parser.add_argument(
        "--out-manifest",
        default=str(PROJECT_ROOT / "data/manifests/geometry_env_settle_smoke_v1.json"),
    )
    parser.add_argument(
        "--out-report",
        default=str(PROJECT_ROOT / "docs/GEOMETRY_ENV_SETTLE_SMOKE.md"),
    )
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.families_yaml).read_text(encoding="utf-8"))
    specs = [from_dict(d) for d in raw["families"]]
    xml_dir = Path(args.xml_dir)
    rows: list[dict[str, Any]] = []

    for spec in specs:
        print(f"[settle] {spec.family_id}", flush=True)
        xml = write_family_xml(spec, xml_dir, tag="matched")
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        # Two resets: centered hover, small lateral hover.
        trials = [
            settle_once(
                model,
                data,
                spec,
                lateral_m=0.0,
                hover_m=0.03,
                steps=args.settle_steps,
            ),
            settle_once(
                model,
                data,
                spec,
                lateral_m=0.005,
                hover_m=0.025,
                steps=args.settle_steps,
            ),
        ]
        fam_ok = all(t["passed"] for t in trials)
        rows.append(
            {
                "family_id": spec.family_id,
                "section": spec.section,
                "xml": str(xml),
                "trials": trials,
                "family_passed": fam_ok,
            }
        )

    v = verdict(rows)
    man = {
        "name": "geometry_env_settle_smoke_v1",
        "created_at": _utc(),
        "protocol": "P0-S0.1",
        "verdict": v,
        "settle_steps": int(args.settle_steps),
        "families": rows,
    }
    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Geometry Env Settle Smoke (P0-S0.1)",
        "",
        f"- 日期：{man['created_at']}",
        f"- 结论：**{v['label']}**",
        f"- reason：{v['reason']}",
        f"- settle_steps：{args.settle_steps}",
        "- 每 family 2 次 reset（居中悬停 / 小侧偏悬停）后短物理 settle",
        "- 训练 / 全量采集 / Semantic P0：**仍禁止**",
        f"- 允许改正式 arena：{v.get('allow_formal_arena_edit')}",
        "",
        "## Families",
    ]
    for r in rows:
        tpass = sum(1 for t in r["trials"] if t["passed"])
        lines.append(
            f"- `{r['family_id']}`: family_passed={r['family_passed']} trials={tpass}/2"
        )
    lines += ["", "仅用临时 XML；未改正式 arena；未采集 demo。", ""]
    Path(args.out_report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": v, "manifest": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
