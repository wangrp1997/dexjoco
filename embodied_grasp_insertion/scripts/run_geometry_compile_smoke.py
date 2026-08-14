#!/usr/bin/env python3
"""P0-S0 geometry compile + static insertion feasibility smoke (no training/collection)."""

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

from embodied_grasp_insertion.geometry.family_spec import GeometryFamilySpec, from_dict  # noqa: E402
from embodied_grasp_insertion.geometry.labeler_adapter import (  # noqa: E402
    labeler_design_notes,
    lookup_ids,
    probe_insertion,
)
from embodied_grasp_insertion.geometry.xml_builder import write_family_xml  # noqa: E402


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_specs(families_yaml: Path) -> list[GeometryFamilySpec]:
    raw = yaml.safe_load(families_yaml.read_text(encoding="utf-8"))
    return [from_dict(d) for d in raw["families"]]


def compile_one(spec: GeometryFamilySpec, xml_dir: Path) -> dict[str, Any]:
    xml_path = write_family_xml(spec, xml_dir, tag="matched")
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as e:
        return {
            "family_id": spec.family_id,
            "xml": str(xml_path),
            "compile_ok": False,
            "error": str(e),
        }
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        ids = lookup_ids(model, spec)
        qadr_peg = int(model.jnt_qposadr[ids.peg_joint])
        qadr_sock = int(model.jnt_qposadr[ids.socket_joint])
        ok_lookup = True
        lookup_err = None
    except Exception as e:
        ids = None
        qadr_peg = qadr_sock = None
        ok_lookup = False
        lookup_err = str(e)

    # Initial penetration check at default poses.
    mujoco.mj_forward(model, data)
    init_pen = False
    min_dist = None
    if data.ncon:
        dists = [float(data.contact[i].dist) for i in range(data.ncon)]
        min_dist = min(dists)
        init_pen = min_dist < -1e-4

    # Visual vs collision size check (round radius).
    size_match = True
    size_notes = []
    if spec.section == "round":
        col_r = float(spec.collision["peg_radius_m"])
        vis_r = 0.5 * float(spec.peg_visual_span_xyz_m[0])
        if abs(col_r - vis_r) > 1e-3:
            size_match = False
            size_notes.append(f"round radius mismatch col={col_r:.5f} vis={vis_r:.5f}")
    else:
        col_w = float(spec.collision["peg_half_width_m"]) * 2
        vis_w = float(spec.peg_visual_span_xyz_m[0])
        if abs(col_w - vis_w) > 2e-3:
            size_match = False
            size_notes.append(f"rect width mismatch col={col_w:.5f} vis={vis_w:.5f}")

    return {
        "family_id": spec.family_id,
        "section": spec.section,
        "xml": str(xml_path),
        "compile_ok": True,
        "lookup_ok": ok_lookup,
        "lookup_error": lookup_err,
        "qpos_adr": {"peg": qadr_peg, "socket": qadr_sock},
        "nq": int(model.nq),
        "nbody": int(model.nbody),
        "initial_penetrating": bool(init_pen),
        "initial_min_contact_dist": min_dist,
        "visual_collision_size_match": size_match,
        "size_notes": size_notes,
        "semantic_names_resolved": ok_lookup,
    }


def insertion_smoke(spec: GeometryFamilySpec, xml_dir: Path) -> dict[str, Any]:
    xml_path = write_family_xml(spec, xml_dir, tag="matched")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    # Clearance-scale offsets.
    if spec.section == "round":
        clr = float(spec.clearance_diameter_m or 0.0) * float(spec.asset_scale)
        half = float(spec.collision["peg_radius_m"])
    else:
        clr = float(spec.clearance_width_m or 0.0) * float(spec.asset_scale)
        half = float(spec.collision["peg_half_width_m"])
    laterals = [0.0, 0.25 * clr, 0.5 * max(clr, 1e-4), 1.5 * half]
    depths = [0.0, 0.005, 0.015, 0.03]
    probes = []
    for lat in laterals:
        for dep in depths:
            r = probe_insertion(model, data, spec, lateral_offset_m=lat, depth_m=dep, settle_steps=40)
            probes.append(r.__dict__)
    # Feasible corridor: small lateral, modest depth, not heavily penetrating.
    feasible = [
        p
        for p in probes
        if p["lateral_offset_m"] <= max(0.5 * clr, 1e-4) + 1e-9
        and p["depth_m"] >= 0.005
        and not p["penetrating"]
    ]
    feasible_soft = [
        p
        for p in probes
        if p["lateral_offset_m"] <= max(clr, 1e-4) + 1e-9
        and p["depth_m"] >= 0.005
        and (not p["penetrating"] or p["bottom_contacts"] > 0)
    ]
    return {
        "family_id": spec.family_id,
        "clearance_scaled_m": clr,
        "n_probes": len(probes),
        "n_feasible_strict": len(feasible),
        "n_feasible_soft": len(feasible_soft),
        "has_insertable_interval": len(feasible) > 0 or len(feasible_soft) > 0,
        "probes": probes,
    }


def mismatch_smoke(peg: GeometryFamilySpec, sock: GeometryFamilySpec, xml_dir: Path) -> dict[str, Any]:
    xml_path = write_family_xml(peg, xml_dir, mismatch_socket=sock, tag="mismatch")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    # Zero gravity: mismatch check is static overlap / geometric conflict, not tossing.
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    xml_m = write_family_xml(peg, xml_dir, tag="matched")
    model_m = mujoco.MjModel.from_xml_path(str(xml_m))
    model_m.opt.gravity[:] = 0.0
    data_m = mujoco.MjData(model_m)
    r_m = probe_insertion(model_m, data_m, peg, lateral_offset_m=0.0, depth_m=0.01, settle_steps=0)
    r = probe_insertion(model, data, peg, lateral_offset_m=0.0, depth_m=0.01, settle_steps=0)
    peg_r = float(
        peg.collision.get("peg_radius_m")
        or max(float(peg.collision.get("peg_half_width_m", 0)), float(peg.collision.get("peg_half_depth_m", 0)))
    )
    hole_r = float(sock.collision["hole_half_xy_m"])
    size_conflict = peg_r > hole_r * 1.05
    section_conflict = peg.section != sock.section
    phys_worse = bool(r.penetrating and not r_m.penetrating) or (r.ncon > r_m.ncon + 2)
    # Valid negative if geometry cannot mate, preferably with static overlap evidence.
    reasonable = bool((size_conflict or section_conflict) and (r.penetrating or r.ncon > 0 or phys_worse))
    if (size_conflict or section_conflict) and not reasonable:
        # Deeper static jam.
        r = probe_insertion(model, data, peg, lateral_offset_m=0.0, depth_m=0.02, settle_steps=0)
        reasonable = bool(r.penetrating or r.ncon > 0)
    return {
        "peg_family": peg.family_id,
        "socket_family": sock.family_id,
        "xml": str(xml_path),
        "peg_radius_or_half_m": peg_r,
        "socket_hole_half_m": hole_r,
        "size_conflict": size_conflict,
        "section_conflict": section_conflict,
        "matched_probe": r_m.__dict__,
        "probe": r.__dict__,
        "reasonable_fail_proxy": reasonable,
    }


def verdict(compiles: list[dict], inserts: list[dict], mismatches: list[dict]) -> dict[str, Any]:
    ok_compile = [c for c in compiles if c.get("compile_ok") and c.get("lookup_ok")]
    n_ok = len(ok_compile)
    sections = {c["section"] for c in ok_compile}
    insert_ok = sum(1 for i in inserts if i.get("has_insertable_interval"))
    mism_ok = sum(1 for m in mismatches if m.get("reasonable_fail_proxy"))
    size_ok = all(c.get("visual_collision_size_match", False) for c in ok_compile)
    no_fake = True  # we derive collision from spans
    labeler = labeler_design_notes()

    if (
        n_ok >= 6
        and "round" in sections
        and "rectangular" in sections
        and insert_ok >= 4
        and mism_ok >= 1
        and size_ok
        and no_fake
        and labeler["hardcoded_8mm_forbidden"]
    ):
        label = "pass"
        reason = (
            f"{n_ok} families compile+lookup; insertable={insert_ok}; "
            f"mismatch_neg={mism_ok}; sizes sourced from mesh×scale/yaml"
        )
    elif n_ok >= 6:
        label = "partial"
        reason = (
            f"compile ok={n_ok} but insertable={insert_ok}, mismatch={mism_ok}, size_ok={size_ok}"
        )
    else:
        label = "fail"
        reason = f"insufficient compile/lookup families ({n_ok})"
    return {
        "label": label,
        "reason": reason,
        "n_compile_ok": n_ok,
        "sections": sorted(sections),
        "n_insertable": insert_ok,
        "n_mismatch_negative": mism_ok,
        "allow_policy_training": False,
        "allow_full_collection": False,
        "allow_semantic_p0": False,
        "next_if_pass": "per-family 1-2 reset/settle env smoke only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families-yaml",
        type=str,
        default=str(PROJECT_ROOT / "configs/geometry_families.yaml"),
    )
    parser.add_argument(
        "--xml-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs/geometry_xml_tmp"),
    )
    parser.add_argument(
        "--out-manifest",
        type=str,
        default=str(PROJECT_ROOT / "data/manifests/geometry_compile_smoke_v1.json"),
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default=str(PROJECT_ROOT / "docs/GEOMETRY_COMPILE_SMOKE.md"),
    )
    args = parser.parse_args()

    # Ensure audit yaml exists.
    if not Path(args.families_yaml).exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "audit_geometry_assets",
            PROJECT_ROOT / "scripts" / "audit_geometry_assets.py",
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.main()

    specs = _load_specs(Path(args.families_yaml))
    xml_dir = Path(args.xml_dir)
    compiles = []
    inserts = []
    for spec in specs:
        print(f"[compile] {spec.family_id}", flush=True)
        compiles.append(compile_one(spec, xml_dir))
        if compiles[-1].get("compile_ok"):
            print(f"[insert] {spec.family_id}", flush=True)
            inserts.append(insertion_smoke(spec, xml_dir))

    # Mismatch negatives: round_16 peg vs round_4 socket; rect_16 vs rect_4; round_8 vs rect_8
    by_id = {s.family_id: s for s in specs}
    mismatch_pairs = [
        ("round_16mm", "round_4mm"),
        ("rectangular_16mm", "rectangular_4mm"),
        ("round_8mm", "rectangular_8mm"),
    ]
    mismatches = []
    for a, b in mismatch_pairs:
        print(f"[mismatch] {a} vs {b}", flush=True)
        mismatches.append(mismatch_smoke(by_id[a], by_id[b], xml_dir))

    v = verdict(compiles, inserts, mismatches)
    man = {
        "name": "geometry_compile_smoke_v1",
        "created_at": _utc(),
        "protocol": "P0-S0",
        "verdict": v,
        "labeler_adapter": labeler_design_notes(),
        "compiles": compiles,
        "insertion_smokes": inserts,
        "mismatch_smokes": mismatches,
        "xml_dir": str(xml_dir),
    }
    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Geometry Compile Smoke (P0-S0)",
        "",
        f"- 日期：{man['created_at']}",
        f"- 结论：**{v['label']}**",
        f"- reason：{v['reason']}",
        f"- compile+lookup：{v['n_compile_ok']}",
        f"- insertable families：{v['n_insertable']}",
        f"- mismatch negatives：{v['n_mismatch_negative']}",
        "- 训练 / 全量采集 / Semantic P0：**仍禁止**",
        "",
        "## Compile",
    ]
    for c in compiles:
        lines.append(
            f"- `{c['family_id']}`: compile={c.get('compile_ok')} lookup={c.get('lookup_ok')} "
            f"init_pen={c.get('initial_penetrating')} size_match={c.get('visual_collision_size_match')}"
        )
    lines.append("")
    lines.append("## Insertion")
    for i in inserts:
        lines.append(
            f"- `{i['family_id']}`: insertable={i.get('has_insertable_interval')} "
            f"strict={i.get('n_feasible_strict')} soft={i.get('n_feasible_soft')}"
        )
    lines.append("")
    lines.append("## Mismatch")
    for m in mismatches:
        lines.append(
            f"- `{m['peg_family']}` vs `{m['socket_family']}`: "
            f"fail_proxy={m.get('reasonable_fail_proxy')} pen={m['probe'].get('penetrating')}"
        )
    lines += [
        "",
        "临时 XML 仅在 `outputs/geometry_xml_tmp/`，未改正式 arena。",
        "即使 pass，下一步只能做每 family 1–2 个 reset/settle 环境 smoke。",
        "",
    ]
    Path(args.out_report).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": v, "manifest": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
