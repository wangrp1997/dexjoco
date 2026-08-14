"""P0-S0: audit IndustReal mesh/YAML vs official 8mm XML scales."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embodied_grasp_insertion.geometry.family_spec import (  # noqa: E402
    ARENA_MESH_SCALE,
    INDUSTREAL_MESH,
    INDUSTREAL_YAML,
    OFFICIAL_8MM_PEG_XML,
    OFFICIAL_8MM_SOCKET_XML,
    GeometryFamilySpec,
    family_id_for,
    mesh_path,
)
from embodied_grasp_insertion.geometry.xml_builder import derive_collision_from_spans  # noqa: E402


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_obj_span(path: Path) -> dict[str, Any]:
    vs = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            vs.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vs:
        raise RuntimeError(f"no vertices in {path}")
    v = np.asarray(vs, dtype=np.float64)
    mn = v.min(axis=0)
    mx = v.max(axis=0)
    span = mx - mn
    return {
        "path": str(path),
        "n_vertices": int(v.shape[0]),
        "min_xyz": mn.tolist(),
        "max_xyz": mx.tolist(),
        "span_xyz": span.tolist(),
        "xy_mean_span": float(0.5 * (span[0] + span[1])),
        "z_span": float(span[2]),
    }


def parse_official_8mm_xml() -> dict[str, Any]:
    peg_txt = OFFICIAL_8MM_PEG_XML.read_text(encoding="utf-8")
    sock_txt = OFFICIAL_8MM_SOCKET_XML.read_text(encoding="utf-8")
    peg_scale = re.search(r'scale="([0-9.]+)\s+[0-9.]+\s+[0-9.]+"', peg_txt)
    sock_scale = re.search(r'scale="([0-9.]+)\s+[0-9.]+\s+[0-9.]+"', sock_txt)
    cyl = re.search(
        r'name="industreal_round_peg_8mm_collision"[^>]*size="([0-9.eE+-]+)\s+([0-9.eE+-]+)"',
        peg_txt,
    )
    tip = re.search(
        r'name="industreal_round_peg_8mm_tip_site"[^>]*pos="([^"]+)"',
        peg_txt,
    )
    return {
        "peg_xml": str(OFFICIAL_8MM_PEG_XML),
        "socket_xml": str(OFFICIAL_8MM_SOCKET_XML),
        "peg_mesh_scale": float(peg_scale.group(1)) if peg_scale else None,
        "socket_mesh_scale": float(sock_scale.group(1)) if sock_scale else None,
        "collision_cylinder_radius_m": float(cyl.group(1)) if cyl else None,
        "collision_cylinder_half_height_m": float(cyl.group(2)) if cyl else None,
        "tip_site_pos": tip.group(1) if tip else None,
        "note": (
            "Official 8mm collision radius≈0.01785 is ~0.5*mesh_xy_span*scale, "
            "NOT the yaml nominal 8mm diameter. Do not treat asset name mm as MuJoCo meters."
        ),
    }


def build_specs() -> list[GeometryFamilySpec]:
    info = yaml.safe_load(INDUSTREAL_YAML.read_text(encoding="utf-8"))
    specs: list[GeometryFamilySpec] = []
    for section, sizes in (("round", [4, 8, 12, 16]), ("rectangular", [4, 8, 12, 16])):
        for size in sizes:
            key = f"{section}_peg_hole_{size}mm"
            block = info[key]
            peg_key = f"{section}_peg_{size}mm"
            hole_key = f"{section}_hole_{size}mm"
            peg_info = block[peg_key]
            hole_info = block[hole_key]
            peg_mesh = f"industreal_{section}_peg_{size}mm.obj"
            sock_mesh = f"industreal_tray_insert_{section}_peg_{size}mm.obj"
            peg_span = load_obj_span(mesh_path(peg_mesh))
            sock_span = load_obj_span(mesh_path(sock_mesh))
            scale = ARENA_MESH_SCALE
            notes = [
                "raw mesh spans match yaml nominal sizes (meters)",
                f"arena visual uses scale={scale} from official 8mm XML convention",
                "collision for smoke derived from measured scaled spans, not copied 8mm primitives",
            ]
            if section == "round":
                peg_d = float(peg_info["diameter"])
                hole_d = float(hole_info["diameter"])
                clearance = hole_d - peg_d
                # Consistency check: mesh xy span vs yaml diameter
                mesh_d = float(peg_span["xy_mean_span"])
                if abs(mesh_d - peg_d) > 5e-4:
                    notes.append(f"WARN mesh_xy_span={mesh_d} vs yaml_diameter={peg_d}")
                spec = GeometryFamilySpec(
                    family_id=family_id_for(section, size),
                    section=section,
                    nominal_size_mm=size,
                    peg_mesh=peg_mesh,
                    socket_mesh=sock_mesh,
                    peg_urdf_key=str(peg_info["urdf_path"]),
                    hole_urdf_key=str(hole_info["urdf_path"]),
                    peg_diameter_m=peg_d,
                    peg_length_m=float(peg_info["length"]),
                    hole_diameter_m=hole_d,
                    hole_height_m=float(hole_info["height"]),
                    hole_depth_m=float(hole_info["depth"]),
                    clearance_diameter_m=clearance,
                    peg_mesh_span_xyz_m=peg_span["span_xyz"],
                    socket_mesh_span_xyz_m=sock_span["span_xyz"],
                    asset_scale=scale,
                    peg_visual_span_xyz_m=[x * scale for x in peg_span["span_xyz"]],
                    socket_visual_span_xyz_m=[x * scale for x in sock_span["span_xyz"]],
                    symmetry="SO2_about_insertion_axis",
                    notes=notes,
                )
            else:
                pw, pd = float(peg_info["width"]), float(peg_info["depth"])
                hw = float(hole_info["width"])
                spec = GeometryFamilySpec(
                    family_id=family_id_for(section, size),
                    section=section,
                    nominal_size_mm=size,
                    peg_mesh=peg_mesh,
                    socket_mesh=sock_mesh,
                    peg_urdf_key=str(peg_info["urdf_path"]),
                    hole_urdf_key=str(hole_info["urdf_path"]),
                    peg_width_m=pw,
                    peg_depth_m=pd,
                    peg_length_m=float(peg_info["length"]),
                    hole_width_m=hw,
                    hole_height_m=float(hole_info["height"]),
                    hole_depth_m=float(hole_info["depth"]),
                    clearance_width_m=hw - pw,
                    peg_mesh_span_xyz_m=peg_span["span_xyz"],
                    socket_mesh_span_xyz_m=sock_span["span_xyz"],
                    asset_scale=scale,
                    peg_visual_span_xyz_m=[x * scale for x in peg_span["span_xyz"]],
                    socket_visual_span_xyz_m=[x * scale for x in sock_span["span_xyz"]],
                    symmetry="requires_yaw_alignment_C2",
                    notes=notes
                    + [
                        "rectangular hole yaml lists width/height/depth; cross-section depth assumed≈width for smoke hole opening"
                    ],
                )
            spec.collision = derive_collision_from_spans(spec)
            specs.append(spec)
    return specs


def write_audit_outputs(
    *,
    out_manifest: str | Path | None = None,
    out_report: str | Path | None = None,
    out_yaml: str | Path | None = None,
) -> dict[str, Any]:
    """Write audit artifacts without parsing sys.argv (safe to call from other scripts)."""
    out_manifest = Path(
        out_manifest or (PROJECT_ROOT / "data/manifests/geometry_asset_audit_v1.json")
    )
    out_report = Path(out_report or (PROJECT_ROOT / "docs/GEOMETRY_ASSET_AUDIT.md"))
    out_yaml = Path(out_yaml or (PROJECT_ROOT / "configs/geometry_families.yaml"))

    official = parse_official_8mm_xml()
    specs = build_specs()
    round8 = next(s for s in specs if s.family_id == "round_8mm")
    measured_r = float(round8.collision["peg_radius_m"])
    official_r = official.get("collision_cylinder_radius_m")
    unit_story = {
        "raw_mesh_unit": "meters (matches industreal_asset_info_pegs.yaml diameters/widths)",
        "xml_mesh_scale": ARENA_MESH_SCALE,
        "xml_scale_source": "official industreal_round_peg_8mm.xml / tray_insert xml",
        "final_mujoco_visual_size": "raw_span * 4.5",
        "official_8mm_collision_radius_m": official_r,
        "measured_scaled_round8_radius_m": measured_r,
        "collision_vs_measured_abs_diff_m": None
        if official_r is None
        else abs(float(official_r) - measured_r),
        "warning": (
            "Asset filename '8mm' refers to IndustReal nominal size in RAW meters (~0.008). "
            "MuJoCo arena visual/collision are ~4.5x larger. Never equate name-mm to collision meters."
        ),
    }

    families_yaml = {
        "asset_scale": ARENA_MESH_SCALE,
        "asset_scale_source": "official_8mm_arena_xml",
        "mesh_dir": str(INDUSTREAL_MESH),
        "yaml_info": str(INDUSTREAL_YAML),
        "families": [s.to_dict() for s in specs],
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(families_yaml, sort_keys=False), encoding="utf-8")

    manifest = {
        "name": "geometry_asset_audit_v1",
        "created_at": _utc(),
        "protocol": "P0-S0",
        "official_8mm_xml": official,
        "unit_story": unit_story,
        "n_families": len(specs),
        "families": [s.to_dict() for s in specs],
        "verdict_hint": "audit_only_not_semantic_p0",
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Geometry Asset Audit (P0-S0)",
        "",
        f"- 日期：{manifest['created_at']}",
        f"- families：{len(specs)}（round+rect × 4/8/12/16）",
        "",
        "## 单位口径（关键）",
        "",
        f"- raw mesh / yaml：米制，round_8 peg Ø≈{round8.peg_diameter_m}",
        f"- 官方 XML mesh scale：{official.get('peg_mesh_scale')}",
        f"- 官方 8mm collision radius：{official_r} m",
        f"- 由 mesh×scale 测得 round_8 radius：{measured_r:.6f} m",
        f"- |官方 collision − 测得|：{unit_story['collision_vs_measured_abs_diff_m']}",
        "",
        unit_story["warning"],
        "",
        "## Family 摘要",
        "",
        "| family | section | yaml peg | mesh xy span | visual Ø/W after×4.5 | clearance (raw) |",
        "|---|---|---|---|---|---|",
    ]
    for s in specs:
        if s.section == "round":
            yaml_p = f"Ø{s.peg_diameter_m:.6f}"
            vis = f"Ø{s.peg_visual_span_xyz_m[0]:.4f}"
            clr = f"{s.clearance_diameter_m:.6f}"
        else:
            yaml_p = f"{s.peg_width_m:.6f}×{s.peg_depth_m:.6f}"
            vis = f"{s.peg_visual_span_xyz_m[0]:.4f}×{s.peg_visual_span_xyz_m[1]:.4f}"
            clr = f"{s.clearance_width_m:.6f}"
        mesh_xy = f"{s.peg_mesh_span_xyz_m[0]:.6f}"
        lines.append(
            f"| {s.family_id} | {s.section} | {yaml_p} | {mesh_xy} | {vis} | {clr} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "- 不得把文件名 mm 当成 MuJoCo collision 直径。",
        "- 不得把 8mm XML collision 数字复制到其他 family。",
        "- 临时 smoke collision 必须来自 measured mesh×scale（+ yaml hole）。",
        "- XML compile 成功 ≠ Semantic P0 通过。",
        "",
    ]
    out_report.write_text("\n".join(lines), encoding="utf-8")
    return {"manifest": str(out_manifest), "yaml": str(out_yaml), "n": len(specs), "specs": specs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-manifest",
        type=str,
        default=str(PROJECT_ROOT / "data/manifests/geometry_asset_audit_v1.json"),
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default=str(PROJECT_ROOT / "docs/GEOMETRY_ASSET_AUDIT.md"),
    )
    parser.add_argument(
        "--out-yaml",
        type=str,
        default=str(PROJECT_ROOT / "configs/geometry_families.yaml"),
    )
    args = parser.parse_args()
    result = write_audit_outputs(
        out_manifest=args.out_manifest,
        out_report=args.out_report,
        out_yaml=args.out_yaml,
    )
    print(json.dumps({k: result[k] for k in ("manifest", "yaml", "n")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
