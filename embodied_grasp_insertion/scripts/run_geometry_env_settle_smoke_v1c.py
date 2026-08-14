#!/usr/bin/env python3
"""P0-S0.1c: calibrated tip/axis + pure geometric clearance (no dynamics).

Fixes v1b issues:
- tip = collision lower end; base TOP below socket_site
- wall / bottom / base / floor partitioned
- freeze peg AND socket; mj_forward only (no mj_step)
- centered + mild: no wall; blocked: must hit wall
- pen tol << radial clearance (not 8 mm)
- mismatch at same depth as matched enter
- refresh audit via write_audit_outputs (no argv steal)

Does NOT edit formal arena. No training / no collection.
"""

from __future__ import annotations

import argparse
import importlib.util
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

from embodied_grasp_insertion.geometry.family_spec import (  # noqa: E402
    INSERT_BOTTOM,
    SOCKET_BASE,
    from_dict,
)
from embodied_grasp_insertion.geometry.labeler_adapter import (  # noqa: E402
    lookup_ids,
    set_free_pose,
    tip_to_socket_site,
)
from embodied_grasp_insertion.geometry.xml_builder import write_family_xml  # noqa: E402


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


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_geometry_assets",
        PROJECT_ROOT / "scripts" / "audit_geometry_assets.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _body_vel(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[float, float]:
    v = np.asarray(data.cvel[body_id], dtype=np.float64)
    return float(np.linalg.norm(v[3:])), float(np.linalg.norm(v[:3]))


def _geom_sets(model: mujoco.MjModel, ids) -> dict[str, set[int]]:
    walls: set[int] = set()
    for i in range(model.ngeom):
        name = model.geom(i).name or ""
        if name.startswith("wall_"):
            walls.add(i)
    floor_id = int(model.geom("floor").id)
    base_id = int(model.geom(SOCKET_BASE).id)
    bottom_id = int(model.geom(INSERT_BOTTOM).id)
    return {
        "walls": walls,
        "bottom": {bottom_id},
        "base": {base_id},
        "floor": {floor_id},
        "peg": {ids.peg_collision},
        "socket_all": walls | {bottom_id, base_id},
    }


def _contact_partition(model: mujoco.MjModel, data: mujoco.MjData, ids) -> dict[str, Any]:
    gsets = _geom_sets(model, ids)
    peg = gsets["peg"]
    counts = {
        "wall": 0,
        "bottom": 0,
        "base": 0,
        "floor_socket": 0,
        "floor_peg": 0,
        "other": 0,
        "ncon": int(data.ncon),
    }
    min_wall = min_bottom = min_base = min_peg_sock = None
    wall_pen = bottom_pen = base_pen = False

    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        pair = {g1, g2}
        d = float(c.dist)
        if peg & pair and (gsets["walls"] & pair):
            counts["wall"] += 1
            min_wall = d if min_wall is None else min(min_wall, d)
            if d < -1e-6:
                wall_pen = True
        elif peg & pair and (gsets["bottom"] & pair):
            counts["bottom"] += 1
            min_bottom = d if min_bottom is None else min(min_bottom, d)
            if d < -1e-6:
                bottom_pen = True
        elif peg & pair and (gsets["base"] & pair):
            counts["base"] += 1
            min_base = d if min_base is None else min(min_base, d)
            if d < -1e-6:
                base_pen = True
        elif gsets["floor"] & pair and (gsets["socket_all"] & pair):
            counts["floor_socket"] += 1
        elif gsets["floor"] & pair and (peg & pair):
            counts["floor_peg"] += 1
        else:
            counts["other"] += 1
        if peg & pair and (gsets["socket_all"] & pair):
            min_peg_sock = d if min_peg_sock is None else min(min_peg_sock, d)

    counts["peg_socket"] = counts["wall"] + counts["bottom"] + counts["base"]
    counts["min_wall_dist"] = min_wall
    counts["min_bottom_dist"] = min_bottom
    counts["min_base_dist"] = min_base
    counts["min_peg_socket_dist"] = min_peg_sock
    counts["wall_penetrating"] = wall_pen
    counts["bottom_penetrating"] = bottom_pen
    counts["base_penetrating"] = base_pen
    return counts


def _radial(spec) -> tuple[float, float, float]:
    hole = float(spec.collision["hole_half_xy_m"])
    if spec.section == "round":
        peg_r = float(spec.collision["peg_radius_m"])
    else:
        peg_r = max(
            float(spec.collision["peg_half_width_m"]),
            float(spec.collision["peg_half_depth_m"]),
        )
    clr = max(hole - peg_r, 1e-9)
    return hole, peg_r, clr


def _pen_tol(clr: float) -> float:
    # Far below radial clearance; never the old 8 mm gate.
    return float(min(0.2 * clr, 5e-5))


def support_settle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec,
    *,
    steps: int = 400,
    low_speed_window: int = 50,
    lin_eps: float = 0.05,
    ang_eps: float = 0.5,
) -> dict[str, Any]:
    mujoco.mj_resetData(model, data)
    ids = lookup_ids(model, spec)
    set_free_pose(model, data, ids.socket_joint, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0, 0, 0]))
    quat_lie = np.array([0.70710678, 0.0, 0.70710678, 0.0])
    peg_r = float(
        spec.collision.get("peg_radius_m")
        or max(
            float(spec.collision.get("peg_half_width_m", 0.01)),
            float(spec.collision.get("peg_half_depth_m", 0.01)),
        )
    )
    set_free_pose(
        model,
        data,
        ids.peg_joint,
        np.array([0.40, 0.0, peg_r + 0.002]),
        quat_lie,
    )
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    part0 = _contact_partition(model, data, ids)
    unexpected0 = part0["peg_socket"] > 0

    lin_hist: list[float] = []
    ang_hist: list[float] = []
    max_lin = max_ang = 0.0
    exploded = False
    sock_z0 = float(data.xpos[ids.socket_body, 2])
    peg_z0 = float(data.xpos[ids.peg_body, 2])
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            exploded = True
            break
        lin_p, ang_p = _body_vel(model, data, ids.peg_body)
        lin_s, ang_s = _body_vel(model, data, ids.socket_body)
        lin, ang = max(lin_p, lin_s), max(ang_p, ang_s)
        max_lin, max_ang = max(max_lin, lin), max(max_ang, ang)
        lin_hist.append(lin)
        ang_hist.append(ang)

    part1 = _contact_partition(model, data, ids)
    sock_z1 = float(data.xpos[ids.socket_body, 2])
    peg_z1 = float(data.xpos[ids.peg_body, 2])
    window = lin_hist[-low_speed_window:] if lin_hist else []
    ang_w = ang_hist[-low_speed_window:] if ang_hist else []
    low_speed = bool(window) and max(window) < lin_eps and max(ang_w) < ang_eps
    z_ok = (-0.05 < sock_z1 < 0.15) and (-0.05 < peg_z1 < 0.20)
    passed = (
        not exploded
        and not unexpected0
        and part1["peg_socket"] == 0
        and part1["floor_socket"] >= 1
        and part1["floor_peg"] >= 1
        and low_speed
        and z_ok
    )
    return {
        "mode": "support_settle",
        "passed": bool(passed),
        "exploded": exploded,
        "unexpected_peg_socket_at_reset": unexpected0,
        "contacts_start": part0,
        "contacts_end": part1,
        "max_lin_vel": max_lin,
        "max_ang_vel": max_ang,
        "final_window_max_lin": float(max(window) if window else 1e9),
        "final_window_max_ang": float(max(ang_w) if ang_w else 1e9),
        "low_speed_window_ok": low_speed,
        "socket_z_start": sock_z0,
        "socket_z_end": sock_z1,
        "peg_z_start": peg_z0,
        "peg_z_end": peg_z1,
        "steps": steps,
    }


def tip_axis_check(spec) -> dict[str, Any]:
    c = spec.collision
    tip = float(c["peg_tip_site_z_m"])
    z0 = float(c["peg_collision_center_z_m"])
    half = float(c["peg_half_length_m"])
    base_top = float(c["base_top_z_m"])
    site = float(c["socket_site_z_m"])
    tip_is_lower = abs(tip - (z0 - half)) < 1e-9
    base_below = base_top < site - 1e-4
    return {
        "tip_is_collision_lower_end": tip_is_lower,
        "base_top_below_socket_site": base_below,
        "tip_z": tip,
        "base_top_z": base_top,
        "socket_site_z": site,
        "insertion_axis_body": c.get("insertion_axis_body"),
        "passed": bool(tip_is_lower and base_below),
    }


def geometric_clearance_probe(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec,
    *,
    lateral_m: float,
    depth_m: float,
    quat_wxyz: np.ndarray | None = None,
) -> dict[str, Any]:
    """Pure geometry: freeze peg + socket, mj_forward only (no mj_step)."""
    mujoco.mj_resetData(model, data)
    ids = lookup_ids(model, spec)
    sock_pos = np.array([0.0, 0.0, 0.05])
    quat = (
        np.array([1.0, 0.0, 0.0, 0.0])
        if quat_wxyz is None
        else np.asarray(quat_wxyz, dtype=np.float64)
    )
    set_free_pose(model, data, ids.socket_joint, sock_pos, np.array([1.0, 0, 0, 0]))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    sock = np.asarray(data.site_xpos[ids.socket_site], dtype=np.float64).copy()
    tip_local = float(spec.collision["peg_tip_site_z_m"])
    tip_target = sock + np.array([lateral_m, 0.0, -depth_m])
    # tip site is on body +z; yaw-only quat keeps body z // world z.
    peg_pos = tip_target - np.array([0.0, 0.0, tip_local])
    set_free_pose(model, data, ids.peg_joint, peg_pos, quat)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    part = _contact_partition(model, data, ids)
    tip = tip_to_socket_site(data, ids)
    return {
        "lateral_m": lateral_m,
        "depth_m": depth_m,
        "quat_wxyz": quat.tolist(),
        "tip_delta": tip.tolist(),
        "contacts": part,
        "socket_z": float(data.xpos[ids.socket_body, 2]),
        "peg_z": float(data.xpos[ids.peg_body, 2]),
        "dynamics_steps": 0,
        "pure_geometric": True,
    }


def _peg_half_along_x(spec) -> float:
    if spec.section == "round":
        return float(spec.collision["peg_radius_m"])
    return float(spec.collision["peg_half_width_m"])


def clearance_suite(model: mujoco.MjModel, data: mujoco.MjData, spec) -> dict[str, Any]:
    hole, peg_r, clr = _radial(spec)
    tol = _pen_tol(clr)
    depth = 0.0  # same depth for matched enter and mismatch negatives
    wall_t = float(spec.collision["wall_thickness_m"])
    # Shallow wall bite: deep 1.5*r box-box can drop MuJoCo contacts (ncon=0).
    half_x = _peg_half_along_x(spec)
    block_lat = (hole - half_x) + 0.5 * wall_t

    above = geometric_clearance_probe(model, data, spec, lateral_m=0.0, depth_m=-0.03)
    enter = geometric_clearance_probe(model, data, spec, lateral_m=0.0, depth_m=depth)
    mild = geometric_clearance_probe(model, data, spec, lateral_m=0.3 * clr, depth_m=depth)
    block = geometric_clearance_probe(model, data, spec, lateral_m=block_lat, depth_m=depth)

    def _no_wall_clear(p: dict[str, Any]) -> bool:
        c = p["contacts"]
        wall_ok = c["wall"] == 0 and not c["wall_penetrating"]
        base_ok = c["base"] == 0 and not c["base_penetrating"]
        # Allow soft non-wall contacts only within tight tol (should be none at mouth).
        d = c["min_peg_socket_dist"]
        pen_ok = d is None or d > -tol
        return bool(wall_ok and base_ok and pen_ok)

    above_ok = above["contacts"]["peg_socket"] == 0
    enter_ok = _no_wall_clear(enter)
    mild_ok = _no_wall_clear(mild)
    block_ok = bool(
        block["contacts"]["wall"] >= 1 or block["contacts"]["wall_penetrating"]
    )
    sock_ok = all(abs(p["socket_z"] - 0.05) < 1e-6 for p in (above, enter, mild, block))
    tip_ok = tip_axis_check(spec)["passed"]
    passed = bool(above_ok and enter_ok and mild_ok and block_ok and sock_ok and tip_ok)
    return {
        "mode": "insertion_frame_clearance_pure_geometric",
        "passed": passed,
        "probe_depth_m": depth,
        "pen_tol_m": tol,
        "clearance_radial_m": clr,
        "block_lateral_m": block_lat,
        "tip_axis": tip_axis_check(spec),
        "above": above,
        "enter": enter,
        "mild_offset": mild,
        "blocked_offset": block,
        "above_ok": above_ok,
        "enter_ok": enter_ok,
        "mild_ok": mild_ok,
        "block_ok": block_ok,
        "socket_frozen_ok": sock_ok,
    }


def mismatch_clearance(
    peg_spec,
    sock_spec,
    xml_dir: Path,
    *,
    depth_m: float,
    quat_wxyz: np.ndarray | None = None,
    note: str = "",
) -> dict[str, Any]:
    xml = write_family_xml(peg_spec, xml_dir, mismatch_socket=sock_spec, tag="mismatch_v1c")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    r = geometric_clearance_probe(
        model, data, peg_spec, lateral_m=0.0, depth_m=depth_m, quat_wxyz=quat_wxyz
    )
    hole = float(sock_spec.collision["hole_half_xy_m"])
    if peg_spec.section == "round":
        peg_extent = float(peg_spec.collision["peg_radius_m"])
    else:
        hw = float(peg_spec.collision["peg_half_width_m"])
        hd = float(peg_spec.collision["peg_half_depth_m"])
        if quat_wxyz is not None:
            # AABB half-extent after 45° yaw into square-wall aperture.
            peg_extent = (abs(hw) + abs(hd)) / np.sqrt(2.0)
        elif sock_spec.section == "round":
            peg_extent = float(np.hypot(hw, hd))
        else:
            peg_extent = max(hw, hd)
    size_conflict = peg_extent > hole * 1.01 or peg_spec.section != sock_spec.section
    hits_wall = r["contacts"]["wall"] >= 1 or r["contacts"]["wall_penetrating"]
    ok = bool(size_conflict and hits_wall)
    return {
        "peg_family": peg_spec.family_id,
        "socket_family": sock_spec.family_id,
        "depth_m": depth_m,
        "passed_as_negative": ok,
        "size_conflict": size_conflict,
        "peg_extent_m": peg_extent,
        "hole_half_m": hole,
        "hits_wall": hits_wall,
        "note": note,
        "probe": r,
        "xml": str(xml),
    }


def verdict(rows: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_ok = sum(1 for r in rows if r.get("family_passed"))
    support_ok = sum(1 for r in rows if (r.get("support") or {}).get("passed"))
    clear_ok = sum(1 for r in rows if (r.get("clearance") or {}).get("passed"))
    mism_ok = sum(1 for m in mismatches if m.get("passed_as_negative"))
    if n_ok == n and n >= 8 and mism_ok >= 3:
        label = "pass"
        reason = f"all {n} families support+clearance ok; mismatch_neg={mism_ok}/3"
    elif support_ok == n and clear_ok < n:
        label = "support_pass_clearance_fail"
        reason = (
            f"support_ok={support_ok}/{n}, clearance_ok={clear_ok}/{n}, "
            f"mismatch={mism_ok}/3"
        )
    else:
        label = "fail"
        reason = (
            f"support_ok={support_ok}/{n}, clearance_ok={clear_ok}/{n}, "
            f"family_passed={n_ok}/{n}, mismatch={mism_ok}/3"
        )
    return {
        "label": label,
        "reason": reason,
        "n_families_passed": n_ok,
        "n_families": n,
        "n_support_ok": support_ok,
        "n_clearance_ok": clear_ok,
        "n_mismatch_ok": mism_ok,
        "allow_policy_training": False,
        "allow_full_collection": False,
        "allow_semantic_p0": False,
        "allow_formal_arena_edit": label == "pass",
        "supersedes": "geometry_env_settle_smoke_v1b (clearance provisional)",
        "v1b_note": (
            "v1b support_settle trusted 8/8; insertion_frame_clearance provisional/invalid "
            "due to base-hit false pen, 8mm tol, mild not gated, dynamics steps"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-S0.1c geometry settle/clearance smoke")
    parser.add_argument(
        "--families-yaml",
        default=str(PROJECT_ROOT / "configs/geometry_families.yaml"),
    )
    parser.add_argument(
        "--xml-dir",
        default=str(PROJECT_ROOT / "outputs/geometry_xml_tmp"),
    )
    parser.add_argument("--refresh-audit", action="store_true", default=True)
    parser.add_argument("--no-refresh-audit", action="store_false", dest="refresh_audit")
    parser.add_argument(
        "--out-manifest",
        default=str(PROJECT_ROOT / "data/manifests/geometry_env_settle_smoke_v1c.json"),
    )
    parser.add_argument(
        "--out-report",
        default=str(PROJECT_ROOT / "docs/GEOMETRY_ENV_SETTLE_SMOKE_V1C.md"),
    )
    args = parser.parse_args()

    if args.refresh_audit:
        audit = _load_audit_module()
        audit.write_audit_outputs()  # does not parse argv

    raw = yaml.safe_load(Path(args.families_yaml).read_text(encoding="utf-8"))
    specs = [from_dict(d) for d in raw["families"]]
    xml_dir = Path(args.xml_dir)
    rows = []
    for spec in specs:
        print(f"[v1c] {spec.family_id}", flush=True)
        xml = write_family_xml(spec, xml_dir, tag="matched_v1c")
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        support = support_settle(model, data, spec)
        clearance = clearance_suite(model, data, spec)
        fam_ok = bool(support["passed"] and clearance["passed"])
        rows.append(
            {
                "family_id": spec.family_id,
                "section": spec.section,
                "xml": str(xml),
                "support": support,
                "clearance": clearance,
                "family_passed": fam_ok,
            }
        )

    by_id = {s.family_id: s for s in specs}
    depth = 0.0
    # Shape neg: rect peg @45° yaw into round socket (square-wall approx of round).
    # Axis-aligned rect→round often still fits the square aperture.
    quat_yaw45 = np.array([0.92387953, 0.0, 0.0, 0.38268343])  # wxyz, +45° about z
    mismatches = [
        mismatch_clearance(by_id["round_16mm"], by_id["round_4mm"], xml_dir, depth_m=depth),
        mismatch_clearance(
            by_id["rectangular_16mm"], by_id["rectangular_4mm"], xml_dir, depth_m=depth
        ),
        mismatch_clearance(
            by_id["rectangular_8mm"],
            by_id["round_8mm"],
            xml_dir,
            depth_m=depth,
            quat_wxyz=quat_yaw45,
            note="rect_peg_yaw45_into_round_square_aperture",
        ),
    ]
    v = verdict(rows, mismatches)
    man = {
        "name": "geometry_env_settle_smoke_v1c",
        "created_at": _utc(),
        "protocol": "P0-S0.1c",
        "verdict": v,
        "note": (
            "P0-S0.1 = numerical_execution_pass / physical_settle_invalid. "
            "P0-S0.1b support_settle=pass; clearance provisional. "
            "P0-S0.1c = tip/base fix + pure geometric clearance + wall partition + tight tol."
        ),
        "families": rows,
        "mismatches": mismatches,
    }
    Path(args.out_manifest).write_text(
        json.dumps(_jsonable(man), indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "# Geometry Env Settle Smoke v1c (P0-S0.1c)",
        "",
        f"- 日期：{man['created_at']}",
        f"- 结论：**{v['label']}**",
        f"- reason：{v['reason']}",
        "- 纠正：v1b clearance 为 provisional；本轮纯几何 + tip/base 校准 + wall 分统",
        "- 训练 / 全量采集 / Semantic P0：**仍禁止**",
        f"- 允许改正式 arena：{v.get('allow_formal_arena_edit')}",
        "",
        "## Families",
    ]
    for r in rows:
        c = r["clearance"]
        lines.append(
            f"- `{r['family_id']}`: family={r['family_passed']} "
            f"support={r['support']['passed']} clearance={c['passed']} "
            f"above={c.get('above_ok')} enter={c.get('enter_ok')} "
            f"mild={c.get('mild_ok')} block={c.get('block_ok')} "
            f"tol={c.get('pen_tol_m'):.2e} "
            f"enter_wall={c['enter']['contacts']['wall']} "
            f"enter_base={c['enter']['contacts']['base']} "
            f"enter_min={c['enter']['contacts']['min_peg_socket_dist']}"
        )
    lines.append("")
    lines.append("## Mismatch (same depth as matched enter)")
    for m in mismatches:
        lines.append(
            f"- `{m['peg_family']}` vs `{m['socket_family']}`: "
            f"neg_ok={m['passed_as_negative']} wall={m['hits_wall']} "
            f"depth={m['depth_m']}"
        )
    lines += ["", "未改正式 arena。", ""]
    Path(args.out_report).write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": v, "manifest": args.out_manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
