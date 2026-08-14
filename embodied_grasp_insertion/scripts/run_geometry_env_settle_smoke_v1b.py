#!/usr/bin/env python3
"""P0-S0.1b: strict support-settle + insertion-frame clearance smoke.

Supersedes the loose P0-S0.1 numerical-only criteria.
Does NOT edit the formal arena. No training / no collection.
"""

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


def _body_vel(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[float, float]:
    # cvel: (rot, lin) in 6D
    v = np.asarray(data.cvel[body_id], dtype=np.float64)
    ang = float(np.linalg.norm(v[:3]))
    lin = float(np.linalg.norm(v[3:]))
    return lin, ang


def _contact_partition(model: mujoco.MjModel, data: mujoco.MjData, ids) -> dict[str, int]:
    floor_id = int(model.geom("floor").id)
    peg_g = ids.peg_collision
    sock_geoms = {ids.insert_bottom}
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name.startswith("wall_") or name == "socket_base":
            sock_geoms.add(i)
    n_floor_sock = n_floor_peg = n_peg_sock = n_other = 0
    for i in range(data.ncon):
        g1, g2 = int(data.contact[i].geom1), int(data.contact[i].geom2)
        pair = {g1, g2}
        if floor_id in pair and (pair & sock_geoms):
            n_floor_sock += 1
        elif floor_id in pair and peg_g in pair:
            n_floor_peg += 1
        elif peg_g in pair and (pair & sock_geoms):
            n_peg_sock += 1
        else:
            n_other += 1
    return {
        "floor_socket": n_floor_sock,
        "floor_peg": n_floor_peg,
        "peg_socket": n_peg_sock,
        "other": n_other,
        "ncon": int(data.ncon),
    }


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
    """Place peg and socket separately on floor; require true low-speed settle."""
    mujoco.mj_resetData(model, data)
    ids = lookup_ids(model, spec)
    # Socket resting on floor (base geom sits near z=0). Peg far away, lying on side.
    set_free_pose(model, data, ids.socket_joint, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0, 0, 0]))
    # 90 deg about y: tip axis along +x, cylinder rests on floor with radius clearance.
    quat_lie = np.array([0.70710678, 0.0, 0.70710678, 0.0])  # wxyz, rot y
    peg_r = float(
        spec.collision.get("peg_radius_m")
        or max(float(spec.collision.get("peg_half_width_m", 0.01)), float(spec.collision.get("peg_half_depth_m", 0.01)))
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
    max_lin = 0.0
    max_ang = 0.0
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
        lin = max(lin_p, lin_s)
        ang = max(ang_p, ang_s)
        max_lin = max(max_lin, lin)
        max_ang = max(max_ang, ang)
        lin_hist.append(lin)
        ang_hist.append(ang)

    part1 = _contact_partition(model, data, ids)
    sock_z1 = float(data.xpos[ids.socket_body, 2])
    peg_z1 = float(data.xpos[ids.peg_body, 2])
    window = lin_hist[-low_speed_window:] if lin_hist else []
    ang_w = ang_hist[-low_speed_window:] if ang_hist else []
    low_speed = bool(window) and max(window) < lin_eps and max(ang_w) < ang_eps
    # Rest near floor; allow slight geom sink but not bounce launch.
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


def clearance_probe(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec,
    *,
    lateral_m: float,
    depth_m: float,
) -> dict[str, Any]:
    """Kinematic placement with socket held fixed; no free-fall bounce."""
    mujoco.mj_resetData(model, data)
    ids = lookup_ids(model, spec)
    # Hold socket fixed at origin height.
    sock_pos = np.array([0.0, 0.0, 0.05])
    set_free_pose(model, data, ids.socket_joint, sock_pos, np.array([1.0, 0, 0, 0]))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    sock = np.asarray(data.site_xpos[ids.socket_site], dtype=np.float64).copy()
    tip_local = float(spec.collision["peg_tip_site_z_m"])
    tip_target = sock + np.array([lateral_m, 0.0, -depth_m])
    peg_pos = tip_target - np.array([0.0, 0.0, tip_local])
    set_free_pose(model, data, ids.peg_joint, peg_pos, np.array([1.0, 0, 0, 0]))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    # Keep socket welded for a few steps while peg is static (kinematic check).
    for _ in range(10):
        set_free_pose(model, data, ids.socket_joint, sock_pos, np.array([1.0, 0, 0, 0]))
        data.qvel[7:13] = 0.0  # socket freejoint dof if after peg... safer zero all then restore peg
        # Zero only socket joint velocities via qvel adr
        jnt = ids.socket_joint
        dof = int(model.jnt_dofadr[jnt])
        data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_step(model, data)
        set_free_pose(model, data, ids.socket_joint, sock_pos, np.array([1.0, 0, 0, 0]))

    part = _contact_partition(model, data, ids)
    tip = tip_to_socket_site(data, ids)
    # Penetration proxy
    min_dist = None
    penetrating = False
    for i in range(data.ncon):
        d = float(data.contact[i].dist)
        if min_dist is None or d < min_dist:
            min_dist = d
        if d < -1e-4:
            penetrating = True
    hole = float(spec.collision["hole_half_xy_m"])
    if spec.section == "round":
        peg_r = float(spec.collision["peg_radius_m"])
    else:
        peg_r = max(float(spec.collision["peg_half_width_m"]), float(spec.collision["peg_half_depth_m"]))
    clearance = hole - peg_r
    return {
        "lateral_m": lateral_m,
        "depth_m": depth_m,
        "tip_delta": tip.tolist(),
        "contacts": part,
        "min_contact_dist": min_dist,
        "penetrating": penetrating,
        "clearance_radial_m": clearance,
        "socket_z": float(data.xpos[ids.socket_body, 2]),
    }


def clearance_suite(model: mujoco.MjModel, data: mujoco.MjData, spec) -> dict[str, Any]:
    hole = float(spec.collision["hole_half_xy_m"])
    if spec.section == "round":
        peg_r = float(spec.collision["peg_radius_m"])
    else:
        peg_r = max(float(spec.collision["peg_half_width_m"]), float(spec.collision["peg_half_depth_m"]))
    clr = max(hole - peg_r, 1e-6)
    above = clearance_probe(model, data, spec, lateral_m=0.0, depth_m=-0.03)
    enter = clearance_probe(model, data, spec, lateral_m=0.0, depth_m=0.0)
    mild = clearance_probe(model, data, spec, lateral_m=0.3 * clr, depth_m=0.0)
    block = clearance_probe(model, data, spec, lateral_m=1.5 * peg_r, depth_m=0.0)

    def _min_d(p):
        return p["min_contact_dist"] if p["min_contact_dist"] is not None else 0.0

    # Hover above: no peg-socket contacts / no deep pen.
    above_ok = above["contacts"]["peg_socket"] == 0 and not (
        above["penetrating"] and _min_d(above) < -1e-3
    )
    # Enter centered: allow tiny soft pad contact (<8mm).
    enter_ok = _min_d(enter) > -0.008
    # Large lateral: blocked by wall (contact or penetration).
    block_ok = block["penetrating"] or block["contacts"]["peg_socket"] >= 1
    sock_ok = all(
        abs(p["socket_z"] - 0.05) < 0.02 for p in (above, enter, mild, block)
    )
    passed = bool(above_ok and enter_ok and block_ok and sock_ok)
    return {
        "mode": "insertion_frame_clearance",
        "passed": passed,
        "above": above,
        "enter": enter,
        "mild_offset": mild,
        "blocked_offset": block,
        "above_ok": above_ok,
        "enter_ok": enter_ok,
        "block_ok": block_ok,
        "socket_held_ok": sock_ok,
        # backward-compatible aliases used in older prints
        "center_ok": enter_ok,
        "center": enter,
    }


def mismatch_clearance(peg_spec, sock_spec, xml_dir: Path) -> dict[str, Any]:
    xml = write_family_xml(peg_spec, xml_dir, mismatch_socket=sock_spec, tag="mismatch_v1b")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    # Use peg tip semantics; socket hole from mismatch family baked into XML.
    r = clearance_probe(model, data, peg_spec, lateral_m=0.0, depth_m=0.015)
    peg_r = float(
        peg_spec.collision.get("peg_radius_m")
        or max(float(peg_spec.collision.get("peg_half_width_m", 0)), float(peg_spec.collision.get("peg_half_depth_m", 0)))
    )
    hole = float(sock_spec.collision["hole_half_xy_m"])
    size_conflict = peg_r > hole * 1.05 or peg_spec.section != sock_spec.section
    ok = bool(size_conflict and (r["penetrating"] or r["contacts"]["peg_socket"] >= 1))
    return {
        "peg_family": peg_spec.family_id,
        "socket_family": sock_spec.family_id,
        "passed_as_negative": ok,
        "size_conflict": size_conflict,
        "probe": r,
        "xml": str(xml),
    }


def verdict(rows: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_ok = sum(1 for r in rows if r.get("family_passed"))
    support_ok = sum(1 for r in rows if (r.get("support") or {}).get("passed"))
    clear_ok = sum(1 for r in rows if (r.get("clearance") or {}).get("passed"))
    mism_ok = sum(1 for m in mismatches if m.get("passed_as_negative"))
    # Strict: require ALL families for pass.
    if n_ok == n and n >= 8 and mism_ok >= 2:
        label = "pass"
        reason = f"all {n} families support+clearance ok; mismatch_neg={mism_ok}"
    elif support_ok >= 6 and clear_ok >= 4:
        label = "partial"
        reason = (
            f"support_ok={support_ok}/{n}, clearance_ok={clear_ok}/{n}, "
            f"family_passed={n_ok}/{n}, mismatch={mism_ok}"
        )
    else:
        label = "fail"
        reason = (
            f"support_ok={support_ok}/{n}, clearance_ok={clear_ok}/{n}, "
            f"family_passed={n_ok}/{n}, mismatch={mism_ok}"
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
        "supersedes": "geometry_env_settle_smoke_v1 (numerical-only)",
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
    parser.add_argument("--refresh-audit", action="store_true", default=True)
    parser.add_argument(
        "--out-manifest",
        default=str(PROJECT_ROOT / "data/manifests/geometry_env_settle_smoke_v1b.json"),
    )
    parser.add_argument(
        "--out-report",
        default=str(PROJECT_ROOT / "docs/GEOMETRY_ENV_SETTLE_SMOKE_V1B.md"),
    )
    args = parser.parse_args()

    if args.refresh_audit:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "audit_geometry_assets",
            PROJECT_ROOT / "scripts" / "audit_geometry_assets.py",
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.main()

    raw = yaml.safe_load(Path(args.families_yaml).read_text(encoding="utf-8"))
    specs = [from_dict(d) for d in raw["families"]]
    xml_dir = Path(args.xml_dir)
    rows = []
    for spec in specs:
        print(f"[v1b] {spec.family_id}", flush=True)
        xml = write_family_xml(spec, xml_dir, tag="matched_v1b")
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
    mismatches = [
        mismatch_clearance(by_id["round_16mm"], by_id["round_4mm"], xml_dir),
        mismatch_clearance(by_id["rectangular_16mm"], by_id["rectangular_4mm"], xml_dir),
        mismatch_clearance(by_id["round_8mm"], by_id["rectangular_8mm"], xml_dir),
    ]
    v = verdict(rows, mismatches)
    man = {
        "name": "geometry_env_settle_smoke_v1b",
        "created_at": _utc(),
        "protocol": "P0-S0.1b",
        "verdict": v,
        "note": (
            "P0-S0.1 is numerical-execution-only and INVALID as physical settle pass. "
            "This v1b splits support-settle vs insertion-frame clearance with strict criteria."
        ),
        "families": rows,
        "mismatches": mismatches,
    }
    Path(args.out_manifest).write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# Geometry Env Settle Smoke v1b (P0-S0.1b)",
        "",
        f"- 日期：{man['created_at']}",
        f"- 结论：**{v['label']}**",
        f"- reason：{v['reason']}",
        "- 纠正：P0-S0.1 仅为 numerical execution pass，**不能**称为物理 settle pass",
        "- 本轮拆分：`support_settle` + `insertion_frame_clearance`；要求 8/8",
        "- 训练 / 全量采集 / Semantic P0：**仍禁止**",
        f"- 允许改正式 arena：{v.get('allow_formal_arena_edit')}",
        "",
        "## Families",
    ]
    for r in rows:
        lines.append(
            f"- `{r['family_id']}`: family={r['family_passed']} "
            f"support={r['support']['passed']} clearance={r['clearance']['passed']} "
            f"above={r['clearance'].get('above_ok')} enter={r['clearance'].get('enter_ok')} "
            f"block={r['clearance'].get('block_ok')} "
            f"max_lin={r['support']['max_lin_vel']:.3f} "
            f"peg_sock0={r['support']['contacts_start']['peg_socket']}"
        )
    lines.append("")
    lines.append("## Mismatch")
    for m in mismatches:
        lines.append(
            f"- `{m['peg_family']}` vs `{m['socket_family']}`: neg_ok={m['passed_as_negative']}"
        )
    lines += ["", "未改正式 arena。", ""]
    Path(args.out_report).write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": v, "manifest": args.out_manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
