#!/usr/bin/env python3
"""P0-S0.2: formal arena per-family reset/settle smoke.

Checks compile, semantic lookup, obs dims, no deep initial peg-socket penetration,
and short settle without explosion. No collection / no training.
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
DEXJOCO_ROOT = PROJECT_ROOT.parent
for _p in (str(PROJECT_ROOT), str(DEXJOCO_ROOT), str(DEXJOCO_ROOT / "dexjoco")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from dexjoco.sim.envs.assembly_geometry import (  # noqa: E402
    DEFAULT_FAMILY_ID,
    arena_xml_path,
    names_for_family,
)
from dexjoco.sim.envs.panda_bimanual_assembly_env import (  # noqa: E402
    PandaBimanualAssemblyGymEnv,
)
from embodied_grasp_insertion.geometry.family_spec import from_dict  # noqa: E402
from embodied_grasp_insertion.geometry.formal_xml_builder import (  # noqa: E402
    write_formal_family_assets,
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


def _peg_socket_contacts(model: mujoco.MjModel, data: mujoco.MjData, names) -> dict[str, Any]:
    peg_g = int(model.geom(names.peg_collision).id)
    sock_ids = {int(model.geom(names.socket_bottom).id), int(model.geom(names.socket_base).id)}
    for i in range(model.ngeom):
        n = model.geom(i).name or ""
        if n.startswith(f"{names.socket_body}_wall_"):
            sock_ids.add(i)
    n_ps = 0
    min_d = None
    for i in range(data.ncon):
        g1, g2 = int(data.contact[i].geom1), int(data.contact[i].geom2)
        pair = {g1, g2}
        if peg_g in pair and (pair & sock_ids):
            n_ps += 1
            d = float(data.contact[i].dist)
            min_d = d if min_d is None else min(min_d, d)
    return {"peg_socket": n_ps, "min_dist": min_d}


def smoke_family(spec, *, n_resets: int, settle_steps: int, pen_tol: float) -> dict[str, Any]:
    names = names_for_family(spec.family_id)
    write_formal_family_assets(spec, overwrite=True)
    xml = arena_xml_path(spec.family_id)
    row: dict[str, Any] = {
        "family_id": spec.family_id,
        "arena_xml": str(xml),
        "names": {
            "peg_body": names.peg_body,
            "socket_body": names.socket_body,
            "socket_site": names.socket_site,
            "peg_tip_site": names.peg_tip_site,
        },
        "resets": [],
    }
    if not xml.is_file():
        row["passed"] = False
        row["error"] = f"missing arena xml: {xml}"
        return row

    # Compile check first
    try:
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    except Exception as e:
        row["passed"] = False
        row["error"] = f"compile_fail: {e}"
        return row

    # Semantic lookup
    try:
        _ = model.body(names.peg_body).id
        _ = model.body(names.socket_body).id
        _ = model.joint(names.peg_joint).id
        _ = model.joint(names.socket_joint).id
        _ = model.geom(names.peg_collision).id
        _ = model.geom(names.socket_bottom).id
        _ = model.site(names.peg_tip_site).id
        _ = model.site(names.socket_site).id
        _ = model.sensor("assembly_peg_pos").id
        _ = model.sensor("assembly_socket_pos").id
        row["lookup_ok"] = True
    except Exception as e:
        row["passed"] = False
        row["lookup_ok"] = False
        row["error"] = f"lookup_fail: {e}"
        return row

    # Env reset smoke (image_obs off for speed)
    try:
        env = PandaBimanualAssemblyGymEnv(
            geometry_family=spec.family_id,
            image_obs=False,
            randomize=False,
            hz=0,
            seed=0,
        )
    except Exception as e:
        row["passed"] = False
        row["error"] = f"env_init_fail: {e}"
        return row

    obs_shapes = None
    all_ok = True
    for ridx in range(n_resets):
        try:
            obs, info = env.reset(seed=ridx)
        except TypeError:
            obs = env.reset()
            info = {}
        state = obs.get("state", obs)
        shapes = {k: list(np.asarray(v).shape) for k, v in state.items()}
        if obs_shapes is None:
            obs_shapes = shapes
            row["obs_shapes"] = shapes
        # Contract: dims match across resets; tcp=14, fingers=32 elems, peg/socket=7.
        tcp_ok = int(np.asarray(state["tcp_pose"]).size) == 14
        grip_ok = int(np.asarray(state["gripper_pose"]).size) == 32
        peg_ok = int(np.asarray(state["peg_ori_pose"]).size) == 7
        sock_ok = int(np.asarray(state["socket_ori_pose"]).size) == 7
        shapes_match = shapes == obs_shapes
        shapes_ok = bool(shapes_match and tcp_ok and grip_ok and peg_ok and sock_ok)

        mujoco.mj_forward(env.model, env.data)
        c0 = _peg_socket_contacts(env.model, env.data, names)
        deep0 = c0["min_dist"] is not None and c0["min_dist"] < -pen_tol

        exploded = False
        max_lin = 0.0
        max_ang = 0.0
        lin_hist: list[float] = []
        ang_hist: list[float] = []
        for _ in range(settle_steps):
            mujoco.mj_step(env.model, env.data)
            if not (np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()):
                exploded = True
                break
            step_lin = 0.0
            step_ang = 0.0
            for bid in (env._peg_body_id, env._socket_body_id):
                v = np.asarray(env.data.cvel[bid], dtype=np.float64)
                step_lin = max(step_lin, float(np.linalg.norm(v[3:])))
                step_ang = max(step_ang, float(np.linalg.norm(v[:3])))
            max_lin = max(max_lin, step_lin)
            max_ang = max(max_ang, step_ang)
            lin_hist.append(step_lin)
            ang_hist.append(step_ang)

        c1 = _peg_socket_contacts(env.model, env.data, names)
        deep1 = c1["min_dist"] is not None and c1["min_dist"] < -pen_tol
        tip = np.asarray(env.data.site_xpos[env._peg_tip_site_id], dtype=np.float64)
        sock = np.asarray(env.data.site_xpos[env._socket_site_id], dtype=np.float64)
        # Target semantics: socket_site exists and is distinct from peg tip at reset.
        tip_sock_dist = float(np.linalg.norm(tip - sock))
        semantics_ok = tip_sock_dist > 0.05  # peg and socket start on opposite table sides

        win = 50
        lin_w = lin_hist[-win:] if lin_hist else []
        ang_w = ang_hist[-win:] if ang_hist else []
        final_max_lin = float(max(lin_w) if lin_w else 1e9)
        final_max_ang = float(max(ang_w) if ang_w else 1e9)
        # Codex-confirmed band (~0.0009 m/s, ~0.018 rad/s); gate with modest margin.
        low_speed_ok = bool(lin_w) and final_max_lin < 0.005 and final_max_ang < 0.05

        reset_ok = bool(
            shapes_ok
            and not deep0
            and not deep1
            and not exploded
            and semantics_ok
            and low_speed_ok
        )
        all_ok = all_ok and reset_ok
        row["resets"].append(
            {
                "reset_idx": ridx,
                "shapes_ok": shapes_ok,
                "shapes_match": shapes_match,
                "tcp_ok": tcp_ok,
                "grip_ok": grip_ok,
                "peg_ok": peg_ok,
                "sock_ok": sock_ok,
                "contacts_start": c0,
                "contacts_end": c1,
                "deep_pen_at_reset": deep0,
                "deep_pen_after_settle": deep1,
                "exploded": exploded,
                "max_lin_vel": max_lin,
                "max_ang_vel": max_ang,
                "final_window_max_lin": final_max_lin,
                "final_window_max_ang": final_max_ang,
                "low_speed_ok": low_speed_ok,
                "tip_to_socket_site_m": tip_sock_dist,
                "semantics_ok": semantics_ok,
                "passed": reset_ok,
            }
        )

    env.close()
    row["passed"] = bool(all_ok and row.get("lookup_ok"))
    row["n_resets_ok"] = sum(1 for r in row["resets"] if r["passed"])
    return row


def verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_ok = sum(1 for r in rows if r.get("passed"))
    if n_ok == n and n >= 8:
        label = "pass"
        reason = f"all {n} families formal arena reset/settle ok"
    elif n_ok >= 1:
        label = "partial"
        reason = f"family_ok={n_ok}/{n}"
    else:
        label = "fail"
        reason = f"family_ok={n_ok}/{n}"
    return {
        "label": label,
        "reason": reason,
        "n_families_passed": n_ok,
        "n_families": n,
        "allow_policy_training": False,
        "allow_full_collection": False,
        "allow_semantic_p0": False,
        "default_family": DEFAULT_FAMILY_ID,
        "next_if_pass": "grasp_stability_gate (still no train/collect)",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families-yaml",
        default=str(PROJECT_ROOT / "configs/geometry_families.yaml"),
    )
    parser.add_argument("--n-resets", type=int, default=2)
    parser.add_argument("--settle-steps", type=int, default=200)
    parser.add_argument("--pen-tol", type=float, default=1e-3)
    parser.add_argument(
        "--out-manifest",
        default=str(PROJECT_ROOT / "data/manifests/formal_arena_reset_smoke_v1.json"),
    )
    parser.add_argument(
        "--out-report",
        default=str(PROJECT_ROOT / "docs/FORMAL_ARENA_RESET_SMOKE.md"),
    )
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.families_yaml).read_text(encoding="utf-8"))
    specs = [from_dict(d) for d in raw["families"]]
    rows = []
    for spec in specs:
        print(f"[formal-reset] {spec.family_id}", flush=True)
        rows.append(
            smoke_family(
                spec,
                n_resets=args.n_resets,
                settle_steps=args.settle_steps,
                pen_tol=args.pen_tol,
            )
        )

    v = verdict(rows)
    man = {
        "name": "formal_arena_reset_smoke_v1",
        "created_at": _utc(),
        "protocol": "P0-S0.2",
        "verdict": v,
        "note": (
            "Proves parameterized formal arena compile + reset/settle plumbing. "
            "Does NOT prove full-depth insert, grasp stability, or hole semantics policy."
        ),
        "families": rows,
    }
    Path(args.out_manifest).write_text(
        json.dumps(_jsonable(man), indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "# Formal Arena Reset Smoke (P0-S0.2)",
        "",
        f"- 日期：{man['created_at']}",
        f"- 结论：**{v['label']}**",
        f"- reason：{v['reason']}",
        "- 默认 `round_8mm` arena 文件未改写；其他族生成旁路 XML",
        "- 训练 / 全量采集 / Semantic P0：**仍禁止**",
        "",
        "## Families",
    ]
    for r in rows:
        lines.append(
            f"- `{r['family_id']}`: passed={r.get('passed')} "
            f"resets_ok={r.get('n_resets_ok')}/{args.n_resets} "
            f"lookup={r.get('lookup_ok')} err={r.get('error')}"
        )
    lines += ["", "未开始采集或训练。", ""]
    Path(args.out_report).write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": v, "manifest": args.out_manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
