#!/usr/bin/env python3
"""Cross-Geometry Contact-Affordance P0 runner."""

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
EMBODIED = DEXJOCO_ROOT / "embodied_grasp_insertion"
for _p in (
    str(PROJECT_ROOT),
    str(DEXJOCO_ROOT),
    str(DEXJOCO_ROOT / "dexjoco"),
    str(EMBODIED),
    str(DEXJOCO_ROOT.parent / "reach_insert_rl"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.geometry.family_spec import from_dict  # noqa: E402
from embodied_grasp_insertion.geometry.formal_xml_builder import (  # noqa: E402
    write_dual_socket_family_assets,
)
from insertion_science.affordance.features import (  # noqa: E402
    contact_affordance_features,
    raw_relation_features,
    shuffle_pairing,
    tip_lat_axis_features,
)
from insertion_science.affordance.geometry_scene import (  # noqa: E402
    make_scene,
    place_tip_in_socket_frame,
    socket_frame,
)
from insertion_science.affordance.probe import fit_predict_binary  # noqa: E402
from insertion_science.affordance.twists import (  # noqa: E402
    apply_twist_and_label,
    restore_peg,
)
from dexjoco.sim.envs.panda_bimanual_assembly_env import (  # noqa: E402
    PandaBimanualAssemblyGymEnv,
)
from insertion_science.affordance.geometry_scene import SceneHandles, characteristic_length  # noqa: E402

PROTOCOL = "CrossGeometryContactAffordanceP0"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_specs(cfg: dict) -> list:
    raw = yaml.safe_load(Path(cfg["families_yaml"]).read_text(encoding="utf-8"))
    by_id = {f["family_id"]: f for f in raw["families"]}
    specs = []
    for fid in cfg["families"]:
        if fid not in by_id:
            raise KeyError(fid)
        specs.append(from_dict(by_id[fid]))
    return specs


def collect_family_rows(spec, cfg: dict, *, instance_key: str = "primary") -> list[dict]:
    dual = instance_key not in ("primary", "a", "0", "")
    if dual:
        assets = write_dual_socket_family_assets(
            spec, secondary_key=cfg["secondary_instance_key"], overwrite=False
        )
        env = PandaBimanualAssemblyGymEnv(
            geometry_family=spec.family_id,
            xml_path=assets["dual_arena_xml"],
            image_obs=False,
            randomize=False,
            hz=0,
            seed=int(cfg["seed"]),
        )
        env.reset()
        from dexjoco.sim.envs.assembly_geometry import names_for_socket_instance

        names = names_for_socket_instance(spec.family_id, cfg["secondary_instance_key"])
        # Retarget env socket handles to secondary instance (same family, different hole).
        env._socket_body_id = int(env.model.body(names.socket_body).id)
        env._socket_joint_id = int(env.model.joint(names.socket_joint).id)
        env._socket_qpos_adr = int(env.model.jnt_qposadr[env._socket_joint_id])
        env._socket_qvel_adr = int(env.model.jnt_dofadr[env._socket_joint_id])
        # Override geom names so insert helpers resolve secondary socket site.
        from dexjoco.sim.envs.assembly_geometry import names_for_family
        base = names_for_family(spec.family_id)
        env._geom_names = names  # secondary names object
        scene = SceneHandles(
            env=env,
            spec=spec,
            char_len=characteristic_length(spec),
            peg_qpos_adr=int(env._peg_qpos_adr),
            peg_qvel_adr=int(env._peg_qvel_adr),
            socket_qpos_adr=int(env._socket_qpos_adr),
            socket_qvel_adr=int(env._socket_qvel_adr),
            instance_key=instance_key,
        )
    else:
        scene = make_scene(spec, seed=int(cfg["seed"]), dual_socket=False)

    rows = []
    try:
        for pose in cfg["poses"]:
            place = place_tip_in_socket_frame(
                scene,
                tip_offset_L=np.asarray(pose["tip_offset_L"], dtype=np.float64),
                tilt_rad=np.asarray(pose["tilt_rad"], dtype=np.float64),
            )
            body_pos0 = np.asarray(
                scene.env.data.qpos[scene.peg_qpos_adr : scene.peg_qpos_adr + 3]
            ).copy()
            body_quat0 = np.asarray(
                scene.env.data.qpos[scene.peg_qpos_adr + 3 : scene.peg_qpos_adr + 7]
            ).copy()
            sock_pos, sock_quat = place["sock_pos"], place["sock_quat"]
            R_socket = place["R_socket"]

            for tw in cfg["twists"]:
                restore_peg(scene, body_pos0, body_quat0, sock_pos, sock_quat)
                d_L = np.asarray(tw["d_L"], dtype=np.float64)
                r_rad = np.asarray(tw["r_rad"], dtype=np.float64)
                d_world = R_socket @ (d_L * scene.char_len)
                # features BEFORE twist (predictive)
                x_tip = tip_lat_axis_features(scene.env)
                x_raw = raw_relation_features(scene)
                x_aff = contact_affordance_features(scene, twist_dir_world=d_world)

                out = apply_twist_and_label(
                    scene,
                    sock_pos=sock_pos,
                    sock_quat=sock_quat,
                    R_socket=R_socket,
                    d_L=d_L,
                    r_rad=r_rad,
                    settle_steps=int(cfg["physics"]["settle_steps"]),
                    gravity_off=bool(cfg["physics"]["gravity_off"]),
                    label_cfg=cfg["label"],
                )
                rows.append(
                    {
                        "family_id": spec.family_id,
                        "section": spec.section,
                        "size_mm": int(spec.nominal_size_mm),
                        "instance_key": instance_key,
                        "pose": pose["name"],
                        "twist": tw["name"],
                        "label": out["label"],
                        "feasible": bool(out["feasible"]),
                        "metrics": {
                            k: out[k]
                            for k in (
                                "progress_frac",
                                "lat_dev",
                                "force_mean_n",
                                "commanded_m",
                            )
                        },
                        "features": {
                            "tip_lat_axis": x_tip.tolist(),
                            "raw_relation": x_raw.tolist(),
                            "contact_affordance": x_aff.tolist(),
                        },
                    }
                )
    finally:
        scene.env.close()
    return rows


def stack_features(rows: list[dict], name: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray([r["features"][name] for r in rows], dtype=np.float64)
    y = np.asarray([int(r["feasible"]) for r in rows], dtype=np.int64)
    return X, y


def eval_family_loo(rows: list[dict], cfg: dict) -> dict[str, Any]:
    families = sorted({r["family_id"] for r in rows})
    feat_names = ["tip_lat_axis", "raw_relation", "contact_affordance"]
    folds = []
    for held in families:
        train = [r for r in rows if r["family_id"] != held]
        test = [r for r in rows if r["family_id"] == held]
        fold = {"held_family": held, "n_train": len(train), "n_test": len(test), "reps": {}}
        for fname in feat_names:
            Xtr, ytr = stack_features(train, fname)
            Xte, yte = stack_features(test, fname)
            fold["reps"][fname] = fit_predict_binary(
                Xtr,
                ytr,
                Xte,
                yte,
                C=float(cfg["probe"]["C"]),
                max_iter=int(cfg["probe"]["max_iter"]),
                seed=int(cfg["seed"]),
            )
        # shuffle negative on affordance
        rng = np.random.default_rng(int(cfg["seed"]) + hash(held) % 10000)
        Xte_s = shuffle_pairing(stack_features(test, "contact_affordance")[0], "contact_affordance", rng)
        yte = stack_features(test, "contact_affordance")[1]
        Xtr, ytr = stack_features(train, "contact_affordance")
        fold["reps"]["contact_affordance_shuffled"] = fit_predict_binary(
            Xtr,
            ytr,
            Xte_s,
            yte,
            C=float(cfg["probe"]["C"]),
            max_iter=int(cfg["probe"]["max_iter"]),
            seed=int(cfg["seed"]),
        )
        tip_acc = fold["reps"]["tip_lat_axis"]["accuracy"]
        aff_acc = fold["reps"]["contact_affordance"]["accuracy"]
        shuf_acc = fold["reps"]["contact_affordance_shuffled"]["accuracy"]
        fold["aff_minus_tip"] = float(aff_acc - tip_acc)
        fold["shuffle_drop"] = float(aff_acc - shuf_acc)
        folds.append(fold)
    return {"folds": folds, "families": families}


def eval_instance_holdout(primary_rows: list[dict], secondary_rows: list[dict], cfg: dict) -> dict:
    feat_names = ["tip_lat_axis", "raw_relation", "contact_affordance"]
    out = {}
    for fname in feat_names:
        Xtr, ytr = stack_features(primary_rows, fname)
        Xte, yte = stack_features(secondary_rows, fname)
        out[fname] = fit_predict_binary(
            Xtr,
            ytr,
            Xte,
            yte,
            C=float(cfg["probe"]["C"]),
            max_iter=int(cfg["probe"]["max_iter"]),
            seed=int(cfg["seed"]),
        )
    out["aff_minus_tip"] = float(
        out["contact_affordance"]["accuracy"] - out["tip_lat_axis"]["accuracy"]
    )
    return out


def judge(family_eval: dict, instance_eval: dict | None, cfg: dict) -> dict[str, Any]:
    min_gap = float(cfg["pass"]["min_aff_minus_tip_per_fold"])
    min_drop = float(cfg["pass"]["min_shuffle_drop"])
    folds = family_eval["folds"]
    all_beat = all(f["aff_minus_tip"] >= min_gap for f in folds)
    all_shuffle = all(f["shuffle_drop"] >= min_drop for f in folds)
    # same-section-only check: compare mean gap on cross-section folds
    # if held family section differs from majority train — already LOO
    mean_gap = float(np.mean([f["aff_minus_tip"] for f in folds]))
    mean_drop = float(np.mean([f["shuffle_drop"] for f in folds]))
    inst_ok = True
    if instance_eval is not None:
        inst_ok = float(instance_eval["aff_minus_tip"]) >= 0.0

    passed = bool(all_beat and all_shuffle and inst_ok)
    if not passed:
        verdict = "fail_stop_affordance_direction"
        summary = (
            "contact-affordance 未在全部 family held-out 上稳定优于 tip/lat/axis，"
            "或 shuffle/instance 负对照未过；停止该方向。"
        )
    else:
        verdict = "pass_affordance_representation"
        summary = "affordance 跨几何 held-out 优于固定 tip/lat/axis，且 shuffle 下降。"
    return {
        "pass": passed,
        "verdict": verdict,
        "summary": summary,
        "all_family_folds_beat_tip": all_beat,
        "all_family_folds_shuffle_drop": all_shuffle,
        "instance_holdout_ok": inst_ok,
        "mean_aff_minus_tip": mean_gap,
        "mean_shuffle_drop": mean_drop,
        "min_aff_minus_tip_required": min_gap,
        "min_shuffle_drop_required": min_drop,
        "stop_direction": (not passed),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "cross_geometry_affordance_p0.yaml",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    out_dir = PROJECT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = PROJECT_ROOT / cfg["manifest_path"]
    man_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / cfg["report_path"]

    specs = load_specs(cfg)
    frozen = {
        "protocol": PROTOCOL,
        "frozen_at": _utc(),
        "families": [s.family_id for s in specs],
        "poses": cfg["poses"],
        "twists": cfg["twists"],
    }
    (out_dir / "protocol_frozen.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    all_rows: list[dict] = []
    for spec in specs:
        print(f"[collect] {spec.family_id}", flush=True)
        rows = collect_family_rows(spec, cfg, instance_key="primary")
        all_rows.extend(rows)
        (out_dir / f"rows_{spec.family_id}.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )

    label_counts = {}
    for r in all_rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    family_eval = eval_family_loo(all_rows, cfg)

    # instance holdout: train on primary of instance families, test on secondary
    inst_rows_pri = []
    inst_rows_sec = []
    for fid in cfg["instance_holdout_families"]:
        spec = next(s for s in specs if s.family_id == fid)
        print(f"[instance] {fid} secondary", flush=True)
        inst_rows_pri.extend([r for r in all_rows if r["family_id"] == fid])
        inst_rows_sec.extend(
            collect_family_rows(spec, cfg, instance_key=cfg["secondary_instance_key"])
        )
    instance_eval = eval_instance_holdout(inst_rows_pri, inst_rows_sec, cfg)

    verdict = judge(family_eval, instance_eval, cfg)

    manifest = {
        "protocol": PROTOCOL,
        "finished_at": _utc(),
        "n_rows": len(all_rows),
        "label_counts": label_counts,
        "feasible_rate": float(np.mean([r["feasible"] for r in all_rows])),
        "family_loo": family_eval,
        "instance_holdout": instance_eval,
        "verdict": verdict,
        "config": cfg,
    }
    man_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, default=float), encoding="utf-8"
    )

    lines = [
        "# Cross-Geometry Contact-Affordance P0 Result",
        "",
        f"- 完成：{_utc()}",
        f"- 判定：`{verdict['verdict']}`",
        f"- 通过：{verdict['pass']}",
        f"- 停止该方向：{verdict['stop_direction']}",
        f"- 摘要：{verdict['summary']}",
        "",
        f"- 样本数：{len(all_rows)}；标签：{label_counts}；feasible_rate={manifest['feasible_rate']:.3f}",
        f"- mean aff−tip：{verdict['mean_aff_minus_tip']:.3f}（门槛 {verdict['min_aff_minus_tip_required']}）",
        f"- mean shuffle drop：{verdict['mean_shuffle_drop']:.3f}（门槛 {verdict['min_shuffle_drop_required']}）",
        f"- instance aff−tip：{instance_eval['aff_minus_tip']:.3f}",
        "",
        "## Family LOO folds",
        "",
        "```json",
        json.dumps(family_eval, indent=2, default=float),
        "```",
        "",
        "## Instance holdout",
        "",
        "```json",
        json.dumps(instance_eval, indent=2, default=float),
        "```",
        "",
        "## 决策",
        "",
    ]
    if verdict["stop_direction"]:
        lines += [
            "- **立即停止** Cross-Geometry Contact-Affordance 方向。",
            "- 不训练 generalist insertion policy。",
            "- 下一步应审查阶段接口与数据支持，而非宣称仿真不可解。",
            "",
        ]
    else:
        lines += ["- 表示硬门通过；仍禁止直接开训，需另批设计数据/接口。", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    state = {
        "date": "2026-08-15",
        "phase": "cross_geometry_affordance_p0_complete",
        "busy": False,
        "training_allowed": False,
        "collection_allowed": False,
        "wrapper_allowed": False,
        "compliance_line": {"status": "abandoned"},
        "cross_geometry_affordance_p0": {
            "pass": verdict["pass"],
            "verdict": verdict["verdict"],
            "stop_direction": verdict["stop_direction"],
            "manifest": str(man_path.relative_to(PROJECT_ROOT)),
            "report": str(report_path.relative_to(PROJECT_ROOT)),
        },
        "next_action": (
            "review_stage_interface_and_data_support"
            if verdict["stop_direction"]
            else "design_next_gate_no_training"
        ),
    }
    (PROJECT_ROOT / "outputs" / "state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )

    print(json.dumps({"verdict": verdict["verdict"], "pass": verdict["pass"]}, indent=2))
    return 0 if verdict["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
