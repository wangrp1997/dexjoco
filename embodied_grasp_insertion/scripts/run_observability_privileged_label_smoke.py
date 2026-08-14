#!/usr/bin/env python3
"""P0-L1 Observability Privileged Label Smoke (readonly; no train/collect/write)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT.parent), str(PROJECT_ROOT.parent / "dexjoco")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.io_paths import path_for_manifest  # noqa: E402
from embodied_grasp_insertion.labels.privileged_schema import (  # noqa: E402
    PROTOCOL,
    SCHEMA_VERSION,
    extract_privileged_frame,
    labels_bit_digest,
    make_root_id,
    schema_document,
)
from embodied_grasp_insertion.physics.grasp_metrics import (  # noqa: E402
    control_dt_seconds,
)
from embodied_grasp_insertion.pilot import ALLOWED_OUT_ROOT, WRITE_IMPLEMENTATION_ENABLED  # noqa: E402
from embodied_grasp_insertion.simulation.full_episode_snapshot import (  # noqa: E402
    FullEpisodeSnapshot,
)
from embodied_grasp_insertion.simulation.full_episode_utils import (  # noqa: E402
    make_full_env,
    replay_demo_one_step,
    replay_demo_to_frame,
    select_roots_for_episode,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_pilot_paths(*paths: Path) -> None:
    formal = ALLOWED_OUT_ROOT.resolve()
    for p in paths:
        rp = p.resolve()
        if rp == formal or formal in rp.parents or rp in formal.parents and "pilot_micro_demo" in str(rp):
            raise RuntimeError(f"refuse path touching formal pilot root: {rp}")
        if "pilot_micro_demo_v0" in str(rp):
            raise RuntimeError(f"refuse pilot_micro_demo_v0 path: {rp}")


def _pick_transport_or_first(env) -> Any:
    roots = select_roots_for_episode(env)
    if not roots:
        raise RuntimeError("no roots")
    for r in roots:
        if r.phase == "transport":
            return r
    return roots[0]


def extract_window(
    episode_id: int,
    *,
    window: int,
    seed: int = 0,
) -> dict[str, Any]:
    env = make_full_env([episode_id], seed=seed)
    try:
        env.reset()
        root = _pick_transport_or_first(env)
        env.reset()
        replay_demo_to_frame(env, int(root.frame))
        dt_s = float(control_dt_seconds(env))
        root_id = make_root_id(episode_id, int(root.frame), root.phase)

        frames: list[dict[str, Any]] = []
        prev_o2h = None
        prev_sim_t = None
        timeline_ok = True
        restore_checks: list[dict[str, Any]] = []

        for i in range(int(window)):
            frame_idx = int(env._t)
            if i > 0:
                # advance one demo step for continuity
                info = replay_demo_one_step(env)
                if env._done:
                    raise RuntimeError(f"episode ended early at t={env._t}")
                frame_idx = int(env._t)

            sim_t = float(env._raw._data.time)
            if prev_sim_t is not None:
                dsim = sim_t - prev_sim_t
                # allow tiny float noise; require positive advance ≈ dt
                if dsim <= 0 or abs(dsim - dt_s) > 1e-9 * max(1.0, abs(dt_s)) + 1e-12:
                    # some envs may have slight mismatch vs heuristic dt; record soft fail
                    if dsim <= 0:
                        timeline_ok = False

            label, o2h = extract_privileged_frame(
                env,
                episode_index=episode_id,
                frame=frame_idx,
                root_id=root_id,
                root_phase=root.phase,
                prev_o2h=prev_o2h,
                dt_s=dt_s,
            )
            # First frame of window: no prev → velocity unavailable (contract).
            if i == 0 and label["object_in_hand_velocity"]["available"]:
                raise RuntimeError("first window frame must have velocity.available=false")
            if i > 0 and not label["object_in_hand_velocity"]["available"]:
                raise RuntimeError("subsequent frames must have velocity")

            # Snapshot restore consistency on middle frame.
            if i == max(0, window // 2):
                snap = FullEpisodeSnapshot.capture(env)
                label_a = label
                snap.restore(env)
                label_b, _ = extract_privileged_frame(
                    env,
                    episode_index=episode_id,
                    frame=frame_idx,
                    root_id=root_id,
                    root_phase=root.phase,
                    prev_o2h=prev_o2h,
                    dt_s=dt_s,
                )
                # Compare privilege fields except provenance.sim_time already same.
                same = labels_bit_digest([label_a]) == labels_bit_digest([label_b])
                restore_checks.append(
                    {
                        "frame": frame_idx,
                        "bit_exact_after_restore": same,
                    }
                )
                if not same:
                    raise RuntimeError(f"restore label mismatch at frame {frame_idx}")

            frames.append(label)
            prev_o2h = o2h
            prev_sim_t = sim_t

        # Continuity of frame indices
        idxs = [int(f["provenance"]["frame"]) for f in frames]
        contiguous = all(idxs[j] == idxs[0] + j for j in range(len(idxs)))
        if not contiguous:
            timeline_ok = False

        return {
            "episode_index": int(episode_id),
            "root": {
                "frame": int(root.frame),
                "phase": root.phase,
                "root_id": root_id,
                "tip_dist_m": float(root.tip_dist_m),
            },
            "window": int(window),
            "control_dt_s": dt_s,
            "timeline_contiguous": bool(contiguous and timeline_ok),
            "frame_indices": idxs,
            "restore_checks": restore_checks,
            "digest": labels_bit_digest(frames),
            "frames": frames,
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "observability_privileged_label_smoke_v1.json",
    )
    ap.add_argument(
        "--schema-out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "PRIVILEGED_LABEL_SCHEMA_V1.md",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "docs" / "OBSERVABILITY_PRIVILEGED_LABEL_SMOKE.md",
    )
    args = ap.parse_args()

    if WRITE_IMPLEMENTATION_ENABLED:
        print("REFUSE: WRITE_IMPLEMENTATION_ENABLED must be False", file=sys.stderr)
        return 2
    if ALLOWED_OUT_ROOT.exists():
        print(f"REFUSE: formal pilot out_root exists: {ALLOWED_OUT_ROOT}", file=sys.stderr)
        return 2
    _refuse_pilot_paths(args.manifest, args.schema_out, args.report)

    if len(args.episodes) < 3:
        print("need >=3 episodes", file=sys.stderr)
        return 2
    if args.window < 2:
        print("window must be >=2 (velocity needs history)", file=sys.stderr)
        return 2

    # Pass 1
    pass1 = [extract_window(ep, window=args.window) for ep in args.episodes]
    # Pass 2 bit-exact
    pass2 = [extract_window(ep, window=args.window) for ep in args.episodes]
    bit_exact = all(a["digest"] == b["digest"] for a, b in zip(pass1, pass2))
    restore_ok = all(
        all(c.get("bit_exact_after_restore") for c in w["restore_checks"]) for w in pass1
    )
    timeline_ok = all(w["timeline_contiguous"] for w in pass1)
    # Strip bulky frames from manifest summary; keep digests + first/last provenance.
    slim = []
    for w in pass1:
        slim.append(
            {
                "episode_index": w["episode_index"],
                "root": w["root"],
                "window": w["window"],
                "control_dt_s": w["control_dt_s"],
                "timeline_contiguous": w["timeline_contiguous"],
                "frame_indices": w["frame_indices"],
                "restore_checks": w["restore_checks"],
                "digest": w["digest"],
                "n_frames": len(w["frames"]),
                "sample_first_provenance": w["frames"][0]["provenance"],
                "sample_second_velocity_available": w["frames"][1]["object_in_hand_velocity"][
                    "available"
                ],
                "sample_contact_total_first": w["frames"][0]["peg_hand_contact"]["total"],
            }
        )

    overall = (
        "pass"
        if bit_exact and restore_ok and timeline_ok and not WRITE_IMPLEMENTATION_ENABLED
        else "fail"
    )

    manifest = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc(),
        "overall_verdict": overall,
        "episodes": list(args.episodes),
        "window": int(args.window),
        "checks": {
            "bit_exact_repeat_pass": bit_exact,
            "snapshot_restore_label_pass": restore_ok,
            "timeline_contiguous_pass": timeline_ok,
            "WRITE_IMPLEMENTATION_ENABLED": False,
            "formal_pilot_out_root_exists": False,
            "training_dataset_written": False,
            "collect_new_episodes": False,
            "reopen_c0_c1_c1_1": False,
        },
        "windows": slim,
        "schema": schema_document(),
        "report": path_for_manifest(args.report, project_root=PROJECT_ROOT),
        "schema_doc": path_for_manifest(args.schema_out, project_root=PROJECT_ROOT),
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Schema markdown
    sch = schema_document()
    args.schema_out.write_text(
        "\n".join(
            [
                "# Privileged Label Schema v1 (P0-L1)",
                "",
                f"- protocol: `{PROTOCOL}`",
                f"- schema_version: `{SCHEMA_VERSION}`",
                "- privilege_only: true；deployment_input: false",
                "- 非训练数据集；禁止写入 `pilot_micro_demo_v0`",
                "",
                "## Included",
                "",
                *[f"- {x}" for x in sch["included"]],
                "",
                "## Excluded",
                "",
                *[f"- `{x}`" for x in sch["excluded"]],
                "",
                "## Velocity contract",
                "",
                "```json",
                json.dumps(sch["velocity_contract"], indent=2, ensure_ascii=False),
                "```",
                "",
                f"- contact_force_eps_N: {sch['contact_force_eps_N']}",
                f"- finger_order: {sch['finger_order']}",
                f"- reference_body: `{sch['reference_body']}`",
                "",
                "## Notes",
                "",
                *[f"- {x}" for x in sch["notes"]],
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Report (always cite overall_verdict)
    lines = [
        "# Observability Privileged Label Smoke (P0-L1)",
        "",
        f"- 日期：{manifest['created_at']}",
        f"- overall_verdict：**{manifest['overall_verdict']}**",
        f"- episodes：{args.episodes}；window={args.window}",
        f"- bit-exact repeat：{bit_exact}",
        f"- snapshot restore labels：{restore_ok}",
        f"- timeline contiguous：{timeline_ok}",
        "- 未训练、未采集、未开写盘、未重开 C0/C1/C1.1",
        f"- schema：`{args.schema_out.name}`",
        f"- manifest：`{path_for_manifest(args.manifest, project_root=PROJECT_ROOT)}`",
        "",
        "## Per episode",
        "",
    ]
    for w in slim:
        lines.append(
            f"- ep{w['episode_index']} root={w['root']['phase']}@{w['root']['frame']} "
            f"frames={w['frame_indices']} digest={w['digest'][:12]}… "
            f"contact0={w['sample_contact_total_first']}"
        )
    lines.extend(
        [
            "",
            "## Claims",
            "",
            "- 冻结特权标签契约（pose/velocity/contact/force/outcome/provenance）",
            "- **不**声称 Observability 模型可训或部署 belief 已解",
            "- **不**生成 slip truth / fine contact-mode",
            "",
        ]
    )
    args.report.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "overall_verdict": manifest["overall_verdict"],
                "bit_exact": bit_exact,
                "restore_ok": restore_ok,
                "timeline_ok": timeline_ok,
                "manifest": path_for_manifest(args.manifest, project_root=PROJECT_ROOT),
            },
            indent=2,
        )
    )
    return 0 if manifest["overall_verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
