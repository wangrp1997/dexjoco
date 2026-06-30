#!/usr/bin/env python3
"""Batch eval action44 PoseInsert on many demos + checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEXJOJO_ROOT = _REPO_ROOT / "dexjoco"
if str(_DEXJOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXJOJO_ROOT))

from interaction_retarget.constants import TASK_ID, default_sidecar_dir
from interaction_retarget.sim.replay import make_assembly_env
from interaction_retarget.skill_replay.insert import InsertReport, run_pose_insert_phase
from pose_insert.paths import default_poseinsert_data_dir
from pose_insert.pre_insert import resolve_peg_lift_end_frame


@dataclass
class Row:
    episode_index: int
    success: bool
    insert_ok: bool
    fail_reason: str
    steps: int = 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description= __doc__)
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("/mnt/hdd/dexjoco/poseinsert_sim/checkpoints/action44"),
    )
    p.add_argument("--ckpts", type=str, default="all", help="comma epochs e.g. 100,200,300 or 'all'")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _manifest_entries(sidecar_dir: Path) -> list[dict]:
    manifest = json.loads((sidecar_dir / "manifest.json").read_text(encoding="utf-8"))
    out: list[dict] = []
    for entry in manifest["episodes"]:
        timing = entry.get("timing", {})
        if timing.get("peg_lift_start") is None or timing.get("right_grasp_frame") is None:
            continue
        out.append(entry)
    return out


def _resolve_ckpts(ckpt_dir: Path, spec: str) -> list[Path]:
    if spec.strip().lower() == "all":
        paths = sorted(ckpt_dir.glob("policy_epoch_*_seed_*.ckpt"))
        last = ckpt_dir / "policy_last.ckpt"
        if last.is_file():
            paths.append(last)
        return paths
    out: list[Path] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part == "last":
            out.append(ckpt_dir / "policy_last.ckpt")
        else:
            out.append(ckpt_dir / f"policy_epoch_{int(part)}_seed_233.ckpt")
    return [p for p in out if p.is_file()]


def _eval_one(entry: dict, *, ckpt: Path, data_root: Path, sidecar_dir: Path, seed: int, max_steps: int) -> Row:
    ep = int(entry["episode_index"])
    peg_lift_end = resolve_peg_lift_end_frame(entry, sidecar_dir)
    env = make_assembly_env(seed=int(seed), randomize=False)
    raw = env.unwrapped
    try:
        report: InsertReport = run_pose_insert_phase(
            env,
            raw,
            reach_mode="demo_replay",
            max_steps=int(max_steps),
            poseinsert_ckpt=ckpt,
            poseinsert_data_root=data_root,
            manifest_entry=entry,
            sidecar_dir=sidecar_dir,
            peg_lift_end_frame=peg_lift_end,
            insert_mode="action44",
        )
    finally:
        env.close()
    return Row(
        episode_index=ep,
        success=bool(report.success),
        insert_ok=bool(report.insert_ok),
        fail_reason=str(report.fail_reason),
        steps=int(report.steps),
    )


def main() -> int:
    args = _parse_args()
    sidecar_dir = default_sidecar_dir(TASK_ID)
    data_root = args.data_root if args.data_root is not None else default_poseinsert_data_dir(TASK_ID)
    ckpts = _resolve_ckpts(args.ckpt_dir.expanduser(), args.ckpts)
    if not ckpts:
        print("no checkpoints", file=sys.stderr)
        return 1
    entries = _manifest_entries(sidecar_dir)
    out_path = args.out if args.out is not None else data_root / "batch_eval_action44.json"

    results: list[dict] = []
    for ckpt in ckpts:
        rows: list[Row] = []
        passed = 0
        insert_ok_n = 0
        for entry in entries:
            row = _eval_one(
                entry,
                ckpt=ckpt,
                data_root=data_root,
                sidecar_dir=sidecar_dir,
                seed=int(args.seed),
                max_steps=int(args.max_steps),
            )
            rows.append(row)
            if row.success:
                passed += 1
            if row.insert_ok:
                insert_ok_n += 1
        total = len(rows)
        summary = {
            "ckpt": str(ckpt),
            "passed": passed,
            "insert_ok": insert_ok_n,
            "total": total,
            "success_rate": passed / max(1, total),
            "insert_ok_rate": insert_ok_n / max(1, total),
            "episodes": [asdict(r) for r in rows],
        }
        results.append(summary)
        print(
            f"{ckpt.name}: success {passed}/{total} ({100*passed/total:.1f}%) "
            f"insert_ok {insert_ok_n}/{total} ({100*insert_ok_n/total:.1f}%)",
            flush=True,
        )

    best = max(results, key=lambda r: (r["insert_ok"], r["passed"]))
    payload = {"checkpoints": results, "best_ckpt": best["ckpt"], "best_insert_ok_rate": best["insert_ok_rate"]}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"best: {best['ckpt']} insert_ok={best['insert_ok']}/{best['total']}", flush=True)
    print(f"saved -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
