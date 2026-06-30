#!/usr/bin/env python3
"""Fast hold-out eval for action44 ckpt selection (no video)."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from interaction_retarget.skill_replay.insert import run_pose_insert_phase
from pose_insert.paths import default_poseinsert_data_dir
from pose_insert.pre_insert import resolve_peg_lift_end_frame
from pose_insert.splits import VAL_EPISODE_INDICES


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--action-blend", type=float, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    sidecar = default_sidecar_dir(TASK_ID)
    data_root = args.data_root if args.data_root is not None else default_poseinsert_data_dir(TASK_ID)
    manifest = json.loads((sidecar / "manifest.json").read_text(encoding="utf-8"))
    entries = [
        e
        for e in manifest["episodes"]
        if int(e["episode_index"]) in VAL_EPISODE_INDICES
        and e.get("timing", {}).get("peg_lift_start") is not None
    ]
    passed = 0
    rows = []
    for entry in sorted(entries, key=lambda e: int(e["episode_index"])):
        ep = int(entry["episode_index"])
        peg_end = resolve_peg_lift_end_frame(entry, sidecar)
        env = make_assembly_env(seed=int(args.seed), randomize=False)
        raw = env.unwrapped
        try:
            kw = dict(
                reach_mode="demo_replay",
                max_steps=int(args.max_steps),
                poseinsert_ckpt=args.ckpt.expanduser(),
                poseinsert_data_root=data_root,
                manifest_entry=entry,
                sidecar_dir=sidecar,
                peg_lift_end_frame=peg_end,
                insert_mode="action44",
            )
            report = run_pose_insert_phase(env, raw, **kw)
        finally:
            env.close()
        ok = bool(report.success and report.insert_ok)
        passed += int(ok)
        rows.append({"ep": ep, "ok": ok, "reason": report.fail_reason})
        print(f"ep{ep} ok={ok} reason={report.fail_reason}", flush=True)
    total = len(rows)
    rate = passed / max(1, total)
    print(f"holdout: {passed}/{total} ({100*rate:.1f}%) ckpt={args.ckpt}", flush=True)
    return 0 if passed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
