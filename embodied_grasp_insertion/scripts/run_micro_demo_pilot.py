#!/usr/bin/env python3
"""Micro-demo pilot v0 — dry-run ONLY (hardened config/path guards).

No trajectory writes. WRITE_IMPLEMENTATION_ENABLED remains False.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEXJOCO_ROOT = PROJECT_ROOT.parent
REACH_ROOT = DEXJOCO_ROOT.parent / "reach_insert_rl"
LAI_ROOT = DEXJOCO_ROOT.parent / "lai"
for _p in (
    str(PROJECT_ROOT),
    str(DEXJOCO_ROOT),
    str(DEXJOCO_ROOT / "dexjoco"),
    str(REACH_ROOT),
    str(LAI_ROOT),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from embodied_grasp_insertion.pilot import (  # noqa: E402
    ALLOWED_OUT_ROOT,
    MAX_EPISODES_PER_FAMILY,
    MAX_FAMILIES,
    MAX_HORIZON_STEPS,
    MAX_TOTAL_TRAJECTORIES,
    MAX_TRAJECTORIES_PER_EPISODE,
    MIN_HORIZON_STEPS,
    PILOT_TAG,
    WRITE_IMPLEMENTATION_ENABLED,
)
from embodied_grasp_insertion.pilot.config_schema import (  # noqa: E402
    PilotConfigError,
    validate_pilot_config,
)
from embodied_grasp_insertion.pilot.dry_run import run_round8_demo_dry_gates  # noqa: E402
from embodied_grasp_insertion.pilot.paths import (  # noqa: E402
    PilotPathError,
    write_dry_run_report_atomic,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Micro-demo pilot v0 dry-run only")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/micro_demo_pilot.yaml"),
    )
    parser.add_argument(
        "--demo-config",
        default=str(PROJECT_ROOT / "configs/finger_controllability_smoke.yaml"),
    )
    parser.add_argument(
        "--dry-run-report",
        default="",
        help="Optional JSON report; must be new file under /tmp (O_EXCL|O_NOFOLLOW).",
    )
    parser.add_argument("--allow-write", action="store_true")
    parser.add_argument("--i-understand-pilot-is-revocable", action="store_true")
    parser.add_argument(
        "--set-dry-run-false",
        action="store_true",
        help="Sets dry_run=false in memory; still refused while write impl disabled.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg_bytes = cfg_path.read_bytes()
    raw_cfg = yaml.safe_load(cfg_bytes.decode("utf-8"))
    demo_cfg = yaml.safe_load(Path(args.demo_config).read_text(encoding="utf-8"))

    result: dict[str, Any] = {
        "protocol": PILOT_TAG,
        "created_at": _utc(),
        "WRITE_IMPLEMENTATION_ENABLED": WRITE_IMPLEMENTATION_ENABLED,
        "allow_write_flag": bool(args.allow_write),
        "revocable_flag": bool(args.i_understand_pilot_is_revocable),
        "allowed_out_root": str(ALLOWED_OUT_ROOT),
        "code_caps": {
            "MAX_FAMILIES": MAX_FAMILIES,
            "MAX_TOTAL_TRAJECTORIES": MAX_TOTAL_TRAJECTORIES,
            "MAX_EPISODES_PER_FAMILY": MAX_EPISODES_PER_FAMILY,
            "MAX_TRAJECTORIES_PER_EPISODE": MAX_TRAJECTORIES_PER_EPISODE,
            "MIN_HORIZON_STEPS": MIN_HORIZON_STEPS,
            "MAX_HORIZON_STEPS": MAX_HORIZON_STEPS,
        },
        "config_sha256": _config_hash(cfg_bytes),
        "demo_config_sha256": _sha256_file(Path(args.demo_config)),
        "seed": int(demo_cfg.get("seed", 0)),
        "training_forbidden": True,
        "disk_writes": [],
        "trajectories": [],
    }

    # Schema validation BEFORE env creation.
    try:
        cfg = validate_pilot_config(raw_cfg)
    except PilotConfigError as e:
        result["verdict"] = "aborted"
        result["reason"] = f"config_schema: {e}"
        print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))
        _maybe_write_tmp_report(args.dry_run_report, result)
        return 3

    dry_run = bool(cfg.dry_run)
    if args.set_dry_run_false:
        dry_run = False
    result["dry_run"] = dry_run
    result["validated_caps"] = {
        "max_families": cfg.max_families,
        "max_total_trajectories": cfg.max_total_trajectories,
        "max_horizon_steps": cfg.max_horizon_steps,
        "families": list(cfg.families),
        "gates": dict(cfg.gates),
    }
    write_requested = bool(
        args.allow_write and args.i_understand_pilot_is_revocable and (not dry_run)
    )
    result["write_requested"] = write_requested

    if args.allow_write or args.i_understand_pilot_is_revocable or (not dry_run) or write_requested:
        result["verdict"] = "refused"
        result["reason"] = (
            "write path disabled: WRITE_IMPLEMENTATION_ENABLED=False; "
            "v0 runner is dry-run-only (in-memory, no out_root/manifest writes)"
        )
        print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))
        _maybe_write_tmp_report(args.dry_run_report, result)
        return 2

    if list(cfg.families) != ["round_8mm"]:
        result["verdict"] = "aborted"
        result["reason"] = "v0 dry-run only supports families=[round_8mm]"
        print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))
        _maybe_write_tmp_report(args.dry_run_report, result)
        return 3

    if cfg.max_total_trajectories < 1:
        result["verdict"] = "aborted"
        result["reason"] = "max_total_trajectories must be >= 1"
        print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))
        _maybe_write_tmp_report(args.dry_run_report, result)
        return 3

    try:
        attempt = run_round8_demo_dry_gates(
            demo_cfg,
            max_horizon_steps=cfg.max_horizon_steps,
            enabled_gates=dict(cfg.gates),
        )
    except Exception as e:
        result["verdict"] = "aborted"
        result["reason"] = f"dry_run_exception: {type(e).__name__}: {e}"
        print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))
        _maybe_write_tmp_report(args.dry_run_report, result)
        return 4

    result["trajectories"].append(
        {
            "traj_id": attempt["traj_id"],
            "passed": attempt["passed"],
            "path": None,
            "buffer_keys": sorted(attempt["buffer"].keys()),
            "labels": attempt["buffer"]["labels"],
            "meta": attempt["buffer"]["meta"],
        }
    )

    if not attempt["passed"]:
        result["verdict"] = "aborted"
        result["reason"] = attempt["buffer"]["labels"].get("stop_reason") or "gate_failure"
        print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))
        _maybe_write_tmp_report(args.dry_run_report, result)
        return 5

    used = int(attempt["buffer"]["meta"].get("horizon_steps_used", -1))
    if used < 0 or used > cfg.max_horizon_steps:
        result["verdict"] = "aborted"
        result["reason"] = f"horizon_accounting_invalid used={used} max={cfg.max_horizon_steps}"
        print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))
        _maybe_write_tmp_report(args.dry_run_report, result)
        return 5

    result["verdict"] = "dry_run_ok"
    result["reason"] = "in_memory_gates_passed; no_disk_writes"
    result["note"] = (
        "Not a learned grasp; not an insertion demo; write not authorized; "
        "formal manifest not written."
    )
    print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))
    _maybe_write_tmp_report(args.dry_run_report, result)
    return 0


def _maybe_write_tmp_report(path: str, result: dict[str, Any]) -> None:
    if not path:
        return
    try:
        dest = write_dry_run_report_atomic(
            path, json.dumps(_jsonable(result), indent=2, ensure_ascii=False) + "\n"
        )
    except PilotPathError as e:
        print(json.dumps({"dry_run_report_error": str(e)}, ensure_ascii=False), flush=True)
        return
    except OSError as e:
        print(json.dumps({"dry_run_report_error": str(e)}, ensure_ascii=False), flush=True)
        return
    print(json.dumps({"dry_run_report_written": str(dest)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
