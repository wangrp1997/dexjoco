"""Schemas for pilot trajectory meta/labels/manifest/states (write-design)."""

from __future__ import annotations

import json
import uuid
from typing import Any

import numpy as np

from embodied_grasp_insertion.pilot import (
    FORBIDDEN_NPZ_OBJECT_DTYPE,
    MAX_HORIZON_STEPS,
    MAX_STATES_NPZ_BYTES,
    MIN_HORIZON_STEPS,
    PILOT_TAG,
)


class PilotSchemaError(ValueError):
    pass


_ROOT_SOURCES = frozenset({"demo_transport", "oracle_establish_formal"})
_MANIFEST_VERDICTS = frozenset({"write_ok", "aborted", "refused", "incomplete"})


def _require_uuid(traj_id: str) -> str:
    try:
        uuid.UUID(str(traj_id))
    except ValueError as e:
        raise PilotSchemaError(f"traj_id must be UUID, got {traj_id!r}") from e
    return str(traj_id)


def _require_bool(d: dict[str, Any], key: str, *, expected: bool | None = None) -> bool:
    if key not in d:
        raise PilotSchemaError(f"missing field: {key}")
    v = d[key]
    if type(v) is not bool:
        raise PilotSchemaError(f"{key} must be bool")
    if expected is not None and v is not expected:
        raise PilotSchemaError(f"{key} must be {expected}, got {v}")
    return v


def _require_str(d: dict[str, Any], key: str) -> str:
    if key not in d:
        raise PilotSchemaError(f"missing field: {key}")
    v = d[key]
    if not isinstance(v, str) or not v:
        raise PilotSchemaError(f"{key} must be non-empty str")
    return v


def _require_int(d: dict[str, Any], key: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if key not in d:
        raise PilotSchemaError(f"missing field: {key}")
    v = d[key]
    if type(v) is not int or isinstance(v, bool):
        raise PilotSchemaError(f"{key} must be int")
    if minimum is not None and v < minimum:
        raise PilotSchemaError(f"{key} must be >= {minimum}")
    if maximum is not None and v > maximum:
        raise PilotSchemaError(f"{key} must be <= {maximum}")
    return v


def validate_meta(meta: Any, *, require_dry_run_false: bool = True) -> dict[str, Any]:
    if not isinstance(meta, dict):
        raise PilotSchemaError("meta must be object")
    _require_uuid(str(meta.get("traj_id", "")))
    if meta.get("pilot_tag") != PILOT_TAG:
        raise PilotSchemaError(f"pilot_tag must be {PILOT_TAG}")
    _require_bool(meta, "training_forbidden", expected=True)
    dry = _require_bool(meta, "dry_run")
    if require_dry_run_false and dry is not False:
        raise PilotSchemaError("committed meta.dry_run must be false")
    _require_str(meta, "geometry_family_id")
    _require_str(meta, "target_instance_id")
    _require_str(meta, "socket_site")
    rs = _require_str(meta, "root_source")
    if rs not in _ROOT_SOURCES:
        raise PilotSchemaError(f"root_source invalid: {rs}")
    _require_bool(meta, "matched_snapshot_branch", expected=True)
    snap_after = meta.get("snap_call_count_after_establish")
    if type(snap_after) is not int or isinstance(snap_after, bool) or snap_after != 0:
        raise PilotSchemaError("snap_call_count_after_establish must be int 0")
    _require_bool(meta, "is_insertion_demo", expected=False)
    _require_str(meta, "created_at")
    used = _require_int(meta, "horizon_steps_used", minimum=0, maximum=MAX_HORIZON_STEPS)
    budget = _require_int(
        meta,
        "horizon_budget_max",
        minimum=MIN_HORIZON_STEPS,
        maximum=MAX_HORIZON_STEPS,
    )
    if used > budget:
        raise PilotSchemaError(
            f"horizon_steps_used={used} exceeds horizon_budget_max={budget}"
        )
    if "oracle_usage" not in meta or not isinstance(meta["oracle_usage"], dict):
        raise PilotSchemaError("oracle_usage must be object")
    if rs == "demo_transport":
        _require_int(meta, "episode_index", minimum=0)
        _require_int(meta, "root_frame", minimum=0)
    return meta


def validate_labels(labels: Any) -> dict[str, Any]:
    if not isinstance(labels, dict):
        raise PilotSchemaError("labels must be object")
    gates = labels.get("gates")
    if not isinstance(gates, list) or not gates:
        raise PilotSchemaError("labels.gates must be non-empty list")
    names = set()
    for g in gates:
        if not isinstance(g, dict) or "name" not in g or "passed" not in g:
            raise PilotSchemaError("each gate needs name/passed")
        if type(g["passed"]) is not bool:
            raise PilotSchemaError("gate.passed must be bool")
        names.add(g["name"])
    for req in ("physical_grasp", "target_hole_semantics", "insert_label_consistency"):
        if req not in names:
            raise PilotSchemaError(f"missing gate {req}")
    if labels.get("all_gates_passed") is not True:
        raise PilotSchemaError("all_gates_passed must be true to commit")
    if any(not g["passed"] for g in gates):
        raise PilotSchemaError("all gates must have passed=true")
    if labels.get("insert_phase") != "skipped":
        raise PilotSchemaError("insert_phase must be skipped")
    if labels.get("insert_ok") is not False:
        raise PilotSchemaError("insert_ok must be false")
    if labels.get("is_insertion_demo") is not False:
        raise PilotSchemaError("is_insertion_demo must be false")
    if labels.get("stop_reason") is not None:
        raise PilotSchemaError("stop_reason must be null on success")
    return labels


def validate_states_arrays(arrays: dict[str, np.ndarray]) -> None:
    if not arrays:
        raise PilotSchemaError("states arrays empty")
    total = 0
    for k, arr in arrays.items():
        if not isinstance(arr, np.ndarray):
            raise PilotSchemaError(f"states[{k}] must be ndarray")
        if FORBIDDEN_NPZ_OBJECT_DTYPE and arr.dtype == object:
            raise PilotSchemaError(f"states[{k}] object dtype forbidden")
        if arr.dtype.kind not in "iufb":
            raise PilotSchemaError(f"states[{k}] dtype {arr.dtype} not allowed")
        total += int(arr.nbytes)
    if total > MAX_STATES_NPZ_BYTES:
        raise PilotSchemaError(f"states nbytes {total} > {MAX_STATES_NPZ_BYTES}")


def validate_run_manifest(man: Any) -> dict[str, Any]:
    if not isinstance(man, dict):
        raise PilotSchemaError("manifest must be object")
    if man.get("protocol") != PILOT_TAG:
        raise PilotSchemaError("manifest.protocol mismatch")
    _require_uuid(str(man.get("run_id", "")))
    _require_str(man, "created_at")
    _require_bool(man, "dry_run")
    if "WRITE_IMPLEMENTATION_ENABLED" not in man or type(man["WRITE_IMPLEMENTATION_ENABLED"]) is not bool:
        raise PilotSchemaError("WRITE_IMPLEMENTATION_ENABLED must be bool")
    if "trajectories" not in man or not isinstance(man["trajectories"], list):
        raise PilotSchemaError("manifest.trajectories must be list")
    verdict = man.get("verdict")
    if verdict not in _MANIFEST_VERDICTS:
        raise PilotSchemaError("manifest.verdict invalid")
    if verdict == "write_ok":
        if not man["trajectories"]:
            raise PilotSchemaError("write_ok manifest requires non-empty trajectories")
        if man.get("dry_run") is not False:
            raise PilotSchemaError("write_ok manifest dry_run must be false")
    for i, row in enumerate(man["trajectories"]):
        if not isinstance(row, dict):
            raise PilotSchemaError(f"trajectories[{i}] must be object")
        _require_uuid(str(row.get("traj_id", "")))
        _require_str(row, "path")
        if type(row.get("gates_ok")) is not bool:
            raise PilotSchemaError(f"trajectories[{i}].gates_ok must be bool")
    if "rollback" in man:
        rb = man["rollback"]
        if not isinstance(rb, dict):
            raise PilotSchemaError("rollback must be object")
        if "traj_path" in rb and (not isinstance(rb["traj_path"], str) or not rb["traj_path"]):
            raise PilotSchemaError("rollback.traj_path must be non-empty str")
    if "training_forbidden" in man and man["training_forbidden"] is not True:
        raise PilotSchemaError("manifest.training_forbidden must be true when present")
    return man


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
