"""Static provenance guard for the ECHO action path."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ACTION_FILES = (
    "public_io.py",
    "optimizer.py",
    "controller.py",
    "kinematic_estimator.py",
)
SIM_SENSOR_FILES = ("sim_wrist.py", "sim_depth.py")
BANNED_IMPORT_ROOTS = {
    "dexquery",
    "hybrid_insert",
    "interaction_retarget",
    "mujoco",
    "priv_snap_insert",
    "privileged_insert_servo",
}
BANNED_IDENTIFIERS = {
    "cfrc_ext",
    "contact_truth",
    "demo",
    "full_state",
    "geom_xpos",
    "info",
    "insert_depth",
    "keypose",
    "mocap_pos",
    "mocap_quat",
    "ncon",
    "privileged_state",
    "qpos",
    "qvel",
    "reward",
    "site_xmat",
    "site_xpos",
    "succeed",
    "teacher",
    "template",
    "xmat",
    "xpos",
}
SIM_BOUNDARY_BANNED = BANNED_IDENTIFIERS - {"demo", "info", "reward", "succeed"}
WRIST_SENSOR_NAMES = {
    "panda/wrist_force_right",
    "panda/wrist_torque_right",
    "panda/wrist_force_left",
    "panda/wrist_torque_left",
}


def audit_python_files(
    paths: list[Path],
    *,
    banned_identifiers: set[str] = BANNED_IDENTIFIERS,
) -> list[str]:
    findings: list[str] = []
    for path in paths:
        if not path.is_file():
            findings.append(f"missing:{path.name}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
            else:
                imports = []
            for module in imports:
                if module.split(".", 1)[0] in BANNED_IMPORT_ROOTS:
                    findings.append(f"{path.name}:{node.lineno}:import:{module}")
            identifier = None
            if isinstance(node, ast.Name):
                identifier = node.id
            elif isinstance(node, ast.Attribute):
                identifier = node.attr
            if identifier in banned_identifiers:
                findings.append(f"{path.name}:{node.lineno}:identifier:{identifier}")
    return sorted(set(findings))


def audit_action_path(root: Path | None = None) -> list[str]:
    base = Path(root or Path(__file__).parent)
    return audit_python_files([base / name for name in ACTION_FILES])


def audit_sim_boundary(root: Path | None = None) -> list[str]:
    base = Path(root or Path(__file__).parent)
    paths = [base / name for name in SIM_SENSOR_FILES]
    findings = audit_python_files(paths, banned_identifiers=SIM_BOUNDARY_BANNED)
    path = paths[0]
    if not path.is_file():
        return findings
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    named_sensors = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("panda/")
    }
    unexpected = named_sensors - WRIST_SENSOR_NAMES
    missing = WRIST_SENSOR_NAMES - named_sensors
    findings.extend(f"sim_wrist.py:unexpected_sensor:{name}" for name in unexpected)
    findings.extend(f"sim_wrist.py:missing_sensor:{name}" for name in missing)
    return sorted(set(findings))


def main() -> int:
    report = {
        "action_path": audit_action_path(),
        "sim_wrist_boundary": audit_sim_boundary(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(any(report.values()))


if __name__ == "__main__":
    raise SystemExit(main())
