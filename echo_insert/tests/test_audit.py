from pathlib import Path

import numpy as np

from echo_insert.audit import audit_action_path, audit_python_files, audit_sim_boundary
from echo_insert.sim_wrist import read_wrist_wrench_local


def test_echo_action_and_sensor_boundaries_are_clean() -> None:
    assert audit_action_path() == []
    assert audit_sim_boundary() == []


def test_audit_rejects_hidden_simulator_inputs(tmp_path: Path) -> None:
    source = tmp_path / "bad.py"
    source.write_text("import mujoco\nvalue = raw._data.site_xpos\n", encoding="utf-8")

    findings = audit_python_files([source])

    assert any("import:mujoco" in finding for finding in findings)
    assert any("identifier:site_xpos" in finding for finding in findings)


def test_action_path_audits_kinematic_estimator(tmp_path: Path) -> None:
    for name in ("public_io.py", "optimizer.py", "controller.py"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "kinematic_estimator.py").write_text(
        "value = raw._data.site_xpos\n",
        encoding="utf-8",
    )

    assert (
        "kinematic_estimator.py:1:identifier:site_xpos"
        in audit_action_path(tmp_path)
    )


def test_sim_boundary_audits_depth_adapter(tmp_path: Path) -> None:
    (tmp_path / "sim_wrist.py").write_text("", encoding="utf-8")
    (tmp_path / "sim_depth.py").write_text(
        "value = raw._data.site_xpos\n",
        encoding="utf-8",
    )

    assert (
        "sim_depth.py:1:identifier:site_xpos"
        in audit_sim_boundary(tmp_path)
    )


def test_named_wrist_sensor_boundary_preserves_side_and_wrench_order() -> None:
    values = {
        "panda/wrist_force_right": [1.0, 2.0, 3.0],
        "panda/wrist_torque_right": [4.0, 5.0, 6.0],
        "panda/wrist_force_left": [7.0, 8.0, 9.0],
        "panda/wrist_torque_left": [10.0, 11.0, 12.0],
    }

    class Data:
        def sensor(self, name: str):
            return type("Sensor", (), {"data": np.asarray(values[name])})()

    raw = type("Raw", (), {"_data": Data()})()

    np.testing.assert_allclose(
        read_wrist_wrench_local(raw),
        np.arange(1.0, 13.0).reshape(2, 6),
    )
