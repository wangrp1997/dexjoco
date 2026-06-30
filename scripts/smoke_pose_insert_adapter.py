#!/usr/bin/env python3
"""Smoke test for pose_insert adapter math (no MuJoCo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEXJOJO_ROOT = _REPO_ROOT / "dexjoco"
if str(_DEXJOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEXJOJO_ROOT))

from pose_insert.adapter import (
    calibrate_peg_to_wrist,
    denormalize_translation,
    matrix4_to_pose9,
    pose9_to_matrix4,
    relative_pose9_to_world_source,
    source_pose7_to_wrist_pose7,
)
from pose_insert.dataset_sim import normalize_translation, pose7_sequence_to_pose9
from pose_insert.poses import source_in_target_poses


def main() -> int:
    source = np.array([0.5, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    target = np.array([0.52, 0.08, 0.25, 0.0, 0.707, 0.0, 0.707], dtype=np.float64)
    rel7 = source_in_target_poses(source.reshape(1, 7), target.reshape(1, 7))[0]

    workspace = np.array([[0.1, 0.1, 0.1, -0.1, -0.1, -0.1] + [0.0] * 6], dtype=np.float64)
    rel_norm = normalize_translation(workspace, rel7.reshape(1, 7))[0]
    pose9 = pose7_sequence_to_pose9(rel_norm.reshape(1, 7))[0]

    mat4 = pose9_to_matrix4(pose9)
    round_pose9 = matrix4_to_pose9(mat4)
    if not np.allclose(round_pose9[:, :2], pose9[:, :2], atol=1e-6):
        print("pose9 roundtrip failed on rotation columns", file=sys.stderr)
        return 1

    denorm = denormalize_translation(workspace, mat4)
    _ = denorm
    recovered = relative_pose9_to_world_source(pose9, target, workspace=workspace)
    if not np.allclose(recovered[:3], source[:3], atol=1e-5):
        print(f"world source mismatch: {recovered[:3]} vs {source[:3]}", file=sys.stderr)
        return 1

    wrist_pose7 = np.concatenate(
        [np.array([0.6, 0.2, 0.35], dtype=np.float64), R.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat()]
    )
    peg_to_wrist = calibrate_peg_to_wrist(source, wrist_pose7)
    back = source_pose7_to_wrist_pose7(source, peg_to_wrist)
    if not np.allclose(back, wrist_pose7, atol=1e-9):
        print("peg->wrist calibration failed", file=sys.stderr)
        return 1

    print("pose_insert adapter smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
