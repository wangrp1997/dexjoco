"""Encoder-only Allegro fingertip forward kinematics."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


RIGHT_JOINT_NAMES = tuple(
    f"{finger}j{joint}_right"
    for finger in ("ff", "mf", "rf", "th")
    for joint in range(4)
)
LEFT_STATE_JOINT_NAMES = tuple(
    f"{finger}j{joint}_left"
    for finger in ("rf", "mf", "ff", "th")
    for joint in range(4)
)
FINGERTIP_BODY_NAMES = (
    ("ff_tip_right", "mf_tip_right", "rf_tip_right", "th_tip_right"),
    ("ff_tip_left", "mf_tip_left", "rf_tip_left", "th_tip_left"),
)
PALM_BODY_NAMES = ("allegro_palm_right", "allegro_palm_left")


def default_assembly_xml() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "dexjoco"
        / "sim"
        / "envs"
        / "xmls"
        / "arena_arm_hand_bimanual_assembly.xml"
    )


class AllegroFingertipKinematics:
    """Compute eight fingertip positions in their corresponding palm frames."""

    def __init__(self, xml_path: Path | None = None) -> None:
        path = Path(xml_path or default_assembly_xml()).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Allegro kinematics MJCF not found: {path}")
        self.xml_path = path
        self._model = mujoco.MjModel.from_xml_path(str(path))
        self._data = mujoco.MjData(self._model)
        self._right_qpos = np.asarray(
            [int(self._model.joint(name).qposadr[0]) for name in RIGHT_JOINT_NAMES],
            dtype=np.int64,
        )
        self._left_qpos = np.asarray(
            [
                int(self._model.joint(name).qposadr[0])
                for name in LEFT_STATE_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self._palm_ids = tuple(
            int(self._model.body(name).id) for name in PALM_BODY_NAMES
        )
        self._tip_ids = tuple(
            tuple(int(self._model.body(name).id) for name in names)
            for names in FINGERTIP_BODY_NAMES
        )
        mujoco.mj_forward(self._model, self._data)

    def positions_in_palm(self, state46: np.ndarray) -> np.ndarray:
        """Return shape ``(2 arms, 4 fingers, 3 xyz)`` in palm coordinates."""
        state = np.asarray(state46, dtype=np.float64).reshape(-1)
        if state.shape != (46,):
            raise ValueError(f"state46 must have shape (46,), got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("state46 contains non-finite values")
        self._data.qpos[self._right_qpos] = state[14:30]
        self._data.qpos[self._left_qpos] = state[30:46]
        mujoco.mj_forward(self._model, self._data)

        result = np.empty((2, 4, 3), dtype=np.float32)
        for arm, (palm_id, tip_ids) in enumerate(
            zip(self._palm_ids, self._tip_ids, strict=True)
        ):
            palm_position = np.asarray(self._data.xpos[palm_id], dtype=np.float64)
            palm_rotation = np.asarray(
                self._data.xmat[palm_id], dtype=np.float64
            ).reshape(3, 3)
            for finger, tip_id in enumerate(tip_ids):
                tip_world = np.asarray(self._data.xpos[tip_id], dtype=np.float64)
                result[arm, finger] = palm_rotation.T @ (
                    tip_world - palm_position
                )
        if not np.isfinite(result).all():
            raise RuntimeError("fingertip kinematics produced non-finite values")
        return result
