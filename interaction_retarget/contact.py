"""MuJoCo contact detection for bimanual assembly (self-contained)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interaction_retarget.constants import (
    LEFT_HAND_ROOT,
    PEG_BODY,
    RIGHT_HAND_ROOT,
    TRAY_BODY,
)

INSERT_GEOM = "industreal_tray_insert_round_peg_8mm_bottom_contact"
DEFAULT_LIFT_THRESHOLD_M = 0.05


@dataclass
class FrameContact:
    tray_contact: bool
    peg_contact: bool
    tray_contact_count: int
    peg_contact_count: int
    tray_contact_pos_world: np.ndarray  # (M, 3)
    peg_contact_pos_world: np.ndarray  # (M, 3)


class AssemblyContactDetector:
    """Hand–object contacts + contact positions (spider-style centers)."""

    def __init__(self, raw_env, *, lift_threshold_m: float = DEFAULT_LIFT_THRESHOLD_M) -> None:
        model = raw_env._model
        self._model = model
        self._lift_threshold_m = float(lift_threshold_m)
        self._peg_rest_z: float | None = None
        self._tray_rest_z: float | None = None

        self._peg_body_id = int(model.body(PEG_BODY).id)
        self._tray_body_id = int(model.body(TRAY_BODY).id)
        self._insert_geom_id = int(model.geom(INSERT_GEOM).id)
        self._peg_geom_ids = self._body_geom_ids(model, self._peg_body_id)
        self._tray_geom_ids = self._body_geom_ids(model, self._tray_body_id)

        left_root = int(model.body(LEFT_HAND_ROOT).id)
        right_root = int(model.body(RIGHT_HAND_ROOT).id)
        self._left_hand_geom_ids = self._body_geom_ids(model, left_root)
        self._right_hand_geom_ids = self._body_geom_ids(model, right_root)

    @staticmethod
    def _subtree_body_ids(model, root_body_id: int) -> set[int]:
        bodies = {int(root_body_id)}
        changed = True
        while changed:
            changed = False
            for bid in range(model.nbody):
                parent = int(model.body_parentid[bid])
                if parent in bodies and bid not in bodies:
                    bodies.add(bid)
                    changed = True
        return bodies

    @classmethod
    def _body_geom_ids(cls, model, body_id: int) -> set[int]:
        bodies = cls._subtree_body_ids(model, int(body_id))
        return {gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in bodies}

    @staticmethod
    def _pair_in(g1: int, g2: int, set_a: set[int], set_b: set[int]) -> bool:
        return (g1 in set_a and g2 in set_b) or (g2 in set_a and g1 in set_b)

    def reset_reference(self, raw_env) -> None:
        data = raw_env._data
        self._peg_rest_z = float(data.xpos[self._peg_body_id, 2])
        self._tray_rest_z = float(data.xpos[self._tray_body_id, 2])

    def compute(self, raw_env) -> FrameContact:
        data = raw_env._data
        tray_count = 0
        peg_count = 0
        tray_points: list[np.ndarray] = []
        peg_points: list[np.ndarray] = []

        for i in range(int(data.ncon)):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            pos = np.asarray(c.pos, dtype=np.float64).copy()

            if self._pair_in(g1, g2, self._tray_geom_ids, self._left_hand_geom_ids):
                tray_count += 1
                tray_points.append(pos)
            if self._pair_in(g1, g2, self._peg_geom_ids, self._right_hand_geom_ids):
                peg_count += 1
                peg_points.append(pos)

        return FrameContact(
            tray_contact=tray_count > 0,
            peg_contact=peg_count > 0,
            tray_contact_count=tray_count,
            peg_contact_count=peg_count,
            tray_contact_pos_world=np.stack(tray_points, axis=0) if tray_points else np.zeros((0, 3)),
            peg_contact_pos_world=np.stack(peg_points, axis=0) if peg_points else np.zeros((0, 3)),
        )

    def lifted(self, raw_env, *, object_name: str) -> bool:
        if self._peg_rest_z is None or self._tray_rest_z is None:
            raise RuntimeError("Call reset_reference() first.")
        data = raw_env._data
        if object_name == "tray":
            z = float(data.xpos[self._tray_body_id, 2])
            return (z - self._tray_rest_z) > self._lift_threshold_m
        if object_name == "peg":
            z = float(data.xpos[self._peg_body_id, 2])
            return (z - self._peg_rest_z) > self._lift_threshold_m
        raise ValueError(object_name)
