"""MuJoCo contact-based outcome labels for bimanual assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexjoco.sim.envs.assembly_geometry import names_for_family, names_from_raw

DEFAULT_LIFT_THRESHOLD_M = 0.05


@dataclass
class AssemblyOutcome:
    tray_ok: bool
    peg_ok: bool
    insert_ok: bool
    tray_contact_count: int
    peg_contact_count: int


class AssemblyContactLabeler:
    """Detect tray / peg grasp outcomes from sim contacts and lift height."""

    _LEFT_HAND_ROOT = "allegro_palm_left"
    _RIGHT_HAND_ROOT = "allegro_palm_right"

    def __init__(
        self,
        raw_env,
        *,
        lift_threshold_m: float = DEFAULT_LIFT_THRESHOLD_M,
        geometry_family: str | None = None,
    ) -> None:
        if lift_threshold_m <= 0:
            raise ValueError(f"lift_threshold_m must be positive, got {lift_threshold_m}")
        self._lift_threshold_m = float(lift_threshold_m)
        self._peg_rest_z: float | None = None
        self._tray_rest_z: float | None = None

        if geometry_family is not None:
            names = names_for_family(geometry_family)
        else:
            names = names_from_raw(raw_env)
        self.geometry_family = names.family_id
        self._names = names

        model = raw_env._model
        self._peg_body_id = int(model.body(names.peg_body).id)
        self._tray_body_id = int(model.body(names.socket_body).id)
        self._insert_geom_id = int(model.geom(names.socket_bottom).id)
        self._peg_geom_ids = self._collect_body_geom_ids(model, self._peg_body_id)
        self._tray_geom_ids = self._collect_body_geom_ids(model, self._tray_body_id)

        left_root = int(model.body(self._LEFT_HAND_ROOT).id)
        right_root = int(model.body(self._RIGHT_HAND_ROOT).id)
        left_bodies = self._collect_subtree_body_ids(model, left_root)
        right_bodies = self._collect_subtree_body_ids(model, right_root)
        self._left_hand_geom_ids = self._collect_bodies_geom_ids(model, left_bodies)
        self._right_hand_geom_ids = self._collect_bodies_geom_ids(model, right_bodies)

    @staticmethod
    def _collect_body_geom_ids(model, body_id: int) -> set[int]:
        subtree = AssemblyContactLabeler._collect_subtree_body_ids(model, body_id)
        return AssemblyContactLabeler._collect_bodies_geom_ids(model, subtree)

    @staticmethod
    def _collect_subtree_body_ids(model, root_body_id: int) -> set[int]:
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

    @staticmethod
    def _collect_bodies_geom_ids(model, body_ids: set[int]) -> set[int]:
        geom_ids: set[int] = set()
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) in body_ids:
                geom_ids.add(gid)
        return geom_ids

    @staticmethod
    def _geom_pair_in_sets(g1: int, g2: int, set_a: set[int], set_b: set[int]) -> bool:
        return (g1 in set_a and g2 in set_b) or (g2 in set_a and g1 in set_b)

    def reset_reference(self, raw_env) -> None:
        """Capture resting object heights after reset / initial-state restore."""
        data = raw_env._data
        self._peg_rest_z = float(data.xpos[self._peg_body_id, 2])
        self._tray_rest_z = float(data.xpos[self._tray_body_id, 2])

    def _lifted(self, raw_env, body_id: int, rest_z: float | None) -> bool:
        if rest_z is None:
            raise RuntimeError("Call reset_reference() before compute().")
        current_z = float(raw_env._data.xpos[body_id, 2])
        return (current_z - rest_z) > self._lift_threshold_m

    def compute(self, raw_env) -> AssemblyOutcome:
        data = raw_env._data
        tray_count = 0
        peg_count = 0
        insert_count = 0

        for i in range(int(data.ncon)):
            contact = data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)

            if self._geom_pair_in_sets(g1, g2, self._tray_geom_ids, self._left_hand_geom_ids):
                tray_count += 1
            if self._geom_pair_in_sets(g1, g2, self._peg_geom_ids, self._right_hand_geom_ids):
                peg_count += 1
            if (g1 == self._insert_geom_id and g2 in self._peg_geom_ids) or (
                g2 == self._insert_geom_id and g1 in self._peg_geom_ids
            ):
                insert_count += 1

        tray_contact = tray_count > 0
        peg_contact = peg_count > 0
        return AssemblyOutcome(
            tray_ok=tray_contact and self._lifted(raw_env, self._tray_body_id, self._tray_rest_z),
            peg_ok=peg_contact and self._lifted(raw_env, self._peg_body_id, self._peg_rest_z),
            insert_ok=insert_count > 0,
            tray_contact_count=tray_count,
            peg_contact_count=peg_count,
        )
