"""MuJoCo ground-truth adapter for P0 assembly primitive experiments."""

from __future__ import annotations

import numpy as np

from dexjoco.sim.envs.assembly_geometry import names_from_raw
from hybrid_insert.geometry import body_z_axis, hole_opening_axis, peg_insert_end_pos

from .primitives import AssemblyPrimitiveSet, PriorSource


class PrivilegedAssemblyPrimitiveProvider:
    """Instantiate assembly primitives from MuJoCo state without visual error."""

    def __init__(self, raw_env) -> None:
        self.names = names_from_raw(raw_env)
        model = raw_env._model
        self._peg_body_id = int(model.body(self.names.peg_body).id)
        self._socket_site_id = int(model.site(self.names.socket_site).id)
        self._socket_bottom_geom_id = int(model.geom(self.names.socket_bottom).id)

    def snapshot(self, raw_env) -> AssemblyPrimitiveSet:
        data = raw_env._data
        peg_body_pos = np.asarray(data.xpos[self._peg_body_id], dtype=np.float64)
        peg_body_xmat = np.asarray(data.xmat[self._peg_body_id], dtype=np.float64)
        hole_entry = np.asarray(data.site_xpos[self._socket_site_id], dtype=np.float64)
        hole_xmat = np.asarray(data.site_xmat[self._socket_site_id], dtype=np.float64)
        hole_bottom = np.asarray(
            data.geom_xpos[self._socket_bottom_geom_id], dtype=np.float64
        )

        return AssemblyPrimitiveSet(
            family_id=self.names.family_id,
            section=self.names.section,
            peg_tip_world=peg_insert_end_pos(peg_body_pos, peg_body_xmat),
            peg_axis_world=body_z_axis(peg_body_xmat),
            hole_entry_world=hole_entry,
            hole_axis_world=hole_opening_axis(hole_entry, hole_xmat, hole_bottom),
            hole_bottom_world=hole_bottom,
            nominal_peg_size_m=self.names.size_mm / 1000.0,
            source=PriorSource.PRIVILEGED,
        )
