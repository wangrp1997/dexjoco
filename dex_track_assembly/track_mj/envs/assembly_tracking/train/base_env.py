"""Base MJX env for Panda bimanual assembly."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import mujoco
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env


def patch_model_for_mjx(model: mujoco.MjModel) -> None:
    """MJX does not support cylinder-box collisions; use capsules instead."""
    for i in range(model.ngeom):
        if (
            model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER
            and model.geom_contype[i] > 0
        ):
            model.geom_type[i] = mujoco.mjtGeom.mjGEOM_CAPSULE


class AssemblyEnv(mjx_env.MjxEnv):
    """Base class for assembly tracking environments."""

    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        patch_model_for_mjx(self._mj_model)
        self._mj_model.opt.timestep = self.sim_dt

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = xml_path

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
