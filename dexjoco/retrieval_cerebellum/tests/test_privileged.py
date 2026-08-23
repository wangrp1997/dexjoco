from types import SimpleNamespace

import numpy as np
import pytest

from dexjoco.sim.envs.assembly_geometry import names_for_family
from retrieval_cerebellum.primitives import PriorSource
from retrieval_cerebellum.privileged import PrivilegedAssemblyPrimitiveProvider


class _NamedModel:
    def __init__(self, names) -> None:
        self._body_ids = {names.peg_body: 0}
        self._site_ids = {names.socket_site: 0}
        self._geom_ids = {names.socket_bottom: 0}

    def body(self, name):
        return SimpleNamespace(id=self._body_ids[name])

    def site(self, name):
        return SimpleNamespace(id=self._site_ids[name])

    def geom(self, name):
        return SimpleNamespace(id=self._geom_ids[name])


def _raw_env():
    names = names_for_family("round_8mm")
    data = SimpleNamespace(
        xpos=np.array([[0.01, 0.0, 1.05]], dtype=np.float64),
        xmat=np.array([np.eye(3)], dtype=np.float64),
        site_xpos=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        site_xmat=np.array([np.eye(3)], dtype=np.float64),
        geom_xpos=np.array([[0.0, 0.0, 0.95]], dtype=np.float64),
    )
    return SimpleNamespace(_geom_names=names, _model=_NamedModel(names), _data=data)


def test_privileged_provider_instantiates_scene_geometry():
    raw = _raw_env()
    provider = PrivilegedAssemblyPrimitiveProvider(raw)

    primitives = provider.snapshot(raw)

    assert primitives.family_id == "round_8mm"
    assert primitives.source is PriorSource.PRIVILEGED
    assert primitives.nominal_peg_size_m == pytest.approx(0.008)
    assert primitives.target_depth_m == pytest.approx(0.05)
    assert primitives.lateral_error_m == pytest.approx(0.01)
    assert primitives.approach_height_m == pytest.approx(0.0635)
    assert primitives.axis_error_rad == pytest.approx(0.0)
    np.testing.assert_allclose(primitives.hole_axis_world, [0.0, 0.0, 1.0])


def test_privileged_provider_orients_hole_axis_outward():
    raw = _raw_env()
    raw._data.site_xmat[0] = np.diag([1.0, -1.0, -1.0])

    primitives = PrivilegedAssemblyPrimitiveProvider(raw).snapshot(raw)

    np.testing.assert_allclose(primitives.hole_axis_world, [0.0, 0.0, 1.0])
