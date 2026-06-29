"""Hand–object contacts + Dexonomy GraspFilter + DexGraspBench QP (refs)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from interaction_retarget.constants import LEFT_HAND_ROOT, PEG_BODY, RIGHT_HAND_ROOT, TRAY_BODY
from interaction_retarget.tpsr.contact_physics import (
    _hand_root,
    _object_body,
    _subtree_geom_ids,
    hand_hand_contact_dists,
)
from interaction_retarget.tpsr.contact_qp import GraspQP, calcu_qp_metric
from interaction_retarget.DexGraspBench.src.util.rot_util import np_normalize_vector

ObjectName = Literal["tray", "peg"]
Side = Literal["left", "right"]

_ZERO_WRENCH = np.zeros(6, dtype=np.float64)


@dataclass(frozen=True)
class GraspFilterConfig:
    """Dexonomy grasp.yaml filter + optional DexGraspBench robust metric."""

    contact_dist_thre_m: float = 0.002
    ho_collision_thre_m: float = 0.0
    hh_collision_thre_m: float = 0.0
    qp_error_thre: float = 0.001
    miu_coef: tuple[float, float] = (0.6, 0.02)
    min_contacts: int = 3
    # DexGraspBench fc_metric/qp.py — averaged six-dir metric (not Dexonomy filter).
    robust_metric_check: bool = False
    robust_metric_thre: float = 0.001


def grasp_filter_cfg_from_tpsr(tpsr_cfg, *, ho_collision_thre_m: float = -0.006) -> GraspFilterConfig:
    """Build GraspFilterConfig from TpsrConfig (single entry point)."""
    return GraspFilterConfig(
        contact_dist_thre_m=tpsr_cfg.qp_contact_dist_thre_m,
        qp_error_thre=tpsr_cfg.qp_error_thre,
        miu_coef=tpsr_cfg.qp_miu_coef,
        min_contacts=tpsr_cfg.min_contact_count,
        robust_metric_check=bool(getattr(tpsr_cfg, "qp_robust_metric", False)),
        robust_metric_thre=float(getattr(tpsr_cfg, "qp_robust_metric_thre", 0.001)),
        ho_collision_thre_m=ho_collision_thre_m,
    )


def _geom_body_name(model, geom_id: int) -> str:
    bid = int(model.geom_bodyid[geom_id])
    return model.body(bid).name


def _subtree_body_ids(model, root_bid: int) -> set[int]:
    bodies = {int(root_bid)}
    changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            if int(model.body_parentid[b]) in bodies and b not in bodies:
                bodies.add(b)
                changed = True
    return bodies


def object_body_mass_kg(raw_env, object_name: ObjectName) -> float:
    model = raw_env._model
    bid = int(model.body(_object_body(object_name)).id)
    return float(sum(model.body_mass[b] for b in _subtree_body_ids(model, bid)))


def object_gravity_center_m(raw_env, object_name: ObjectName) -> np.ndarray:
    """Dexonomy gen_grasp ext_center = obj_gravity_center."""
    model = raw_env._model
    data = raw_env._data
    bid = int(model.body(_object_body(object_name)).id)
    bodies = _subtree_body_ids(model, bid)
    total_m = 0.0
    com = np.zeros(3, dtype=np.float64)
    for b in bodies:
        m = float(model.body_mass[b])
        if m <= 0.0:
            continue
        total_m += m
        com += m * np.asarray(data.xipos[b], dtype=np.float64)
    if total_m > 0.0:
        return com / total_m
    return np.asarray(data.xpos[bid], dtype=np.float64).copy()


def hand_object_contacts(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    contact_dist_thre_m: float = 0.002,
) -> dict[str, np.ndarray | list]:
    """Dexonomy MuJoCo_OptEnv.get_contacts hand–object branch."""
    model = raw_env._model
    data = raw_env._data
    obj_id = int(model.body(_object_body(object_name)).id)
    hand_id = int(model.body(_hand_root(side)).id)
    obj_geoms = _subtree_geom_ids(model, obj_id)
    hand_geoms = _subtree_geom_ids(model, hand_id)

    dist: list[float] = []
    pos: list[np.ndarray] = []
    normal: list[np.ndarray] = []
    bn1: list[str] = []
    bn2: list[str] = []

    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if not ((g1 in obj_geoms and g2 in hand_geoms) or (g2 in obj_geoms and g1 in hand_geoms)):
            continue
        frame = np.asarray(c.frame, dtype=np.float64).reshape(9)
        n01 = frame[0:3]
        if g1 in hand_geoms and g2 in obj_geoms:
            contact_normal = n01
            hand_name = _geom_body_name(model, g1)
            obj_name = _geom_body_name(model, g2)
        else:
            contact_normal = -n01
            hand_name = _geom_body_name(model, g2)
            obj_name = _geom_body_name(model, g1)
        d = float(c.dist)
        if d > contact_dist_thre_m:
            continue
        contact_normal = np_normalize_vector(contact_normal.reshape(1, 3)).reshape(3)
        cp = np.asarray(c.pos, dtype=np.float64) - d * contact_normal
        dist.append(d)
        pos.append(cp)
        normal.append(contact_normal)
        bn1.append(hand_name)
        bn2.append(obj_name)

    if not dist:
        empty = np.zeros((0, 3), dtype=np.float64)
        return {
            "dist": empty,
            "pos": empty,
            "normal": empty,
            "bn1": [],
            "bn2": [],
        }
    return {
        "dist": np.asarray(dist, dtype=np.float64),
        "pos": np.stack(pos, axis=0),
        "normal": np.stack(normal, axis=0),
        "bn1": bn1,
        "bn2": bn2,
    }


@dataclass
class GraspFilterResult:
    ok: bool
    ho: dict
    hh_min_dist_m: float
    max_qp_error: float
    wrench_errors: dict[str, float]
    robust_qp_metric: float | None = None


class GraspFilter:
    """Dexonomy GraspFilter._qp_filter: zero ext_wrench at object COM."""

    def __init__(self, cfg: GraspFilterConfig | None = None):
        self.cfg = cfg or GraspFilterConfig()
        self._qp = GraspQP(list(self.cfg.miu_coef))

    def forward(
        self,
        raw_env,
        *,
        side: Side,
        object_name: ObjectName,
    ) -> GraspFilterResult:
        cfg = self.cfg
        ho = hand_object_contacts(
            raw_env,
            side=side,
            object_name=object_name,
            contact_dist_thre_m=cfg.contact_dist_thre_m,
        )
        hh_dists = hand_hand_contact_dists(raw_env, side=side)
        hh_min = float(np.min(hh_dists)) if hh_dists.size else float("inf")

        wrench_errors: dict[str, float] = {}
        max_qp = 1.0
        robust_metric: float | None = None

        if ho["pos"].shape[0] < cfg.min_contacts:
            return GraspFilterResult(False, ho, hh_min, max_qp, wrench_errors, robust_metric)
        if hh_dists.size and hh_min < cfg.hh_collision_thre_m:
            return GraspFilterResult(False, ho, hh_min, max_qp, wrench_errors, robust_metric)
        if ho["dist"].size and float(np.min(ho["dist"])) < cfg.ho_collision_thre_m:
            return GraspFilterResult(False, ho, hh_min, max_qp, wrench_errors, robust_metric)

        pos = ho["pos"]
        normal = ho["normal"]
        ext_center = object_gravity_center_m(raw_env, object_name)

        # Dexonomy gen_grasp: ext_wrench = zeros(6) at filter stage.
        _, err = self._qp.solve(pos, normal, _ZERO_WRENCH, ext_center)
        err = float(err)
        wrench_errors["filter"] = err
        max_qp = err
        if err > cfg.qp_error_thre:
            return GraspFilterResult(False, ho, hh_min, max_qp, wrench_errors, robust_metric)

        if cfg.robust_metric_check:
            robust_metric = float(calcu_qp_metric(pos, normal, list(cfg.miu_coef)))
            if robust_metric > cfg.robust_metric_thre:
                return GraspFilterResult(
                    False, ho, hh_min, max_qp, wrench_errors, robust_metric
                )

        return GraspFilterResult(True, ho, hh_min, max_qp, wrench_errors, robust_metric)


def qp_wrench_error(
    raw_env,
    *,
    side: Side,
    object_name: ObjectName,
    cfg: GraspFilterConfig | None = None,
) -> float:
    """Dexonomy filter QP error (zero wrench at object COM)."""
    res = GraspFilter(cfg).forward(raw_env, side=side, object_name=object_name)
    return res.max_qp_error
