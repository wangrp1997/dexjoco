"""Privileged grasp metrics from MuJoCo truth (not 85D-only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from interaction_retarget.constants import PEG_BODY
from dexjoco.sim.envs.assembly_geometry import names_from_raw

REFERENCE_BODY = "allegro_palm_right"


def _peg_body_name(raw_or_model=None) -> str:
    """Prefer env geometry_family names; fall back to legacy round_8mm constant."""
    if raw_or_model is None:
        return PEG_BODY
    if hasattr(raw_or_model, "_model"):
        return names_from_raw(raw_or_model).peg_body
    return PEG_BODY

# Body-name prefixes → contact class. Little finger does not exist on this Allegro.
FINGER_CLASS_PREFIXES: dict[str, tuple[str, ...]] = {
    "palm": ("allegro_palm_right",),
    "index": ("ff_",),
    "middle": ("mf_",),
    "ring": ("rf_",),
    "thumb": ("th_",),
}


@dataclass
class ObjectInHandPose:
    reference_body: str
    translation: np.ndarray  # (3,) peg origin in reference frame
    rotvec: np.ndarray  # (3,) peg rotation relative to reference (continuous)


@dataclass
class PegHandContact:
    total: int
    by_class: dict[str, int]
    unknown_geom_names: list[str]
    unknown_count: int


@dataclass
class GraspStepMetrics:
    object_in_hand: ObjectInHandPose
    peg_hand_contact: PegHandContact
    right_finger_force: np.ndarray  # (12,)
    right_finger_force_norm: np.ndarray  # (4,)
    contact_active: np.ndarray  # (4,) bool tip-force proxy
    tray_ok: bool
    peg_ok: bool
    insert_ok: bool
    peg_world_pos: np.ndarray
    peg_world_quat_wxyz: np.ndarray
    # Slip proxies (NOT ground truth).
    slip_proxy_tangential_rel_vel: float
    slip_proxy_pose_drift_rate: float | None


def _body_xmat(data, body_id: int) -> np.ndarray:
    return np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)


def _body_xpos(data, body_id: int) -> np.ndarray:
    return np.asarray(data.xpos[body_id], dtype=np.float64).copy()


def _body_xquat_wxyz(data, body_id: int) -> np.ndarray:
    # MuJoCo xquat is wxyz.
    return np.asarray(data.xquat[body_id], dtype=np.float64).copy()


def object_in_hand_pose(raw, *, reference_body: str = REFERENCE_BODY) -> ObjectInHandPose:
    model, data = raw._model, raw._data
    peg_id = int(model.body(names_from_raw(raw).peg_body).id)
    ref_id = int(model.body(reference_body).id)
    peg_pos = _body_xpos(data, peg_id)
    ref_pos = _body_xpos(data, ref_id)
    peg_rot = R.from_matrix(_body_xmat(data, peg_id))
    ref_rot = R.from_matrix(_body_xmat(data, ref_id))
    rel_rot = ref_rot.inv() * peg_rot
    rel_t = ref_rot.inv().apply(peg_pos - ref_pos)
    return ObjectInHandPose(
        reference_body=reference_body,
        translation=np.asarray(rel_t, dtype=np.float64),
        rotvec=np.asarray(rel_rot.as_rotvec(), dtype=np.float64),
    )


def relative_pose_error(a: ObjectInHandPose, b: ObjectInHandPose) -> tuple[float, float]:
    """Return (translation_l2, rotation_angle_rad) between two o2h poses."""
    dt = float(np.linalg.norm(a.translation - b.translation))
    ra = R.from_rotvec(a.rotvec)
    rb = R.from_rotvec(b.rotvec)
    dR = ra.inv() * rb
    dang = float(np.linalg.norm(dR.as_rotvec()))
    return dt, dang


def _geom_name(model, geom_id: int) -> str:
    try:
        return str(model.geom(geom_id).name)
    except Exception:
        return f"geom_id_{geom_id}"


def _body_name(model, body_id: int) -> str:
    try:
        return str(model.body(body_id).name)
    except Exception:
        return f"body_id_{body_id}"


def _classify_right_hand_body(body_name: str) -> str | None:
    if body_name == "allegro_palm_right" or body_name.startswith("allegro_palm_right"):
        return "palm"
    for cls, prefixes in FINGER_CLASS_PREFIXES.items():
        if cls == "palm":
            continue
        for p in prefixes:
            if body_name.startswith(p) and body_name.endswith("_right"):
                return cls
            if body_name.startswith(p) and "_right" in body_name:
                return cls
    if body_name.endswith("_right") and any(
        k in body_name for k in ("ff_", "mf_", "rf_", "th_", "palm")
    ):
        # Known family but unmapped spelling.
        return None
    return None


def peg_hand_contact_counts(raw) -> PegHandContact:
    model, data = raw._model, raw._data
    peg_id = int(model.body(names_from_raw(raw).peg_body).id)
    peg_geoms = {
        int(g) for g in range(model.ngeom) if int(model.geom_bodyid[g]) == peg_id
    }
    # Include peg subtree geoms.
    for bid in range(model.nbody):
        if int(model.body_parentid[bid]) == peg_id or bid == peg_id:
            for g in range(model.ngeom):
                if int(model.geom_bodyid[g]) == bid:
                    peg_geoms.add(int(g))

    counts = {k: 0 for k in ("palm", "index", "middle", "ring", "thumb")}
    unknown_names: list[str] = []
    unknown = 0
    total = 0

    for i in range(int(data.ncon)):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 in peg_geoms:
            hand_g, peg_g = g2, g1
        elif g2 in peg_geoms:
            hand_g, peg_g = g1, g2
        else:
            continue
        body_id = int(model.geom_bodyid[hand_g])
        bname = _body_name(model, body_id)
        cls = _classify_right_hand_body(bname)
        if cls is None:
            # Only count as peg-hand if body looks like right hand subtree.
            if "right" not in bname:
                continue
            unknown += 1
            unknown_names.append(f"{bname}/{_geom_name(model, hand_g)}")
            total += 1
            continue
        counts[cls] += 1
        total += 1

    return PegHandContact(
        total=int(total),
        by_class={k: int(v) for k, v in counts.items()},
        unknown_geom_names=sorted(set(unknown_names)),
        unknown_count=int(unknown),
    )


def compute_step_metrics(
    env,
    *,
    root_o2h: ObjectInHandPose | None = None,
    prev_o2h: ObjectInHandPose | None = None,
    dt: float = 1.0,
    contact_force_eps: float = 0.05,
) -> GraspStepMetrics:
    raw = env._raw
    o2h = object_in_hand_pose(raw)
    contact = peg_hand_contact_counts(raw)
    outcome = env._labeler.compute(raw)

    right_force = np.zeros(12, dtype=np.float64)
    if env._force_labeler is not None:
        frame = env._force_labeler.compute(raw)
        right_force = np.asarray(frame.right_finger_force, dtype=np.float64).copy()
    force_norm = np.linalg.norm(right_force.reshape(4, 3), axis=1)
    active = force_norm >= float(contact_force_eps)

    # Tangential relative velocity proxy in palm frame.
    peg_id = int(raw._model.body(names_from_raw(raw).peg_body).id)
    palm_id = int(raw._model.body(REFERENCE_BODY).id)
    # cvel is 6D spatial velocity in body frame / world mixed; use xpos finite diff preference via prev.
    slip_tangential = 0.0
    if prev_o2h is not None and dt > 0:
        dpos = (o2h.translation - prev_o2h.translation) / dt
        # Tangential = components perpendicular to palm approach approx using full rel vel magnitude
        # minus radial; here use full translation rate as conservative proxy.
        slip_tangential = float(np.linalg.norm(dpos))

    drift_rate = None
    if root_o2h is not None and prev_o2h is not None and dt > 0:
        t0, r0 = relative_pose_error(root_o2h, prev_o2h)
        t1, r1 = relative_pose_error(root_o2h, o2h)
        drift_rate = float(np.hypot(t1 - t0, r1 - r0) / dt)

    peg_id = int(raw._model.body(names_from_raw(raw).peg_body).id)
    return GraspStepMetrics(
        object_in_hand=o2h,
        peg_hand_contact=contact,
        right_finger_force=right_force,
        right_finger_force_norm=force_norm,
        contact_active=active.astype(bool),
        tray_ok=bool(outcome.tray_ok),
        peg_ok=bool(outcome.peg_ok),
        insert_ok=bool(outcome.insert_ok),
        peg_world_pos=_body_xpos(raw._data, peg_id),
        peg_world_quat_wxyz=_body_xquat_wxyz(raw._data, peg_id),
        slip_proxy_tangential_rel_vel=slip_tangential,
        slip_proxy_pose_drift_rate=drift_rate,
    )


def control_dt_seconds(env) -> float:
    """Real seconds per FullEpisodeEnv.step from MuJoCo time advance."""
    raw = env._raw
    t0 = float(raw._data.time)
    # Do not step here; prefer model.opt.timestep * observed substeps if known.
    # Default assembly: 0.002 * 10 = 0.02 (measured once in P0-C1).
    tip = float(raw._model.opt.timestep)
    # Heuristic: many DexJoCo envs use 10 substeps; verify via attribute if present.
    n = getattr(raw, "frame_skip", None) or getattr(env._env, "frame_skip", None)
    if n is None:
        n = 10
    return float(tip) * float(n)


def summarize_rollout_metrics(
    steps: list[GraspStepMetrics],
    *,
    root_o2h: ObjectInHandPose,
) -> dict[str, Any]:
    """P0-C0-compatible summary (retention vs first rollout step). Prefer v2 for P0-C1."""
    return summarize_rollout_metrics_v2(
        steps,
        root_o2h=root_o2h,
        root_contact=None,
        control_dt_s=1.0,
        legacy_peg_loss_alias=True,
    )


def summarize_rollout_metrics_v2(
    steps: list[GraspStepMetrics],
    *,
    root_o2h: ObjectInHandPose,
    root_contact: PegHandContact | None,
    control_dt_s: float,
    root_peg_world_z: float | None = None,
    legacy_peg_loss_alias: bool = False,
    drop_z_thresh_m: float = 0.03,
    drop_trans_thresh_m: float = 0.05,
) -> dict[str, Any]:
    if not steps:
        return {
            "num_steps": 0,
            "terminal_peg_ok": False,
            "error": "empty_rollout",
        }

    if root_contact is None:
        root_total = float(steps[0].peg_hand_contact.total)
        root_by = dict(steps[0].peg_hand_contact.by_class)
    else:
        root_total = float(root_contact.total)
        root_by = dict(root_contact.by_class)

    root_z = (
        float(root_peg_world_z)
        if root_peg_world_z is not None
        else float(steps[0].peg_world_pos[2])
    )

    trans_drift = []
    rot_drift = []
    contact_totals = []
    active_ratio = []
    for s in steps:
        dt, dr = relative_pose_error(root_o2h, s.object_in_hand)
        trans_drift.append(dt)
        rot_drift.append(dr)
        contact_totals.append(s.peg_hand_contact.total)
        active_ratio.append(float(np.mean(s.contact_active)))

    trans_drift = np.asarray(trans_drift, dtype=np.float64)
    rot_drift = np.asarray(rot_drift, dtype=np.float64)
    contact_totals = np.asarray(contact_totals, dtype=np.float64)

    retention = []
    loss_steps = 0
    class_ret = {k: [] for k in root_by}
    for s in steps:
        c = float(s.peg_hand_contact.total)
        if root_total <= 0:
            retention.append(1.0 if c > 0 else 0.0)
        else:
            retention.append(float(min(c / root_total, 1.0)))
        if c <= 0:
            loss_steps += 1
        for k in root_by:
            r0 = float(root_by[k])
            r1 = float(s.peg_hand_contact.by_class.get(k, 0))
            class_ret[k].append(1.0 if r0 <= 0 and r1 >= 0 else float(min(r1 / max(r0, 1.0), 1.0)))

    peg_ok_series = [bool(s.peg_ok) for s in steps]
    terminal_peg_ok = bool(peg_ok_series[-1])
    peg_contact_present_end = int(contact_totals[-1]) > 0
    peg_contact_absent_steps = int(loss_steps)
    peg_z_end = float(steps[-1].peg_world_pos[2])
    peg_z_drop = float(root_z - peg_z_end)
    # AssemblyContactLabeler lift threshold default 0.05m — below_lift is privilege proxy.
    below_lift_threshold_end = not bool(steps[-1].peg_ok)  # labeler already encodes lift+contact

    o2h_trans_end = float(trans_drift[-1])
    object_dropped_proxy = bool(
        (not peg_contact_present_end)
        and (peg_z_drop >= float(drop_z_thresh_m))
        and (o2h_trans_end >= float(drop_trans_thresh_m))
    )

    dt = float(control_dt_s) if control_dt_s > 0 else 1.0
    # Re-scale slip proxies if they were computed with dt=1 wrongly: recompute from o2h series.
    slip_tangential = []
    slip_rot = []
    prev = root_o2h
    for s in steps:
        dpos = (s.object_in_hand.translation - prev.translation) / dt
        _, dang = relative_pose_error(prev, s.object_in_hand)
        slip_tangential.append(float(np.linalg.norm(dpos)))
        slip_rot.append(float(dang / dt))
        prev = s.object_in_hand
    slip_tangential = np.asarray(slip_tangential, dtype=np.float64)
    slip_rot = np.asarray(slip_rot, dtype=np.float64)

    out = {
        "num_steps": len(steps),
        "reference_body": root_o2h.reference_body,
        "control_dt_s": dt,
        "root_contact_total": int(root_total),
        "root_contact_by_class": {k: int(v) for k, v in root_by.items()},
        "contact_retention_vs_root_mean": float(np.mean(retention)),
        "contact_retention_vs_root_end": float(retention[-1]),
        "contact_loss_steps": int(loss_steps),
        "contact_class_retention": {
            k: float(np.mean(v)) if v else 0.0 for k, v in class_ret.items()
        },
        # Legacy aliases used by older analyzers.
        "contact_retention_mean": float(np.mean(retention)),
        "trans_drift_end_m": float(trans_drift[-1]),
        "trans_drift_max_m": float(trans_drift.max()),
        "trans_drift_mean_m": float(trans_drift.mean()),
        "rot_drift_end_rad": float(rot_drift[-1]),
        "rot_drift_max_rad": float(rot_drift.max()),
        "rot_drift_mean_rad": float(rot_drift.mean()),
        "contact_total_end": int(contact_totals[-1]),
        "contact_total_mean": float(contact_totals.mean()),
        "finger_force_active_ratio_mean": float(np.mean(active_ratio)),
        "slip_proxy_tangential_rel_vel_mean_mps": float(slip_tangential.mean()),
        "slip_proxy_tangential_rel_vel_max_mps": float(slip_tangential.max()),
        "slip_proxy_rot_rate_mean_radps": float(slip_rot.mean()),
        "slip_proxy_rot_rate_max_radps": float(slip_rot.max()),
        # Keep old key names but document units when dt!=1.
        "slip_proxy_tangential_rel_vel_mean": float(slip_tangential.mean()),
        "slip_proxy_tangential_rel_vel_max": float(slip_tangential.max()),
        "terminal_peg_ok": terminal_peg_ok,
        "peg_ok_end": terminal_peg_ok,
        "peg_ok_frac": float(np.mean(peg_ok_series)),
        "peg_retained_all_steps": bool(all(peg_ok_series)),
        "peg_contact_present_end": peg_contact_present_end,
        "peg_contact_absent_steps": peg_contact_absent_steps,
        "peg_world_z_end": peg_z_end,
        "peg_world_z_drop_from_root": peg_z_drop,
        "below_lift_threshold_end": bool(below_lift_threshold_end),
        "object_dropped_proxy": object_dropped_proxy,
        "insert_ok_any": bool(any(s.insert_ok for s in steps)),
        "insert_ok_end": bool(steps[-1].insert_ok),
        "unknown_contact_geoms": sorted(
            {n for s in steps for n in s.peg_hand_contact.unknown_geom_names}
        ),
    }
    if legacy_peg_loss_alias:
        out["peg_retained"] = bool(all(peg_ok_series))
        out["peg_loss"] = not terminal_peg_ok
    return out


def metrics_to_jsonable(m: GraspStepMetrics) -> dict[str, Any]:
    o2h = m.object_in_hand
    return {
        "object_in_hand": {
            "reference_body": o2h.reference_body,
            "translation": o2h.translation.tolist(),
            "rotvec": o2h.rotvec.tolist(),
        },
        "peg_hand_contact": asdict(m.peg_hand_contact),
        "right_finger_force": m.right_finger_force.tolist(),
        "right_finger_force_norm": m.right_finger_force_norm.tolist(),
        "contact_active": m.contact_active.astype(bool).tolist(),
        "tray_ok": m.tray_ok,
        "peg_ok": m.peg_ok,
        "insert_ok": m.insert_ok,
        "peg_world_pos": m.peg_world_pos.tolist(),
        "peg_world_quat_wxyz": m.peg_world_quat_wxyz.tolist(),
        "slip_proxy_tangential_rel_vel": m.slip_proxy_tangential_rel_vel,
        "slip_proxy_pose_drift_rate": m.slip_proxy_pose_drift_rate,
    }
