"""P0-L1 privileged observability label schema (teacher-side only).

Not a training dataset. Not deployment observation.
Slip truth and fine contact modes are intentionally excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from embodied_grasp_insertion.geometry.target_hole import target_hole_from_raw
from embodied_grasp_insertion.physics.grasp_metrics import (
    REFERENCE_BODY,
    ObjectInHandPose,
    object_in_hand_pose,
    peg_hand_contact_counts,
)
from embodied_grasp_insertion.simulation.full_episode_utils import (
    geometry_family_id_from_env,
)

PROTOCOL = "P0-L1"
SCHEMA_VERSION = "privileged_label_v1"
CONTACT_FORCE_EPS = 0.05  # N; FingerForceLabeler tip force norm threshold
FINGER_ORDER = ("index", "middle", "ring", "thumb")  # matches FINGER_TIP_BODIES_RIGHT

# Explicitly NOT in v1 schema (must stay absent):
EXCLUDED_FIELDS = (
    "slip_truth",
    "slip",  # bare name forbidden; only slip_proxy allowed in later schemas
    "contact_mode_capture",
    "contact_mode_rim",
    "contact_mode_jam",
    "contact_mode_partial",
    "contact_mode_seated",
    "contact_mode_backout",
    "regrasp_needed",
    "peg_loss_risk",
)

VELOCITY_CONTRACT = {
    "method": "finite_difference_between_consecutive_control_frames",
    "dt_source": "control_dt_seconds = model.opt.timestep * frame_skip (default skip=10)",
    "linear": "v_lin = (t_k - t_{k-1}) / dt ; t in allegro_palm_right frame (m/s)",
    "angular": (
        "omega = rotvec(R_{k-1}^{-1} R_k) / dt ; "
        "R from o2h rotvec (peg relative palm); rad/s; NOT raw rotvec subtraction"
    ),
    "first_frame": "velocity fields null (insufficient history)",
    "reference_body": REFERENCE_BODY,
}


@dataclass(frozen=True)
class VelocityLabel:
    available: bool
    linear_mps: tuple[float, float, float] | None
    angular_radps: tuple[float, float, float] | None
    dt_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "linear_mps": None if self.linear_mps is None else list(self.linear_mps),
            "angular_radps": None if self.angular_radps is None else list(self.angular_radps),
            "dt_s": self.dt_s,
            "contract": VELOCITY_CONTRACT,
        }


def o2h_velocity_from_poses(
    prev: ObjectInHandPose,
    cur: ObjectInHandPose,
    dt_s: float,
) -> VelocityLabel:
    """Frozen finite-difference velocity (privilege)."""
    if dt_s <= 0:
        raise ValueError(f"dt_s must be > 0, got {dt_s}")
    if prev.reference_body != cur.reference_body:
        raise ValueError("reference_body mismatch between poses")
    v_lin = (np.asarray(cur.translation) - np.asarray(prev.translation)) / float(dt_s)
    r_prev = R.from_rotvec(np.asarray(prev.rotvec, dtype=np.float64))
    r_cur = R.from_rotvec(np.asarray(cur.rotvec, dtype=np.float64))
    dR = r_prev.inv() * r_cur
    omega = np.asarray(dR.as_rotvec(), dtype=np.float64) / float(dt_s)
    return VelocityLabel(
        available=True,
        linear_mps=tuple(float(x) for x in v_lin.tolist()),
        angular_radps=tuple(float(x) for x in omega.tolist()),
        dt_s=float(dt_s),
    )


def null_velocity() -> VelocityLabel:
    return VelocityLabel(available=False, linear_mps=None, angular_radps=None, dt_s=None)


def make_root_id(episode_index: int, frame: int, phase: str) -> str:
    return f"{int(episode_index)}:{int(frame)}:{str(phase)}"


def extract_privileged_frame(
    env,
    *,
    episode_index: int,
    frame: int,
    root_id: str,
    root_phase: str,
    prev_o2h: ObjectInHandPose | None,
    dt_s: float,
    contact_force_eps: float = CONTACT_FORCE_EPS,
) -> tuple[dict[str, Any], ObjectInHandPose]:
    """Derive one privileged label frame from live env state (after reset/replay)."""
    raw = env._raw
    o2h = object_in_hand_pose(raw)
    contact = peg_hand_contact_counts(raw)
    outcome = env._labeler.compute(raw)

    right_force = np.zeros(12, dtype=np.float64)
    if env._force_labeler is not None:
        ff = env._force_labeler.compute(raw)
        right_force = np.asarray(ff.right_finger_force, dtype=np.float64).copy()
    force_norm = np.linalg.norm(right_force.reshape(4, 3), axis=1)
    active = force_norm >= float(contact_force_eps)

    if prev_o2h is None:
        vel = null_velocity()
    else:
        vel = o2h_velocity_from_poses(prev_o2h, o2h, dt_s)

    hole = target_hole_from_raw(raw)
    family = geometry_family_id_from_env(env)

    label = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "privilege_only": True,
        "deployment_input": False,
        "provenance": {
            "episode_index": int(episode_index),
            "frame": int(frame),
            "env_t": int(env._t),
            "raw_env_step": int(getattr(raw, "env_step", -1)),
            "sim_time_s": float(raw._data.time),
            "geometry_family_id": str(family),
            "target_instance_id": str(hole.target_instance_id),
            "socket_site": str(hole.socket_site),
            "root_id": str(root_id),
            "root_phase": str(root_phase),
            "zarr_path": str(env._spec.zarr_path) if env._spec is not None else None,
            "reference_body": REFERENCE_BODY,
            "control_dt_s": float(dt_s),
            "contact_force_eps_N": float(contact_force_eps),
            "finger_order": list(FINGER_ORDER),
        },
        "object_in_hand_pose_6d": {
            "translation_m": o2h.translation.astype(np.float64).tolist(),
            "rotvec_rad": o2h.rotvec.astype(np.float64).tolist(),
            "reference_body": o2h.reference_body,
        },
        "object_in_hand_velocity": vel.to_dict(),
        "peg_hand_contact": {
            "total": int(contact.total),
            "by_finger": {k: int(contact.by_class[k]) for k in ("palm", *FINGER_ORDER)},
            "unknown_count": int(contact.unknown_count),
        },
        "finger_force": {
            "right_force_world_N": right_force.astype(np.float64).tolist(),
            "right_force_norm_N": force_norm.astype(np.float64).tolist(),
            "contact_active": active.astype(bool).tolist(),
            "finger_order": list(FINGER_ORDER),
            "eps_N": float(contact_force_eps),
            "source": "FingerForceLabeler.cfrc_ext_tip_bodies",
        },
        "outcome_raw": {
            "tray_ok": bool(outcome.tray_ok),
            "peg_ok": bool(outcome.peg_ok),
            "insert_ok": bool(outcome.insert_ok),
        },
        "excluded_explicitly": list(EXCLUDED_FIELDS),
    }
    validate_privileged_label(label)
    return label, o2h


def validate_privileged_label(label: dict[str, Any]) -> None:
    if label.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    if label.get("privilege_only") is not True:
        raise ValueError("privilege_only must be true")
    if label.get("deployment_input") is not False:
        raise ValueError("deployment_input must be false")
    for bad in EXCLUDED_FIELDS:
        if bad in label:
            raise ValueError(f"forbidden field present: {bad}")
        # Also forbid nested bare 'slip' keys that claim truth.
    if "slip" in label or "slip_truth" in label:
        raise ValueError("slip truth forbidden in L1")
    pose = label["object_in_hand_pose_6d"]
    if len(pose["translation_m"]) != 3 or len(pose["rotvec_rad"]) != 3:
        raise ValueError("o2h pose dims")
    vel = label["object_in_hand_velocity"]
    if vel["available"]:
        if vel["linear_mps"] is None or vel["angular_radps"] is None:
            raise ValueError("velocity available but null vectors")
        if len(vel["linear_mps"]) != 3 or len(vel["angular_radps"]) != 3:
            raise ValueError("velocity dims")
    else:
        if vel["linear_mps"] is not None or vel["angular_radps"] is not None:
            raise ValueError("unavailable velocity must be null")
    ff = label["finger_force"]
    if len(ff["right_force_world_N"]) != 12:
        raise ValueError("finger force must be 12")
    if len(ff["right_force_norm_N"]) != 4 or len(ff["contact_active"]) != 4:
        raise ValueError("finger norm/active must be 4")
    for k in ("tray_ok", "peg_ok", "insert_ok"):
        if type(label["outcome_raw"][k]) is not bool:
            raise ValueError(f"outcome {k} must be bool")


def schema_document() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "privilege_only": True,
        "included": [
            "object_in_hand_pose_6d",
            "object_in_hand_velocity (finite-diff contract)",
            "peg_hand_contact count/by_finger",
            "finger_force norm/contact_active",
            "outcome_raw tray_ok/peg_ok/insert_ok",
            "provenance family/instance/episode/frame/root",
        ],
        "excluded": list(EXCLUDED_FIELDS),
        "velocity_contract": VELOCITY_CONTRACT,
        "contact_force_eps_N": CONTACT_FORCE_EPS,
        "finger_order": list(FINGER_ORDER),
        "reference_body": REFERENCE_BODY,
        "notes": [
            "Not a training dataset.",
            "Never write into pilot_micro_demo_v0.",
            "Do not name slip without _proxy suffix; L1 omits slip entirely.",
            "Fine contact modes require a separate contract before generation.",
        ],
    }


def labels_bit_digest(frames: list[dict[str, Any]]) -> str:
    """Stable digest for bit-exact repeat checks (JSON canonical)."""
    import hashlib
    import json

    payload = json.dumps(frames, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
