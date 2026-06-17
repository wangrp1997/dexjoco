"""Per-finger contact forces and wrist wrenches from MuJoCo replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assembly_contacts import AssemblyContactLabeler, AssemblyOutcome

FINGER_TIP_BODIES_RIGHT = (
    "ff_tip_right",
    "mf_tip_right",
    "rf_tip_right",
    "th_tip_right",
)
FINGER_TIP_BODIES_LEFT = (
    "ff_tip_left",
    "mf_tip_left",
    "rf_tip_left",
    "th_tip_left",
)
WRIST_FORCE_SENSORS = (
    "panda/wrist_force_right",
    "panda/wrist_force_left",
)
WRIST_TORQUE_SENSORS = (
    "panda/wrist_torque_right",
    "panda/wrist_torque_left",
)
WRIST_ATTACHMENT_SITES = (
    "attachment_site_right",
    "attachment_site_left",
)
NUM_FINGERS = 4
FINGER_FORCE_DIM = NUM_FINGERS * 3
WRIST_FT_DIM = 6


@dataclass
class ForceFrame:
    """Privileged sim forces for one replay step (post env.step)."""

    right_finger_force: np.ndarray  # (12,) world frame [fx,fy,fz] x 4 fingers
    left_finger_force: np.ndarray  # (12,)
    wrist_ft_right: np.ndarray  # (6,) [fx,fy,fz,tx,ty,tz] world frame
    wrist_ft_left: np.ndarray  # (6,)
    outcome: AssemblyOutcome


class FingerForceLabeler:
    """Extract fingertip contact forces and wrist wrenches from MuJoCo state."""

    def __init__(self, raw_env) -> None:
        model = raw_env._model
        self._right_tip_body_ids = tuple(
            int(model.body(name).id) for name in FINGER_TIP_BODIES_RIGHT
        )
        self._left_tip_body_ids = tuple(
            int(model.body(name).id) for name in FINGER_TIP_BODIES_LEFT
        )
        self._wrist_site_ids = tuple(
            int(model.site(name).id) for name in WRIST_ATTACHMENT_SITES
        )
        self._wrist_force_sensors = WRIST_FORCE_SENSORS
        self._wrist_torque_sensors = WRIST_TORQUE_SENSORS
        self._contact_labeler = AssemblyContactLabeler(raw_env)

    def reset_reference(self, raw_env) -> None:
        self._contact_labeler.reset_reference(raw_env)

    def compute(self, raw_env) -> ForceFrame:
        data = raw_env._data
        right_finger = _finger_forces_from_cfrc(data, self._right_tip_body_ids)
        left_finger = _finger_forces_from_cfrc(data, self._left_tip_body_ids)
        wrist_right = _wrist_ft_from_sensors(
            data,
            site_id=self._wrist_site_ids[0],
            force_sensor=self._wrist_force_sensors[0],
            torque_sensor=self._wrist_torque_sensors[0],
        )
        wrist_left = _wrist_ft_from_sensors(
            data,
            site_id=self._wrist_site_ids[1],
            force_sensor=self._wrist_force_sensors[1],
            torque_sensor=self._wrist_torque_sensors[1],
        )
        outcome = self._contact_labeler.compute(raw_env)
        return ForceFrame(
            right_finger_force=right_finger,
            left_finger_force=left_finger,
            wrist_ft_right=wrist_right,
            wrist_ft_left=wrist_left,
            outcome=outcome,
        )


def _finger_forces_from_cfrc(data, tip_body_ids: tuple[int, ...]) -> np.ndarray:
    forces = np.zeros(FINGER_FORCE_DIM, dtype=np.float64)
    for finger_idx, body_id in enumerate(tip_body_ids):
        force = np.asarray(data.cfrc_ext[body_id, :3], dtype=np.float64)
        forces[finger_idx * 3 : (finger_idx + 1) * 3] = force
    return forces


def _wrist_ft_from_sensors(
    data,
    *,
    site_id: int,
    force_sensor: str,
    torque_sensor: str,
) -> np.ndarray:
    """Read wrist 6D wrench from MuJoCo site sensors, expressed in world frame."""
    site_rot = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    force_local = np.asarray(data.sensor(force_sensor).data, dtype=np.float64)
    force_world = site_rot @ force_local
    try:
        torque_local = np.asarray(data.sensor(torque_sensor).data, dtype=np.float64)
    except KeyError:
        torque_local = np.zeros(3, dtype=np.float64)
    torque_world = site_rot @ torque_local
    return np.concatenate([force_world, torque_world])


def summarize_force_episode(frames: list[ForceFrame], *, contact_eps: float = 0.5) -> dict:
    """Aggregate per-episode force / contact statistics for logging."""
    if not frames:
        return {
            "num_frames": 0,
            "insert_ok_rate": 0.0,
            "right_finger_contact_rate": [0.0] * NUM_FINGERS,
            "left_finger_contact_rate": [0.0] * NUM_FINGERS,
            "right_finger_force_norm_mean": [0.0] * NUM_FINGERS,
            "left_finger_force_norm_mean": [0.0] * NUM_FINGERS,
            "right_finger_force_norm_max": 0.0,
            "left_finger_force_norm_max": 0.0,
            "wrist_force_norm_mean_right": 0.0,
            "wrist_force_norm_mean_left": 0.0,
            "wrist_torque_norm_mean_right": 0.0,
            "wrist_torque_norm_mean_left": 0.0,
            "wrist_force_norm_max_right": 0.0,
            "wrist_force_norm_max_left": 0.0,
            "wrist_torque_norm_max_right": 0.0,
            "wrist_torque_norm_max_left": 0.0,
        }

    n = len(frames)
    insert_ok = np.mean([float(f.outcome.insert_ok) for f in frames])

    def finger_stats(side: str) -> tuple[list[float], list[float], float]:
        contact_rates: list[float] = []
        norm_means: list[float] = []
        max_norm = 0.0
        for finger_idx in range(NUM_FINGERS):
            norms = []
            contacts = 0
            for frame in frames:
                vec = _finger_vec(frame, side, finger_idx)
                norm = float(np.linalg.norm(vec))
                norms.append(norm)
                if norm > contact_eps:
                    contacts += 1
            contact_rates.append(contacts / n)
            norm_means.append(float(np.mean(norms)) if norms else 0.0)
            max_norm = max(max_norm, max(norms) if norms else 0.0)
        return contact_rates, norm_means, max_norm

    r_contact, r_mean, r_max = finger_stats("right")
    l_contact, l_mean, l_max = finger_stats("left")

    wrist_r_f = np.array([np.linalg.norm(f.wrist_ft_right[:3]) for f in frames])
    wrist_l_f = np.array([np.linalg.norm(f.wrist_ft_left[:3]) for f in frames])
    wrist_r_t = np.array([np.linalg.norm(f.wrist_ft_right[3:]) for f in frames])
    wrist_l_t = np.array([np.linalg.norm(f.wrist_ft_left[3:]) for f in frames])

    return {
        "num_frames": n,
        "insert_ok_rate": float(insert_ok),
        "right_finger_contact_rate": r_contact,
        "left_finger_contact_rate": l_contact,
        "right_finger_force_norm_mean": r_mean,
        "left_finger_force_norm_mean": l_mean,
        "right_finger_force_norm_max": r_max,
        "left_finger_force_norm_max": l_max,
        "wrist_force_norm_mean_right": float(wrist_r_f.mean()),
        "wrist_force_norm_mean_left": float(wrist_l_f.mean()),
        "wrist_torque_norm_mean_right": float(wrist_r_t.mean()),
        "wrist_torque_norm_mean_left": float(wrist_l_t.mean()),
        "wrist_force_norm_max_right": float(wrist_r_f.max()),
        "wrist_force_norm_max_left": float(wrist_l_f.max()),
        "wrist_torque_norm_max_right": float(wrist_r_t.max()),
        "wrist_torque_norm_max_left": float(wrist_l_t.max()),
    }


def format_episode_force_summary(episode_index: int, stats: dict) -> str:
    r_contact = stats["right_finger_contact_rate"]
    l_contact = stats["left_finger_contact_rate"]
    r_mean = stats["right_finger_force_norm_mean"]
    l_mean = stats["left_finger_force_norm_mean"]
    return (
        f"ep {episode_index}: frames={stats['num_frames']} "
        f"insert_ok={stats['insert_ok_rate']:.1%} | "
        f"R contact%[ff,mf,rf,th]=[{', '.join(f'{x:.1%}' for x in r_contact)}] "
        f"L=[{', '.join(f'{x:.1%}' for x in l_contact)}] | "
        f"R |F|_mean[N]={', '.join(f'{x:.2f}' for x in r_mean)} "
        f"L=[{', '.join(f'{x:.2f}' for x in l_mean)}] | "
        f"wrist_R |F|={stats['wrist_force_norm_mean_right']:.2f}N "
        f"|T|={stats['wrist_torque_norm_mean_right']:.2f}Nm "
        f"wrist_L |F|={stats['wrist_force_norm_mean_left']:.2f}N "
        f"|T|={stats['wrist_torque_norm_mean_left']:.2f}Nm | "
        f"max_finger_R={stats['right_finger_force_norm_max']:.2f}N "
        f"max_finger_L={stats['left_finger_force_norm_max']:.2f}N"
    )


def _finger_vec(frame: ForceFrame, side: str, finger_idx: int) -> np.ndarray:
    arr = frame.right_finger_force if side == "right" else frame.left_finger_force
    return arr[finger_idx * 3 : (finger_idx + 1) * 3]
