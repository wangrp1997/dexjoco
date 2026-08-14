"""Calibrated right-hand finger target-offset pulses (P0-C1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from embodied_grasp_insertion.simulation.full_episode_utils import (
    LEFT_FINGER_IDX,
    RIGHT_FINGER_IDX,
    WRIST_IDX,
)

# hold44 / action44 right finger slice ↔ MuJoCo actuator order (ffa0..tha3).
RIGHT_FINGER_ACTUATOR_NAMES = (
    "ffa0_right",
    "ffa1_right",
    "ffa2_right",
    "ffa3_right",
    "mfa0_right",
    "mfa1_right",
    "mfa2_right",
    "mfa3_right",
    "rfa0_right",
    "rfa1_right",
    "rfa2_right",
    "rfa3_right",
    "tha0_right",
    "tha1_right",
    "tha2_right",
    "tha3_right",
)

RIGHT_FINGER_JOINT_NAMES = (
    "ffj0_right",
    "ffj1_right",
    "ffj2_right",
    "ffj3_right",
    "mfj0_right",
    "mfj1_right",
    "mfj2_right",
    "mfj3_right",
    "rfj0_right",
    "rfj1_right",
    "rfj2_right",
    "rfj3_right",
    "thj0_right",
    "thj1_right",
    "thj2_right",
    "thj3_right",
)

FINGER_TIP_FOR_JOINT = {
    "ffj0_right": "ff_tip_right",
    "ffj1_right": "ff_tip_right",
    "ffj2_right": "ff_tip_right",
    "ffj3_right": "ff_tip_right",
    "mfj0_right": "mf_tip_right",
    "mfj1_right": "mf_tip_right",
    "mfj2_right": "mf_tip_right",
    "mfj3_right": "mf_tip_right",
    "rfj0_right": "rf_tip_right",
    "rfj1_right": "rf_tip_right",
    "rfj2_right": "rf_tip_right",
    "rfj3_right": "rf_tip_right",
    "thj0_right": "th_tip_right",
    "thj1_right": "th_tip_right",
    "thj2_right": "th_tip_right",
    "thj3_right": "th_tip_right",
}


@dataclass
class JointSemantics:
    joint_index: int  # 0..15 within right fingers
    actuator_name: str
    joint_name: str
    flexion_sign: int  # +1 or -1 on action/target delta that flexes
    extension_sign: int
    confidence: str
    ambiguous_reason: str | None = None


def load_semantics(manifest: dict[str, Any]) -> list[JointSemantics]:
    out: list[JointSemantics] = []
    for row in manifest["right_joints"]:
        flex = row.get("flexion_direction")
        if flex not in ("positive", "negative"):
            raise RuntimeError(
                f"joint {row.get('joint_name')} lacks reliable flexion_direction: {flex}"
            )
        sign = 1 if flex == "positive" else -1
        out.append(
            JointSemantics(
                joint_index=int(row["joint_index"]),
                actuator_name=str(row["actuator_name"]),
                joint_name=str(row["joint_name"]),
                flexion_sign=sign,
                extension_sign=-sign,
                confidence=str(row.get("confidence", "unknown")),
                ambiguous_reason=row.get("ambiguous_reason"),
            )
        )
    if len(out) != 16:
        raise RuntimeError(f"expected 16 calibrated right joints, got {len(out)}")
    return out


def finger_scale(env) -> float:
    return float(env.finger_scale)


def clip_target_to_limits(
    env,
    target44: np.ndarray,
    *,
    right_only: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Clip right (and optionally left) finger targets to actuator ctrlrange."""
    out = np.asarray(target44, dtype=np.float64).copy()
    model = env._raw._model
    hits: list[dict[str, Any]] = []
    indices = RIGHT_FINGER_IDX if right_only else RIGHT_FINGER_IDX + LEFT_FINGER_IDX
    # Right actuators are model actuators 7..22 in assembly env.
    for local_i, act44_i in enumerate(RIGHT_FINGER_IDX):
        act_id = 7 + local_i
        lo, hi = np.asarray(model.actuator_ctrlrange[act_id], dtype=np.float64)
        before = float(out[act44_i])
        clipped = float(np.clip(before, lo, hi))
        if clipped != before:
            hits.append(
                {
                    "joint_index": local_i,
                    "actuator_name": RIGHT_FINGER_ACTUATOR_NAMES[local_i],
                    "before": before,
                    "after": clipped,
                    "lo": float(lo),
                    "hi": float(hi),
                }
            )
            out[act44_i] = clipped
    if not right_only:
        for local_i, act44_i in enumerate(LEFT_FINGER_IDX):
            act_id = 7 + 16 + local_i  # left hand actuators follow right in this model?
            # Safer: discover by name.
            name = RIGHT_FINGER_ACTUATOR_NAMES[local_i].replace("_right", "_left")
            try:
                act_id = int(model.actuator(name).id)
            except Exception:
                continue
            lo, hi = np.asarray(model.actuator_ctrlrange[act_id], dtype=np.float64)
            before = float(out[act44_i])
            clipped = float(np.clip(before, lo, hi))
            if clipped != before:
                hits.append(
                    {
                        "joint_index": local_i,
                        "actuator_name": name,
                        "before": before,
                        "after": clipped,
                        "lo": float(lo),
                        "hi": float(hi),
                    }
                )
                out[act44_i] = clipped
    return out, hits


def right_finger_signed_slack(env) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint positive/negative available target offset from current hold (rad)."""
    if env._hold44 is None:
        raise RuntimeError("env must be reset/restored before slack query")
    model = env._raw._model
    hold = np.asarray(env._hold44, dtype=np.float64)
    pos = np.zeros(16, dtype=np.float64)
    neg = np.zeros(16, dtype=np.float64)
    for local_i, act44_i in enumerate(RIGHT_FINGER_IDX):
        act_id = 7 + local_i
        lo, hi = np.asarray(model.actuator_ctrlrange[act_id], dtype=np.float64)
        cur = float(hold[act44_i])
        pos[local_i] = max(0.0, float(hi) - cur)
        neg[local_i] = max(0.0, cur - float(lo))
    return pos, neg


def max_feasible_scale_for_offset(
    offset: np.ndarray,
    pos_slack: np.ndarray,
    neg_slack: np.ndarray,
) -> float:
    """Largest s in [0,1] s.t. s*offset stays within signed slack (no clip)."""
    off = np.asarray(offset, dtype=np.float64).reshape(16)
    s = 1.0
    for i in range(16):
        v = float(off[i])
        if v > 0.0:
            room = float(pos_slack[i])
            if v > room + 1e-15:
                s = min(s, room / v)
        elif v < 0.0:
            room = float(neg_slack[i])
            if -v > room + 1e-15:
                s = min(s, room / (-v))
    return float(max(0.0, s))


def project_matched_feasible_offsets(
    env,
    offsets: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Project close/open/random offsets onto a shared feasible L2 budget.

    If any would clip, shrink all matched offsets by the same scale so clip_count=0
    and realized L2 budgets match.
    """
    pos, neg = right_finger_signed_slack(env)
    scales = {
        name: max_feasible_scale_for_offset(off, pos, neg) for name, off in offsets.items()
    }
    common = float(min(scales.values())) if scales else 1.0
    projected = {
        name: np.asarray(off, dtype=np.float64).reshape(16) * common
        for name, off in offsets.items()
    }
    l2s = {name: float(np.linalg.norm(v)) for name, v in projected.items()}
    # Force exact shared L2 for random vs close/open if present.
    ref_names = [n for n in ("calibrated_close_low", "calibrated_open_low") if n in projected]
    if ref_names and "random_matched" in projected:
        ref_l2 = float(np.mean([l2s[n] for n in ref_names]))
        vec = projected["random_matched"]
        n = float(np.linalg.norm(vec)) + 1e-12
        # Re-check feasibility after L2 rematch.
        cand = vec / n * ref_l2
        s2 = max_feasible_scale_for_offset(cand, pos, neg)
        projected["random_matched"] = cand * s2
        if s2 < 1.0 - 1e-12:
            # Shrink all to the new random-feasible L2.
            new_l2 = float(np.linalg.norm(projected["random_matched"]))
            for name in ref_names:
                cur = projected[name]
                cn = float(np.linalg.norm(cur)) + 1e-12
                projected[name] = cur / cn * new_l2
            common *= s2
    l2s = {name: float(np.linalg.norm(v)) for name, v in projected.items()}
    meta = {
        "per_joint_pos_slack": pos.tolist(),
        "per_joint_neg_slack": neg.tolist(),
        "per_mode_raw_scale": scales,
        "common_feasible_scale": common,
        "projected_l2": l2s,
        "shared_l2": float(np.mean(list(l2s.values()))) if l2s else 0.0,
    }
    return projected, meta


def target_offset_to_pulse_actions(
    env,
    *,
    right_offset_rad: np.ndarray,
    horizon: int,
    pulse_steps: int = 2,
    wrist_seq: np.ndarray | None = None,
    allow_clip: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build (horizon,44) deltas: reach right finger target offset then hold.

    Left fingers and (unless wrist_seq provided) wrists stay at 0 delta.
    When allow_clip=False (P0-C1.1), any limit hit is recorded and realized offset
    must equal the requested offset (caller should pre-project).
    """
    if env._hold44 is None:
        raise RuntimeError("env must be reset/restored before building interventions")
    offset = np.asarray(right_offset_rad, dtype=np.float64).reshape(16)
    scale = finger_scale(env)
    hold0 = np.asarray(env._hold44, dtype=np.float64).copy()
    desired = hold0.copy()
    desired[RIGHT_FINGER_IDX] = hold0[RIGHT_FINGER_IDX] + offset
    desired_clipped, limit_hits = clip_target_to_limits(env, desired, right_only=True)
    if allow_clip:
        desired_use = desired_clipped
    else:
        desired_use = desired
        # Still report what clip would have done.
    realized_offset = desired_use[RIGHT_FINGER_IDX] - hold0[RIGHT_FINGER_IDX]
    if not allow_clip and limit_hits:
        # Keep requested offset; fairness layer must fail rather than silently accept asymmetry.
        realized_offset = offset.copy()

    n_pulse = max(1, int(pulse_steps))
    actions = np.zeros((int(horizon), 44), dtype=np.float64)
    # Distribute normalized delta over first n_pulse steps.
    remaining = realized_offset.copy()
    for k in range(min(n_pulse, horizon)):
        steps_left = n_pulse - k
        chunk = remaining / float(steps_left)
        norm = chunk / (scale + 1e-12)
        # Avoid silent action saturation: if needed, leave leftover for meta.
        saturated = np.abs(norm) > 1.0 + 1e-12
        norm = np.clip(norm, -1.0, 1.0)
        actions[k, RIGHT_FINGER_IDX] = norm
        applied = norm * scale
        remaining = remaining - applied
        if saturated.any() and not allow_clip:
            # leftover tracks unrealized target; clip_count stays about joint limits.
            pass

    if wrist_seq is not None:
        w = np.asarray(wrist_seq, dtype=np.float64)
        if w.shape != actions.shape:
            raise ValueError(f"wrist_seq shape {w.shape} != {actions.shape}")
        actions[:, WRIST_IDX] = w[:, WRIST_IDX]

    meta = {
        "requested_right_offset_rad": offset.tolist(),
        "realized_right_offset_rad": realized_offset.tolist(),
        "realized_l2": float(np.linalg.norm(realized_offset)),
        "realized_abs_max": float(np.abs(realized_offset).max()),
        "pulse_steps": n_pulse,
        "finger_scale": scale,
        "limit_hits": limit_hits if allow_clip else ([] if not limit_hits else limit_hits),
        "clip_count": 0 if (not allow_clip and not limit_hits) else len(limit_hits),
        "leftover_offset_after_pulse": remaining.tolist(),
        "allow_clip": bool(allow_clip),
    }
    if not allow_clip:
        meta["clip_count"] = len(limit_hits)
    return actions, meta


def build_calibrated_right_offset(
    semantics: list[JointSemantics],
    *,
    mode: str,
    low_rad: float,
    medium_rad: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return 16D right finger target offset in radians."""
    off = np.zeros(16, dtype=np.float64)
    if mode == "hold":
        return off
    if mode == "calibrated_close_low":
        for s in semantics:
            off[s.joint_index] = s.flexion_sign * float(low_rad)
        return off
    if mode == "calibrated_close_medium":
        for s in semantics:
            off[s.joint_index] = s.flexion_sign * float(medium_rad)
        return off
    if mode == "calibrated_open_low":
        for s in semantics:
            off[s.joint_index] = s.extension_sign * float(low_rad)
        return off
    if mode == "random_matched":
        if rng is None:
            raise ValueError("rng required for random_matched")
        # Same L2 as close_low (all joints |low_rad|).
        budget = float(np.sqrt(16.0) * abs(low_rad))
        vec = rng.normal(size=16)
        n = float(np.linalg.norm(vec)) + 1e-12
        return vec / n * budget
    raise ValueError(f"unknown mode {mode}")


def build_right_demo_replay_actions(
    env,
    *,
    root_frame: int,
    horizon: int,
    wrist_seq: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replay only right demo finger targets; left fingers hold; wrists from wrist_seq."""
    from embodied_grasp_insertion.simulation.full_episode_utils import abs44_from_demo

    assert env._hold44 is not None and env._actions is not None
    scale = env._scale()
    hold = env._hold44.copy()
    actions = np.zeros((int(horizon), 44), dtype=np.float64)
    offsets = []
    for k in range(horizon):
        demo_idx = root_frame + k
        if demo_idx >= len(env._actions):
            break
        demo_abs = abs44_from_demo(env, demo_idx)
        delta = (demo_abs - hold) / (scale + 1e-12)
        delta = np.clip(delta, -1.0, 1.0)
        # Right fingers only.
        actions[k, RIGHT_FINGER_IDX] = delta[RIGHT_FINGER_IDX]
        hold = hold.copy()
        hold[RIGHT_FINGER_IDX] = hold[RIGHT_FINGER_IDX] + actions[k, RIGHT_FINGER_IDX] * scale[RIGHT_FINGER_IDX]
        offsets.append((hold[RIGHT_FINGER_IDX] - env._hold44[RIGHT_FINGER_IDX]).copy())
    if wrist_seq is not None:
        w = np.asarray(wrist_seq, dtype=np.float64)
        actions[:, WRIST_IDX] = w[: actions.shape[0], WRIST_IDX]
    final_off = offsets[-1] if offsets else np.zeros(16)
    meta = {
        "realized_right_offset_rad": final_off.tolist(),
        "realized_l2": float(np.linalg.norm(final_off)),
        "n_demo_steps": len(offsets),
    }
    return actions, meta


def assert_left_fingers_zero(actions: np.ndarray) -> None:
    if not np.allclose(actions[:, LEFT_FINGER_IDX], 0.0, atol=0.0):
        raise RuntimeError("left finger actions must remain exactly zero in P0-C1")
