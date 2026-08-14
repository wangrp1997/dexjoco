"""P0-Obs-D0 pure helpers: roots, windows, atomic split, leak checks (no MuJoCo)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

HORIZONS = (1, 4, 8, 16)
SPLIT_SEED = 20260814
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test = remainder


@dataclass(frozen=True)
class FrameRec:
    frame: int
    peg_ok: bool
    insert_ok: bool
    tip_dist_m: float
    tray_ok: bool
    contact_total: int
    terminated: bool
    truncated: bool
    gap_from_prev: bool  # True if frame != prev+1


@dataclass(frozen=True)
class RootRec:
    episode_index: int
    frame: int
    phase: str
    tip_dist_m: float
    peg_ok: bool
    insert_ok: bool
    contact_total: int

    @property
    def root_id(self) -> str:
        return f"{self.episode_index}:{self.frame}:{self.phase}"


def derive_roots_from_history(
    episode_index: int,
    history: Sequence[FrameRec],
    *,
    early_offset: int = 5,
    transport_tip_min_m: float = 0.08,
) -> list[RootRec]:
    """Mirror select_roots_for_episode criteria (early_grasp + transport only)."""
    if not history:
        return []
    first_peg_ok = None
    transport_frame = None
    for h in history:
        f = int(h.frame)
        if first_peg_ok is None and h.peg_ok and not h.insert_ok:
            first_peg_ok = f
        if (
            transport_frame is None
            and h.peg_ok
            and not h.insert_ok
            and first_peg_ok is not None
            and f >= first_peg_ok + 20
            and float(h.tip_dist_m) >= float(transport_tip_min_m)
        ):
            transport_frame = f

    roots: list[RootRec] = []
    by_f = {h.frame: h for h in history}
    if first_peg_ok is not None:
        early = min(first_peg_ok + int(early_offset), history[-1].frame)
        eh = by_f.get(early)
        if eh is not None and eh.peg_ok and not eh.insert_ok:
            roots.append(
                RootRec(
                    episode_index=int(episode_index),
                    frame=int(early),
                    phase="early_grasp",
                    tip_dist_m=float(eh.tip_dist_m),
                    peg_ok=True,
                    insert_ok=False,
                    contact_total=int(eh.contact_total),
                )
            )
    if transport_frame is not None:
        th = by_f[transport_frame]
        roots.append(
            RootRec(
                episode_index=int(episode_index),
                frame=int(transport_frame),
                phase="transport",
                tip_dist_m=float(th.tip_dist_m),
                peg_ok=True,
                insert_ok=False,
                contact_total=int(th.contact_total),
            )
        )
    return roots


def count_root_anchored_windows(
    roots: Sequence[RootRec],
    *,
    last_frame: int,
    terminated_at: int | None,
    horizons: Sequence[int] = HORIZONS,
) -> dict[str, int]:
    """Count roots that admit a contiguous window [f, f+H-1] before end/terminate."""
    out = {f"H{h}": 0 for h in horizons}
    end = int(last_frame)
    if terminated_at is not None:
        end = min(end, int(terminated_at))
    for r in roots:
        for h in horizons:
            if int(r.frame) + int(h) - 1 <= end:
                out[f"H{h}"] += 1
    return out


def count_phase_contiguous_windows(
    history: Sequence[FrameRec],
    *,
    horizons: Sequence[int] = HORIZONS,
) -> dict[str, int]:
    """Count starts of H consecutive peg_ok & ~insert_ok frames with no gaps."""
    out = {f"H{h}": 0 for h in horizons}
    if not history:
        return out
    n = len(history)
    phase = [bool(h.peg_ok and not h.insert_ok) for h in history]
    for hlen in horizons:
        for i in range(0, n - hlen + 1):
            if not all(phase[i : i + hlen]):
                continue
            frames = [history[j].frame for j in range(i, i + hlen)]
            if frames == list(range(frames[0], frames[0] + hlen)):
                out[f"H{hlen}"] += 1
    return out


def atomic_episode_split(
    episode_ids: Sequence[int],
    *,
    seed: int = SPLIT_SEED,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> dict[str, list[int]]:
    """Deterministic episode-atomic split (roots inherit episode split)."""
    ids = sorted(int(x) for x in episode_ids)
    # Stable shuffle via hash rank
    ranked = sorted(ids, key=lambda e: hashlib.sha256(f"{seed}:{e}".encode()).hexdigest())
    n = len(ranked)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    train = ranked[:n_train]
    val = ranked[n_train : n_train + n_val]
    test = ranked[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def assign_roots_to_splits(
    roots: Sequence[RootRec],
    episode_split: dict[str, list[int]],
) -> dict[str, list[str]]:
    ep_to_split = {}
    for sp, eps in episode_split.items():
        for e in eps:
            ep_to_split[int(e)] = sp
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for r in roots:
        sp = ep_to_split[int(r.episode_index)]
        out[sp].append(r.root_id)
    return out


def check_split_leakage(
    episode_split: dict[str, list[int]],
    root_split: dict[str, list[str]],
) -> dict[str, Any]:
    """Ensure episodes and root_ids are disjoint across splits."""
    issues: list[str] = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter_ep = set(episode_split[a]) & set(episode_split[b])
        if inter_ep:
            issues.append(f"episode_leak_{a}_{b}:{sorted(inter_ep)[:5]}")
        inter_r = set(root_split[a]) & set(root_split[b])
        if inter_r:
            issues.append(f"root_leak_{a}_{b}:{sorted(inter_r)[:5]}")
    # root episode membership must match
    ep_to_split = {}
    for sp, eps in episode_split.items():
        for e in eps:
            ep_to_split[int(e)] = sp
    for sp, rids in root_split.items():
        for rid in rids:
            ep = int(rid.split(":")[0])
            if ep_to_split.get(ep) != sp:
                issues.append(f"root_episode_mismatch:{rid}")
    return {"ok": len(issues) == 0, "issues": issues}


def digest_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
