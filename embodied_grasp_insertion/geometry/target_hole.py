"""Target-hole semantics helpers (P0-S0.3).

Single-target hole **metadata / plumbing** only.
Does NOT claim the policy knows which hole to insert into.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dexjoco.sim.envs.assembly_geometry import (
    AssemblyGeometryNames,
    names_for_family,
    names_for_socket_instance,
    names_from_raw,
)
from hybrid_insert.assembly_contacts import AssemblyContactLabeler
from interaction_retarget.skill_replay.insert import _insert_geometry


@dataclass(frozen=True)
class TargetHoleSpec:
    geometry_family_id: str
    peg_body: str
    socket_body: str
    socket_site: str
    socket_bottom: str
    peg_tip_site: str
    # Distinguishes hole instances even within the same family (site identity).
    target_instance_id: str = ""

    def __post_init__(self) -> None:
        if not self.target_instance_id:
            object.__setattr__(self, "target_instance_id", self.socket_site)

    @classmethod
    def from_names(cls, names: AssemblyGeometryNames) -> "TargetHoleSpec":
        return cls(
            geometry_family_id=names.family_id,
            peg_body=names.peg_body,
            socket_body=names.socket_body,
            socket_site=names.socket_site,
            socket_bottom=names.socket_bottom,
            peg_tip_site=names.peg_tip_site,
            target_instance_id=names.socket_site,
        )

    @classmethod
    def from_family(cls, family_id: str) -> "TargetHoleSpec":
        return cls.from_names(names_for_family(family_id))

    @classmethod
    def from_family_instance(cls, family_id: str, instance_key: str = "primary") -> "TargetHoleSpec":
        return cls.from_names(names_for_socket_instance(family_id, instance_key))

    def with_instance(self, target_instance_id: str, *, socket_site: str | None = None) -> "TargetHoleSpec":
        return TargetHoleSpec(
            geometry_family_id=self.geometry_family_id,
            peg_body=self.peg_body,
            socket_body=self.socket_body,
            socket_site=socket_site or self.socket_site,
            socket_bottom=self.socket_bottom,
            peg_tip_site=self.peg_tip_site,
            target_instance_id=str(target_instance_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def target_hole_from_raw(raw) -> TargetHoleSpec:
    return TargetHoleSpec.from_names(names_from_raw(raw))


def list_same_family_instances(family_id: str, *, keys: tuple[str, ...] = ("primary", "b")) -> list[TargetHoleSpec]:
    """Enumerate explicit same-family socket instances (S0.3b plumbing)."""
    return [TargetHoleSpec.from_family_instance(family_id, k) for k in keys]


def socket_site_pose_xyz(raw, socket_site: str) -> np.ndarray:
    model, data = raw._model, raw._data
    sid = int(model.site(socket_site).id)
    return np.asarray(data.site_xpos[sid], dtype=np.float64).copy()


def claim_matches_env(claimed: TargetHoleSpec, true_target: TargetHoleSpec) -> bool:
    """Match family AND hole instance identity (site / instance id)."""
    return (
        claimed.geometry_family_id == true_target.geometry_family_id
        and claimed.target_instance_id == true_target.target_instance_id
        and claimed.socket_site == true_target.socket_site
    )


def semantic_target_features(
    raw,
    *,
    claimed_target: TargetHoleSpec | None = None,
    claimed_socket_pose_xyz: np.ndarray | None = None,
) -> dict[str, Any]:
    """Geometry features relative to claimed target (default = env's true socket).

    Prefer passing ``claimed_target`` (and optional external ``claimed_socket_pose_xyz``
    when the claimed site is not in the model) so wrong-target checks exercise this API.
    """
    names = names_from_raw(raw)
    true_target = TargetHoleSpec.from_names(names)
    claimed = claimed_target or true_target
    tip, socket_true, hole, dist_true = _insert_geometry(raw)

    model, data = raw._model, raw._data
    socket_claimed = None
    dist_claimed = None

    if claimed_socket_pose_xyz is not None:
        socket_claimed = np.asarray(claimed_socket_pose_xyz, dtype=np.float64).reshape(3)
        dist_claimed = float(np.linalg.norm(tip - socket_claimed))
    elif claimed.socket_site == true_target.socket_site and claim_matches_env(claimed, true_target):
        socket_claimed = socket_true
        dist_claimed = float(dist_true)
    else:
        try:
            sid = int(model.site(claimed.socket_site).id)
            socket_claimed = np.asarray(data.site_xpos[sid], dtype=np.float64)
            dist_claimed = float(np.linalg.norm(tip - socket_claimed))
        except Exception:
            socket_claimed = None
            dist_claimed = None

    matches = claim_matches_env(claimed, true_target)
    if claimed_socket_pose_xyz is not None:
        # External pose claim cannot be the env site instance.
        matches = False

    return {
        "true_target": true_target.to_dict(),
        "claimed_target": claimed.to_dict(),
        "claim_matches_env": matches,
        "tip": np.asarray(tip, dtype=np.float64).tolist(),
        "socket_true": np.asarray(socket_true, dtype=np.float64).tolist(),
        "socket_claimed": None if socket_claimed is None else socket_claimed.tolist(),
        "hole_axis_true": np.asarray(hole, dtype=np.float64).tolist(),
        "tip_to_true_m": float(dist_true),
        "tip_to_claimed_m": dist_claimed,
        "label_vector": [
            true_target.geometry_family_id,
            true_target.target_instance_id,
            true_target.socket_body,
            true_target.socket_site,
            float(dist_true),
            float(socket_true[0]),
            float(socket_true[1]),
            float(socket_true[2]),
        ],
    }


def wrong_target_offset_pose(socket_xyz: np.ndarray, *, offset_m: float = 0.12) -> np.ndarray:
    """Synthetic wrong target pose: shift socket laterally (same scene)."""
    p = np.asarray(socket_xyz, dtype=np.float64).copy()
    p[1] += float(offset_m)
    return p


def tip_to_pose(tip_xyz: np.ndarray, pose_xyz: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(tip_xyz) - np.asarray(pose_xyz)))


def make_family_labeler(raw, family_id: str | None = None) -> AssemblyContactLabeler:
    return AssemblyContactLabeler(raw, geometry_family=family_id)
