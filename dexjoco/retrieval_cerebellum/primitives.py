"""Scene-instantiated geometry and contact priors for assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


def _readonly_vector3(value: np.ndarray, *, name: str, unit: bool = False) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    vector = vector.copy()
    if unit:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            raise ValueError(f"{name} must have non-zero norm")
        vector /= norm
    vector.setflags(write=False)
    return vector


class PriorSource(str, Enum):
    """Origin of an instantiated prior."""

    PRIVILEGED = "privileged"
    RETRIEVED = "retrieved"


@dataclass(frozen=True)
class ContactRegion:
    """Category-level candidate contact region expressed in object coordinates."""

    finger: str
    center_object: np.ndarray
    normal_object: np.ndarray
    radius_m: float
    role: str = "stabilize"

    def __post_init__(self) -> None:
        if not self.finger.strip():
            raise ValueError("finger must be non-empty")
        if self.radius_m <= 0.0:
            raise ValueError(f"radius_m must be positive, got {self.radius_m}")
        object.__setattr__(
            self,
            "center_object",
            _readonly_vector3(self.center_object, name="center_object"),
        )
        object.__setattr__(
            self,
            "normal_object",
            _readonly_vector3(self.normal_object, name="normal_object", unit=True),
        )


@dataclass(frozen=True)
class AssemblyPrimitiveSet:
    """World-frame assembly primitives instantiated for the current scene."""

    family_id: str
    section: str
    peg_tip_world: np.ndarray
    peg_axis_world: np.ndarray
    hole_entry_world: np.ndarray
    hole_axis_world: np.ndarray
    hole_bottom_world: np.ndarray
    nominal_peg_size_m: float
    source: PriorSource
    contact_regions: tuple[ContactRegion, ...] = field(default_factory=tuple)
    clearance_m: float | None = None

    def __post_init__(self) -> None:
        if not self.family_id.strip():
            raise ValueError("family_id must be non-empty")
        if self.section not in ("round", "rectangular"):
            raise ValueError(f"Unsupported section {self.section!r}")
        if self.nominal_peg_size_m <= 0.0:
            raise ValueError(
                f"nominal_peg_size_m must be positive, got {self.nominal_peg_size_m}"
            )
        if self.clearance_m is not None and self.clearance_m < 0.0:
            raise ValueError(f"clearance_m must be non-negative, got {self.clearance_m}")

        object.__setattr__(
            self,
            "peg_tip_world",
            _readonly_vector3(self.peg_tip_world, name="peg_tip_world"),
        )
        object.__setattr__(
            self,
            "peg_axis_world",
            _readonly_vector3(self.peg_axis_world, name="peg_axis_world", unit=True),
        )
        object.__setattr__(
            self,
            "hole_entry_world",
            _readonly_vector3(self.hole_entry_world, name="hole_entry_world"),
        )
        object.__setattr__(
            self,
            "hole_axis_world",
            _readonly_vector3(self.hole_axis_world, name="hole_axis_world", unit=True),
        )
        object.__setattr__(
            self,
            "hole_bottom_world",
            _readonly_vector3(self.hole_bottom_world, name="hole_bottom_world"),
        )
        object.__setattr__(self, "contact_regions", tuple(self.contact_regions))

    @property
    def target_depth_m(self) -> float:
        """Positive depth from the opening toward the socket bottom."""
        bottom_offset = self.hole_bottom_world - self.hole_entry_world
        return max(0.0, -float(np.dot(bottom_offset, self.hole_axis_world)))

    @property
    def approach_height_m(self) -> float:
        """Signed peg-tip height outside the opening along the outward hole axis."""
        relative = self.peg_tip_world - self.hole_entry_world
        return float(np.dot(relative, self.hole_axis_world))

    @property
    def insertion_depth_m(self) -> float:
        return max(0.0, -self.approach_height_m)

    @property
    def lateral_error_vector_world(self) -> np.ndarray:
        relative = self.peg_tip_world - self.hole_entry_world
        lateral = relative - self.hole_axis_world * np.dot(relative, self.hole_axis_world)
        lateral = np.asarray(lateral, dtype=np.float64)
        lateral.setflags(write=False)
        return lateral

    @property
    def lateral_error_m(self) -> float:
        return float(np.linalg.norm(self.lateral_error_vector_world))

    @property
    def axis_error_rad(self) -> float:
        """Smallest line-alignment error; peg axis direction is sign-invariant."""
        cosine = float(np.clip(abs(np.dot(self.peg_axis_world, self.hole_axis_world)), 0.0, 1.0))
        return float(np.arccos(cosine))

    def feature_vector(self) -> np.ndarray:
        """Compact geometry features for handoff policies and low-level skills."""
        features = np.array(
            [
                self.lateral_error_m,
                self.axis_error_rad,
                self.approach_height_m,
                self.insertion_depth_m,
                self.target_depth_m,
                self.nominal_peg_size_m,
                -1.0 if self.clearance_m is None else self.clearance_m,
            ],
            dtype=np.float32,
        )
        return features
