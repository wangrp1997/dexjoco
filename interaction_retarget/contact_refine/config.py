"""ContactOpt capsule params for Allegro + industreal (scaled from MANO defaults).

Source defaults: refs/contactopt/contactopt/optimize_pose.py
  caps_top=0.0005, caps_bot=-0.001, caps_rad=0.001
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContactCapsConfig:
    caps_top: float = 0.004
    caps_bot: float = -0.012
    caps_rad: float = 0.010
    contact_norm_method: int = 0


# Sweep grid for industreal mesh scale 4.5
CAPS_RAD_SWEEP = (0.001, 0.005, 0.008, 0.010, 0.012, 0.015)

DEFAULT_CAPS = ContactCapsConfig()
