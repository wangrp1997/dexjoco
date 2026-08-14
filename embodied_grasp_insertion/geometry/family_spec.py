"""Unified GeometryFamilySpec for multi-peg/socket plumbing (P0-S0)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEXJOCO_XMLS = Path("/home/wangrenpeng/dexjoco/dexjoco/dexjoco/sim/envs/xmls")
INDUSTREAL_MESH = DEXJOCO_XMLS / "industreal/mesh/industreal_pegs"
INDUSTREAL_YAML = DEXJOCO_XMLS / "industreal/yaml/industreal_asset_info_pegs.yaml"
OFFICIAL_8MM_PEG_XML = DEXJOCO_XMLS / "industreal_round_peg_8mm.xml"
OFFICIAL_8MM_SOCKET_XML = DEXJOCO_XMLS / "industreal_tray_insert_round_peg_8mm.xml"

# Arena convention: visual meshes use this scale. Not a physical SI unit by itself.
ARENA_MESH_SCALE = 4.5

# Unified semantic names inside temporary smoke XML (not official 8mm body names).
PEG_BODY = "peg_body"
PEG_JOINT = "peg_joint"
PEG_VISUAL = "peg_visual"
PEG_COLLISION = "peg_collision"
PEG_TIP_SITE = "peg_tip_site"
PEG_GRASP_SITE = "peg_grasp_site"
SOCKET_BODY = "socket_body"
SOCKET_JOINT = "socket_joint"
SOCKET_VISUAL = "socket_visual"
SOCKET_BASE = "socket_base"
SOCKET_SITE = "socket_site"
INSERT_BOTTOM = "insert_bottom_contact"


@dataclass
class GeometryFamilySpec:
    family_id: str
    section: str  # round | rectangular
    nominal_size_mm: int
    peg_mesh: str
    socket_mesh: str
    peg_urdf_key: str
    hole_urdf_key: str
    # Nominal SI from industreal_asset_info_pegs.yaml (raw mesh units).
    peg_diameter_m: float | None = None
    peg_width_m: float | None = None
    peg_depth_m: float | None = None
    peg_length_m: float = 0.05
    hole_diameter_m: float | None = None
    hole_width_m: float | None = None
    hole_height_m: float = 0.028
    hole_depth_m: float = 0.023
    clearance_diameter_m: float | None = None
    clearance_width_m: float | None = None
    # Measured from mesh vertices (raw, before XML scale).
    peg_mesh_span_xyz_m: list[float] = field(default_factory=list)
    socket_mesh_span_xyz_m: list[float] = field(default_factory=list)
    # Applied mesh scale and resulting MuJoCo visual sizes.
    asset_scale: float = ARENA_MESH_SCALE
    asset_scale_source: str = "arena_arm_hand_bimanual_assembly_8mm_xml_convention"
    peg_visual_span_xyz_m: list[float] = field(default_factory=list)
    socket_visual_span_xyz_m: list[float] = field(default_factory=list)
    # Collision parameters derived from measured scaled spans (NOT copied from 8mm).
    collision: dict[str, Any] = field(default_factory=dict)
    # Unified semantic names
    peg_body: str = PEG_BODY
    peg_tip_site: str = PEG_TIP_SITE
    peg_grasp_site: str = PEG_GRASP_SITE
    peg_collision_geom: str = PEG_COLLISION
    socket_body: str = SOCKET_BODY
    socket_site: str = SOCKET_SITE
    insert_bottom_geom: str = INSERT_BOTTOM
    mating_frame: str = "socket_site_z_up"
    symmetry: str = "SO2_about_insertion_axis"  # or C2/D2 for rectangular
    insertion_axis: str = "socket_site_+z_then_world_z_aligned"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def from_dict(d: dict[str, Any]) -> GeometryFamilySpec:
    known = set(GeometryFamilySpec.__dataclass_fields__.keys())
    return GeometryFamilySpec(**{k: v for k, v in d.items() if k in known})


def family_id_for(section: str, size_mm: int) -> str:
    return f"{section}_{size_mm}mm"


def mesh_path(name: str) -> Path:
    return INDUSTREAL_MESH / name
