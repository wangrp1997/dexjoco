"""Assembly peg/socket MuJoCo name resolution (formal arena).

Default family remains round_8mm with existing asset/arena XML files unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_FAMILY_ID = "round_8mm"
XMLS_DIR = Path(__file__).resolve().parent / "xmls"
DEFAULT_ARENA_XML = XMLS_DIR / "arena_arm_hand_bimanual_assembly.xml"


@dataclass(frozen=True)
class AssemblyGeometryNames:
    family_id: str
    section: str  # round | rectangular
    size_mm: int
    peg_body: str
    peg_joint: str
    peg_visual: str
    peg_collision: str
    peg_collision_upper: str
    peg_tip_site: str
    peg_grasp_site: str
    socket_body: str
    socket_joint: str
    socket_visual: str
    socket_base: str
    socket_site: str
    socket_bottom: str
    peg_mesh: str
    socket_mesh: str
    peg_asset_xml: str
    socket_asset_xml: str

    @property
    def is_default_8mm_round(self) -> bool:
        return self.family_id == DEFAULT_FAMILY_ID


def parse_family_id(family_id: str) -> tuple[str, int]:
    fid = str(family_id).strip()
    if fid.endswith("mm"):
        fid = fid[:-2]
    if "_" not in fid:
        raise ValueError(f"bad family_id={family_id!r}; expected e.g. round_8mm")
    section, size_s = fid.rsplit("_", 1)
    if section not in ("round", "rectangular"):
        raise ValueError(f"bad section in family_id={family_id!r}")
    return section, int(size_s)


def names_for_family(family_id: str = DEFAULT_FAMILY_ID) -> AssemblyGeometryNames:
    section, size_mm = parse_family_id(family_id)
    peg = f"industreal_{section}_peg_{size_mm}mm"
    sock = f"industreal_tray_insert_{section}_peg_{size_mm}mm"
    return AssemblyGeometryNames(
        family_id=f"{section}_{size_mm}mm",
        section=section,
        size_mm=size_mm,
        peg_body=peg,
        peg_joint=f"{peg}_joint",
        peg_visual=f"{peg}_visual",
        peg_collision=f"{peg}_collision",
        peg_collision_upper=f"{peg}_collision_upper",
        peg_tip_site=f"{peg}_tip_site",
        peg_grasp_site=f"{peg}_grasp_site",
        socket_body=sock,
        socket_joint=f"{sock}_joint",
        socket_visual=f"{sock}_visual",
        socket_base=f"{sock}_base",
        socket_site=f"{sock}_socket_site",
        socket_bottom=f"{sock}_bottom_contact",
        peg_mesh=f"{peg}_mesh",
        socket_mesh=f"{sock}_mesh",
        peg_asset_xml=f"{peg}.xml",
        socket_asset_xml=f"{sock}.xml",
    )


def names_for_socket_instance(
    family_id: str,
    instance_key: str = "primary",
) -> AssemblyGeometryNames:
    """Same family, distinct socket body/site/joint names for multi-hole arenas.

    ``primary`` keeps the canonical family names (env true target).
    Other keys (e.g. ``b``) suffix socket identifiers with ``__inst_<key>``.
    """
    base = names_for_family(family_id)
    key = str(instance_key or "primary").strip()
    if key in ("", "primary", "0", "a"):
        return base
    tag = key if key.startswith("inst_") else f"inst_{key}"
    sock = f"{base.socket_body}__{tag}"
    return AssemblyGeometryNames(
        family_id=base.family_id,
        section=base.section,
        size_mm=base.size_mm,
        peg_body=base.peg_body,
        peg_joint=base.peg_joint,
        peg_visual=base.peg_visual,
        peg_collision=base.peg_collision,
        peg_collision_upper=base.peg_collision_upper,
        peg_tip_site=base.peg_tip_site,
        peg_grasp_site=base.peg_grasp_site,
        socket_body=sock,
        socket_joint=f"{sock}_joint",
        socket_visual=f"{sock}_visual",
        socket_base=f"{sock}_base",
        socket_site=f"{sock}_socket_site",
        socket_bottom=f"{sock}_bottom_contact",
        peg_mesh=base.peg_mesh,
        socket_mesh=f"{sock}_mesh",
        peg_asset_xml=base.peg_asset_xml,
        socket_asset_xml=f"{sock}.xml",
    )


def arena_xml_path(family_id: str = DEFAULT_FAMILY_ID, *, xmls_dir: Path | None = None) -> Path:
    xmls = Path(xmls_dir) if xmls_dir is not None else XMLS_DIR
    names = names_for_family(family_id)
    if names.is_default_8mm_round:
        return xmls / "arena_arm_hand_bimanual_assembly.xml"
    return xmls / f"arena_arm_hand_bimanual_assembly__{names.family_id}.xml"


def names_from_raw(raw) -> AssemblyGeometryNames:
    """Resolve names from env (`_geom_names` / `geometry_family`) or default 8mm."""
    if getattr(raw, "_geom_names", None) is not None:
        return raw._geom_names
    family = getattr(raw, "geometry_family", None) or DEFAULT_FAMILY_ID
    return names_for_family(family)
