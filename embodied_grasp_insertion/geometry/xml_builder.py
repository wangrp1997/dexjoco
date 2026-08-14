"""Build temporary MuJoCo XML for geometry families (no formal arena edits)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from embodied_grasp_insertion.geometry.family_spec import (
    INSERT_BOTTOM,
    PEG_BODY,
    PEG_COLLISION,
    PEG_GRASP_SITE,
    PEG_JOINT,
    PEG_TIP_SITE,
    PEG_VISUAL,
    SOCKET_BASE,
    SOCKET_BODY,
    SOCKET_JOINT,
    SOCKET_SITE,
    SOCKET_VISUAL,
    GeometryFamilySpec,
    mesh_path,
)


def _escape(p: Path) -> str:
    return str(p.resolve())


def build_family_xml(spec: GeometryFamilySpec, *, mismatch_socket: GeometryFamilySpec | None = None) -> str:
    """Minimal world: peg + socket freejoints. Collision from measured scaled spans."""
    sock = mismatch_socket or spec
    peg_mesh = mesh_path(spec.peg_mesh)
    sock_mesh = mesh_path(sock.socket_mesh)
    scale = float(spec.asset_scale)
    peg_col_spec = spec.collision
    sock_col = sock.collision
    # Peg collision: cylinder (round) or box (rectangular) centered along peg length.
    peg_half_len = float(peg_col_spec["peg_half_length_m"])
    peg_z0 = float(peg_col_spec["peg_collision_center_z_m"])
    tip_z = float(peg_col_spec["peg_tip_site_z_m"])
    grasp_z = float(peg_col_spec["peg_grasp_site_z_m"])
    if spec.section == "round":
        peg_col = (
            f'      <geom name="{PEG_COLLISION}" type="cylinder" '
            f'pos="0 0 {peg_z0:.6f}" size="{peg_col_spec["peg_radius_m"]:.6f} {peg_half_len:.6f}" '
            f'friction="0.7 0.2 0.005" density="100" group="3"/>'
        )
    else:
        peg_col = (
            f'      <geom name="{PEG_COLLISION}" type="box" '
            f'pos="0 0 {peg_z0:.6f}" '
            f'size="{peg_col_spec["peg_half_width_m"]:.6f} {peg_col_spec["peg_half_depth_m"]:.6f} {peg_half_len:.6f}" '
            f'friction="0.7 0.2 0.005" density="100" group="3"/>'
        )

    # Socket opening/walls from SOCKET family collision (critical for mismatch negatives).
    hole_half = float(sock_col["hole_half_xy_m"])
    wall_t = float(sock_col["wall_thickness_m"])
    wall_h = float(sock_col["wall_half_height_m"])
    wall_z = float(sock_col["wall_center_z_m"])
    bottom_z = float(sock_col["bottom_center_z_m"])
    bottom_half_h = float(sock_col["bottom_half_height_m"])
    site_z = float(sock_col["socket_site_z_m"])
    base_half = float(sock_col["base_half_xy_m"])
    base_h = float(sock_col["base_half_height_m"])

    walls = []
    # pos/neg y walls
    for name, ypos in (("wall_pos_y", hole_half + wall_t), ("wall_neg_y", -(hole_half + wall_t))):
        walls.append(
            f'      <geom name="{name}" type="box" pos="0 {ypos:.6f} {wall_z:.6f}" '
            f'size="{hole_half + wall_t:.6f} {wall_t:.6f} {wall_h:.6f}" '
            f'friction="0.8 0.02 0.002" density="100" group="3"/>'
        )
    for name, xpos in (("wall_pos_x", hole_half + wall_t), ("wall_neg_x", -(hole_half + wall_t))):
        walls.append(
            f'      <geom name="{name}" type="box" pos="{xpos:.6f} 0 {wall_z:.6f}" '
            f'size="{wall_t:.6f} {hole_half:.6f} {wall_h:.6f}" '
            f'friction="0.8 0.02 0.002" density="100" group="3"/>'
        )
    walls_xml = "\n".join(walls)

    comment = (
        f"family={spec.family_id}; socket={sock.family_id}; "
        f"scale={scale}; collision_from_measured_mesh_not_copied_8mm"
    )
    return f"""<mujoco model="geometry_smoke_{spec.family_id}">
  <!-- {comment} -->
  <compiler angle="radian" meshdir="/" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>
  <asset>
    <mesh name="peg_mesh" file="{_escape(peg_mesh)}" scale="{scale} {scale} {scale}"/>
    <mesh name="socket_mesh" file="{_escape(sock_mesh)}" scale="{scale} {scale} {scale}"/>
    <material name="peg_mat" rgba="0.86 0.73 0.39 1"/>
    <material name="socket_mat" rgba="0.22 0.28 0.34 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.05" rgba="0.8 0.8 0.8 1" contype="1" conaffinity="1"/>

    <body name="{SOCKET_BODY}" pos="0 0 0.05">
      <freejoint name="{SOCKET_JOINT}"/>
      <geom name="{SOCKET_VISUAL}" type="mesh" mesh="socket_mesh" material="socket_mat"
            contype="0" conaffinity="0" density="0" group="1"/>
      <geom name="{SOCKET_BASE}" type="box" pos="0 0 {base_h:.6f}"
            size="{base_half:.6f} {base_half:.6f} {base_h:.6f}"
            friction="0.8 0.02 0.002" density="100" mass="0.05" group="3"/>
{walls_xml}
      <geom name="{INSERT_BOTTOM}" type="box" pos="0 0 {bottom_z:.6f}"
            size="{hole_half:.6f} {hole_half:.6f} {bottom_half_h:.6f}"
            friction="0.8 0.02 0.002" density="100" mass="0.01" rgba="1 0 0 0.3" group="3"/>
      <site name="{SOCKET_SITE}" pos="0 0 {site_z:.6f}" size="0.005" rgba="0 0 1 0.4"/>
    </body>

    <body name="{PEG_BODY}" pos="0 0 0.25">
      <freejoint name="{PEG_JOINT}"/>
      <geom name="{PEG_VISUAL}" type="mesh" mesh="peg_mesh" material="peg_mat"
            contype="0" conaffinity="0" density="0" group="1"/>
{peg_col}
      <site name="{PEG_TIP_SITE}" pos="0 0 {tip_z:.6f}" size="0.004" rgba="0 1 0 0.4"/>
      <site name="{PEG_GRASP_SITE}" pos="0 0 {grasp_z:.6f}" size="0.005" rgba="1 0 0 0.4"/>
    </body>
  </worldbody>
</mujoco>
"""


def write_family_xml(
    spec: GeometryFamilySpec,
    out_dir: Path,
    *,
    mismatch_socket: GeometryFamilySpec | None = None,
    tag: str = "matched",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fid = spec.family_id if mismatch_socket is None else f"{spec.family_id}__vs__{mismatch_socket.family_id}"
    path = out_dir / f"{fid}__{tag}.xml"
    path.write_text(build_family_xml(spec, mismatch_socket=mismatch_socket), encoding="utf-8")
    return path


def derive_collision_from_spans(spec: GeometryFamilySpec) -> dict[str, Any]:
    """Collision from measured mesh spans * asset_scale (not copied from 8mm numbers).

    Insertion tip = lower end of the collision primitive (not full visual mesh top).
    Official 8mm tip_site z=0.0795 is a different body-frame convention; we record the
    fraction for documentation but use geometric insert-end for smoke primitives so
    tip-at-socket-site does not bury the whole cylinder into the base.
    """
    s = float(spec.asset_scale)
    peg_span = [float(x) * s for x in spec.peg_mesh_span_xyz_m]
    sock_span = [float(x) * s for x in spec.socket_mesh_span_xyz_m]
    peg_len = peg_span[2]
    peg_half_len = 0.40 * peg_len
    peg_z0 = 0.55 * peg_len
    tip_z = peg_z0 - peg_half_len  # insert end
    grasp_z = peg_z0 + 0.25 * peg_half_len
    official_tip_frac = 0.0795 / 0.225

    if spec.section == "round":
        peg_radius = 0.5 * float(np_mean(peg_span[0], peg_span[1]))
        if spec.hole_diameter_m is not None:
            hole_r = 0.5 * float(spec.hole_diameter_m) * s
        else:
            hole_r = peg_radius * 1.02
        hole_half = hole_r
    else:
        peg_radius = None
        if spec.hole_width_m is not None:
            hole_half = 0.5 * float(spec.hole_width_m) * s
        else:
            hole_half = max(0.5 * peg_span[0], 0.5 * peg_span[1]) * 1.02

    wall_t = max(0.004, 0.15 * hole_half)
    wall_h = 0.5 * max(sock_span[2] * 0.7, 0.04)
    # Stack along +z: base plate, then insert pad, then socket_site (hole mouth).
    # Critical: base TOP must stay strictly below socket_site, else centered tip hits base.
    # Stack: base plate → insert pad → socket_site (mouth). Base top MUST be < site.
    base_h = 0.003
    base_top = 2.0 * base_h
    bottom_half_h = 0.001
    bottom_z = base_top + bottom_half_h + 0.001
    bottom_top = bottom_z + bottom_half_h
    site_z = bottom_top + 0.006
    wall_z = site_z + wall_h
    base_half = max(sock_span[0], sock_span[1]) * 0.55
    # Leave an open hole through the base plate (same half as insert pad).
    base_hole_half = hole_half

    out: dict[str, Any] = {
        "source": "measured_mesh_span_times_asset_scale_plus_yaml_hole_when_available",
        "tip_placement_source": "collision_lower_end_insert_tip",
        "official_8mm_tip_fraction_of_scaled_length": official_tip_frac,
        "peg_half_length_m": peg_half_len,
        "peg_collision_center_z_m": peg_z0,
        "peg_tip_site_z_m": tip_z,
        "peg_grasp_site_z_m": grasp_z,
        "hole_half_xy_m": hole_half,
        "wall_thickness_m": wall_t,
        "wall_half_height_m": wall_h,
        "wall_center_z_m": wall_z,
        "bottom_center_z_m": bottom_z,
        "bottom_half_height_m": bottom_half_h,
        "socket_site_z_m": site_z,
        "base_half_xy_m": base_half,
        "base_half_height_m": base_h,
        "base_top_z_m": base_top,
        "base_hole_half_xy_m": base_hole_half,
        "peg_visual_span_xyz_m": peg_span,
        "socket_visual_span_xyz_m": sock_span,
        "tip_is_collision_lower_end": True,
        "insertion_axis_body": "+z_from_tip_through_peg",
        "assert_base_top_below_site": base_top < site_z - 1e-4,
    }
    if spec.section == "round":
        out["peg_radius_m"] = peg_radius
    else:
        out["peg_half_width_m"] = 0.5 * peg_span[0]
        out["peg_half_depth_m"] = 0.5 * peg_span[1]
    return out


def np_mean(a: float, b: float) -> float:
    return 0.5 * (float(a) + float(b))
