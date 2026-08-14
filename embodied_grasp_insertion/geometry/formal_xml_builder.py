"""Build formal IndustReal peg/socket + arena XMLs for geometry families.

- round_8mm: reuses existing official XMLs / default arena (no overwrite).
- other families: scale official 8mm collision layout in XY by measured radii;
  keep official Z layout (all pegs share visual length 0.225 m).
- Does not enable collection or training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dexjoco.sim.envs.assembly_geometry import (
    DEFAULT_FAMILY_ID,
    XMLS_DIR,
    AssemblyGeometryNames,
    arena_xml_path,
    names_for_family,
    names_for_socket_instance,
)
from embodied_grasp_insertion.geometry.family_spec import GeometryFamilySpec

# Official round_8mm collision layout (do not invent new Z for formal assets).
_OFF_PEG_R = 0.01785
_OFF_PEG_Z0 = 0.081
_OFF_PEG_HALF_H = 0.0675
_OFF_PEG_UPPER_Z = 0.1935
_OFF_PEG_UPPER_HALF = 0.045
_OFF_TIP_Z = 0.0795
_OFF_GRASP_Z = 0.042
_OFF_PEG_POS = (-0.10, -0.16, 0.94)
_OFF_PEG_QUAT = (0.6964, -0.1228, 0.6964, 0.1228)

_OFF_HOLE_HALF = 0.018  # bottom_contact half-xy
_OFF_BASE_HALF = 0.09
_OFF_BASE_H = 0.00675
_OFF_WALL_INNER = 0.02025  # wall_pos_y.y - wall_pos_y.size_y
_OFF_WALL_T = 0.018
_OFF_WALL_H = 0.05625
_OFF_WALL_Z = 0.06975
_OFF_BOTTOM_Z = 0.0145
_OFF_BOTTOM_H = 0.001
_OFF_SITE_Z = 0.027
_OFF_SOCK_POS = (-0.10, 0.16, 0.92)

_MESH_REL = "./industreal/mesh/industreal_pegs"
_SCALE = 4.5


def _peg_xy_scale(spec: GeometryFamilySpec) -> tuple[float, float, float]:
    """Return (sx, sy, ref_r_or_half) for collision XY vs official round_8mm."""
    c = spec.collision
    if spec.section == "round":
        r = float(c["peg_radius_m"])
        s = r / _OFF_PEG_R
        return s, s, r
    hw = float(c["peg_half_width_m"])
    hd = float(c["peg_half_depth_m"])
    return hw / _OFF_PEG_R, hd / _OFF_PEG_R, max(hw, hd)


def _hole_xy_scale(spec: GeometryFamilySpec) -> float:
    return float(spec.collision["hole_half_xy_m"]) / _OFF_HOLE_HALF


def build_peg_asset_xml(spec: GeometryFamilySpec, names: AssemblyGeometryNames) -> str:
    sx, sy, _ = _peg_xy_scale(spec)
    mesh_file = f"{_MESH_REL}/{spec.peg_mesh}"
    if spec.section == "round":
        col = (
            f'      <geom name="{names.peg_collision}"\n'
            f'            type="cylinder"\n'
            f'            pos="0 0 {_OFF_PEG_Z0}"\n'
            f'            size="{_OFF_PEG_R * sx:.6f} {_OFF_PEG_HALF_H}"\n'
            f'            friction="0.7 0.2 0.005"\n'
            f'            solref="0.0015 1.0"\n'
            f'            solimp="0.995 0.999 0.0001"\n'
            f'            density="100"\n'
            f'            group="3" />\n'
            f'\n'
            f'      <geom name="{names.peg_collision_upper}"\n'
            f'            type="box"\n'
            f'            pos="0 0 {_OFF_PEG_UPPER_Z}"\n'
            f'            size="{_OFF_PEG_R * sx:.6f} {_OFF_PEG_R * sy:.6f} {_OFF_PEG_UPPER_HALF}"\n'
            f'            friction="0.7 0.2 0.005"\n'
            f'            solref="0.0015 1.0"\n'
            f'            solimp="0.995 0.999 0.0001"\n'
            f'            density="100"\n'
            f'            group="3" />'
        )
    else:
        col = (
            f'      <geom name="{names.peg_collision}"\n'
            f'            type="box"\n'
            f'            pos="0 0 {_OFF_PEG_Z0}"\n'
            f'            size="{_OFF_PEG_R * sx:.6f} {_OFF_PEG_R * sy:.6f} {_OFF_PEG_HALF_H}"\n'
            f'            friction="0.7 0.2 0.005"\n'
            f'            solref="0.0015 1.0"\n'
            f'            solimp="0.995 0.999 0.0001"\n'
            f'            density="100"\n'
            f'            group="3" />\n'
            f'\n'
            f'      <geom name="{names.peg_collision_upper}"\n'
            f'            type="box"\n'
            f'            pos="0 0 {_OFF_PEG_UPPER_Z}"\n'
            f'            size="{_OFF_PEG_R * sx:.6f} {_OFF_PEG_R * sy:.6f} {_OFF_PEG_UPPER_HALF}"\n'
            f'            friction="0.7 0.2 0.005"\n'
            f'            solref="0.0015 1.0"\n'
            f'            solimp="0.995 0.999 0.0001"\n'
            f'            density="100"\n'
            f'            group="3" />'
        )
    px, py, pz = _OFF_PEG_POS
    qw, qx, qy, qz = _OFF_PEG_QUAT
    return f"""<mujoco model="{names.peg_body}">
  <!-- formal asset; XY scaled from official 8mm layout; Z layout preserved -->
  <asset>
    <mesh name="{names.peg_mesh}"
          file="{mesh_file}"
          scale="{_SCALE} {_SCALE} {_SCALE}" />
    <material name="{names.peg_body}_mat"
              rgba="0.86 0.73 0.39 1"
              specular="0.8"
              shininess="0.7" />
  </asset>

  <worldbody>
    <body name="{names.peg_body}"
          pos="{px} {py} {pz}"
          quat="{qw} {qx} {qy} {qz}">
      <freejoint name="{names.peg_joint}" />

      <geom name="{names.peg_visual}"
            type="mesh"
            mesh="{names.peg_mesh}"
            material="{names.peg_body}_mat"
            contype="0"
            conaffinity="0"
            density="0"
            group="1" />

{col}

      <site name="{names.peg_grasp_site}"
            pos="0 0 {_OFF_GRASP_Z}"
            size="0.006 0.006 0.006"
            rgba="1 0 0 0" />
      <site name="{names.peg_tip_site}"
            pos="0 0 {_OFF_TIP_Z}"
            size="0.0045 0.0045 0.0045"
            rgba="0 1 0 0" />
    </body>

  </worldbody>
</mujoco>
"""


def build_socket_asset_xml(
    spec: GeometryFamilySpec,
    names: AssemblyGeometryNames,
    *,
    body_pos_xyz: tuple[float, float, float] | None = None,
) -> str:
    s = _hole_xy_scale(spec)
    # Keep official relation: site above base top (critical).
    assert _OFF_SITE_Z > 2.0 * _OFF_BASE_H
    # Mesh *file* from family spec; MuJoCo mesh *name* may be instance-suffixed.
    mesh_file = f"{_MESH_REL}/{spec.socket_mesh}"
    hole = _OFF_HOLE_HALF * s
    wall_inner = _OFF_WALL_INNER * s
    wall_t = _OFF_WALL_T * s
    wall_outer_y = wall_inner + wall_t
    # wall box half-sizes scaled in XY; Z unchanged
    wall_size_long = (_OFF_WALL_INNER + _OFF_WALL_T) * s  # 0.03825 * s for pos_y wall x-extent
    base_half = _OFF_BASE_HALF * max(s, 1.0)  # base at least covers walls
    # bottom pad tracks hole opening
    bottom_half = hole
    px, py, pz = body_pos_xyz if body_pos_xyz is not None else _OFF_SOCK_POS
    return f"""<mujoco model="{names.socket_body}">
  <!-- formal asset; XY scaled from official 8mm; Z layout preserved (site above base) -->
  <asset>
    <mesh name="{names.socket_mesh}"
          file="{mesh_file}"
          scale="{_SCALE} {_SCALE} {_SCALE}" />
    <material name="{names.socket_body}_mat"
              rgba="0.22 0.28 0.34 1"
              specular="0.85"
              shininess="0.75" />
  </asset>

  <worldbody>
    <body name="{names.socket_body}" pos="{px} {py} {pz}">
      <freejoint name="{names.socket_joint}" />

      <geom name="{names.socket_visual}"
            type="mesh"
            mesh="{names.socket_mesh}"
            material="{names.socket_body}_mat"
            contype="0"
            conaffinity="0"
            density="0"
            group="1" />

      <geom name="{names.socket_base}"
            type="box"
            pos="0 0 {_OFF_BASE_H}"
            size="{base_half:.6f} {base_half:.6f} {_OFF_BASE_H}"
            friction="0.8 0.02 0.002"
            solref="0.0015 1.0"
            solimp="0.995 0.999 0.0001"
            density="100"
            mass="0.01"
            group="3" />

      <geom name="{names.socket_body}_wall_pos_y"
            type="box"
            pos="0 {wall_outer_y:.6f} {_OFF_WALL_Z}"
            size="{wall_size_long:.6f} {wall_t:.6f} {_OFF_WALL_H}"
            friction="0.8 0.02 0.002"
            solref="0.0015 1.0"
            solimp="0.995 0.999 0.0001"
            density="100"
            mass="0.01"
            group="3" />
      <geom name="{names.socket_body}_wall_neg_y"
            type="box"
            pos="0 {-wall_outer_y:.6f} {_OFF_WALL_Z}"
            size="{wall_size_long:.6f} {wall_t:.6f} {_OFF_WALL_H}"
            friction="0.8 0.02 0.002"
            solref="0.0015 1.0"
            solimp="0.995 0.999 0.0001"
            density="100"
            mass="0.01"
            group="3" />
      <geom name="{names.socket_body}_wall_pos_x"
            type="box"
            pos="{wall_outer_y:.6f} 0 {_OFF_WALL_Z}"
            size="{wall_t:.6f} {wall_size_long:.6f} {_OFF_WALL_H}"
            friction="0.8 0.02 0.002"
            solref="0.0015 1.0"
            solimp="0.995 0.999 0.0001"
            density="100"
            mass="0.01"
            group="3" />
      <geom name="{names.socket_body}_wall_neg_x"
            type="box"
            pos="{-wall_outer_y:.6f} 0 {_OFF_WALL_Z}"
            size="{wall_t:.6f} {wall_size_long:.6f} {_OFF_WALL_H}"
            friction="0.8 0.02 0.002"
            solref="0.0015 1.0"
            solimp="0.995 0.999 0.0001"
            density="100"
            mass="0.01"
            group="3" />

      <geom name="{names.socket_bottom}"
            type="box"
            pos="0 0 {_OFF_BOTTOM_Z}"
            size="{bottom_half:.6f} {bottom_half:.6f} {_OFF_BOTTOM_H}"
            friction="0.8 0.02 0.002"
            solref="0.0015 1.0"
            solimp="0.995 0.999 0.0001"
            density="100"
            mass="0.01"
            rgba="1 0 0 1"
            group="3" />

      <site name="{names.socket_site}"
            pos="0 0 {_OFF_SITE_Z}"
            size="0.006 0.006 0.006"
            rgba="0 0 1 0" />
    </body>
  </worldbody>
</mujoco>
"""


def build_arena_xml(names: AssemblyGeometryNames) -> str:
    """Arena variant: same as default, but peg/socket includes + sensor objnames."""
    # Read default arena and only swap the three geometry-related lines conceptually.
    return f"""<?xml version='1.0' encoding='utf-8'?>
<mujoco model="Arena_Allegro_Bimanual_Assembly_{names.family_id}">
  <!-- parameterized family={names.family_id}; default round_8mm arena left untouched -->
  <include file="panda_allegro_bimanual.xml" />
  <include file="{names.peg_asset_xml}" />
  <include file="{names.socket_asset_xml}" />

  <option timestep=".002" noslip_iterations="5" noslip_tolerance="0" />
  <statistic center="0.3 0 0.4" extent=".8" />
  <visual>
    <headlight diffuse=".4 .4 .4" ambient=".5 .5 .5" />
    <global azimuth="160" elevation="-20" offheight="2048" offwidth="2048" />
    <quality offsamples="8" />
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="32" height="512" />
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4" />
    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true" reflectance="0" />

    <texture builtin="gradient" height="256" rgb1=".9 .9 1." rgb2=".2 .3 .4" type="skybox" width="256"/>
    <texture file="./table_arena/textures/light-gray-floor-tile.png" type="2d" name="texplane"/>
    <material name="floorplane" reflectance="0.01" shininess="0.0" specular="0.0" texrepeat="2 2" texture="texplane" texuniform="true"/>
    <texture file="./table_arena/textures/steel-brushed.png" type="cube" name="tex-steel-brushed"/>
    <material name="table_legs_metal" reflectance="0.8" shininess="0.8" texrepeat="1 1" texture="tex-steel-brushed" />
    <texture file="./table_arena/textures/light-gray-plaster.png" type="2d" name="tex-light-gray-plaster"/>
    <material name="walls_mat" reflectance="0.0" shininess="0.1" specular="0.1" texrepeat="3 3" texture="tex-light-gray-plaster" texuniform="true" />
    <texture  name="textable" builtin="flat" height="512" width="512" rgb1="0.5 0.5 0.5" rgb2="0.5 0.5 0.5"/>
    <material name="table_mat" texture="textable" />

    <texture file="./table_arena/textures/desktop_random_textures/bamboo.png" type="2d" name="tex-bamboo"/>
    <material name="table_bamboo" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-bamboo"/>
    <texture file="./table_arena/textures/desktop_random_textures/blue-wood.png" type="2d" name="tex-blue-wood"/>
    <material name="table_blue-wood" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-blue-wood"/>
    <texture file="./table_arena/textures/desktop_random_textures/brass-ambra.png" type="2d" name="tex-brass-ambra"/>
    <material name="table_brass-ambra" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-brass-ambra"/>
    <texture file="./table_arena/textures/desktop_random_textures/ceramic.png" type="2d" name="tex-ceramic"/>
    <material name="table_ceramic" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-ceramic"/>
    <texture file="./table_arena/textures/desktop_random_textures/cream-plaster.png" type="2d" name="tex-cream-plaster"/>
    <material name="table_cream-plaster" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-cream-plaster"/>
    <texture file="./table_arena/textures/desktop_random_textures/dark_wood_planks_2.png" type="2d" name="tex-dark_wood_planks_2"/>
    <material name="table_dark_wood_planks_2" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-dark_wood_planks_2"/>
    <texture file="./table_arena/textures/desktop_random_textures/dark-wood.png" type="2d" name="tex-dark-wood"/>
    <material name="table_dark-wood" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-dark-wood"/>
    <texture file="./table_arena/textures/desktop_random_textures/gray-plaster.png" type="2d" name="tex-gray-plaster"/>
    <material name="table_gray-plaster" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-gray-plaster"/>
    <texture file="./table_arena/textures/desktop_random_textures/gray_wood_planks.png" type="2d" name="tex-gray_wood_planks"/>
    <material name="table_gray_wood_planks" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-gray_wood_planks"/>
    <texture file="./table_arena/textures/desktop_random_textures/light-wood.png" type="2d" name="tex-light-wood"/>
    <material name="table_light-wood" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-light-wood"/>
    <texture file="./table_arena/textures/desktop_random_textures/metal.png" type="2d" name="tex-metal"/>
    <material name="table_metal" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-metal"/>
    <texture file="./table_arena/textures/desktop_random_textures/pink-plaster.png" type="2d" name="tex-pink-plaster"/>
    <material name="table_pink-plaster" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-pink-plaster"/>
    <texture file="./table_arena/textures/desktop_random_textures/red-wood.png" type="2d" name="tex-red-wood"/>
    <material name="table_red-wood" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-red-wood"/>
    <texture file="./table_arena/textures/desktop_random_textures/steel-scratched.png" type="2d" name="tex-steel-scratched"/>
    <material name="table_steel-scratched" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-steel-scratched"/>
    <texture file="./table_arena/textures/desktop_random_textures/walnut_wood_grain.png" type="2d" name="tex-walnut_wood_grain"/>
    <material name="table_walnut_wood_grain" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-walnut_wood_grain"/>
    <texture file="./table_arena/textures/desktop_random_textures/warm_wood_grain_2.png" type="2d" name="tex-warm_wood_grain_2"/>
    <material name="table_warm_wood_grain_2" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-warm_wood_grain_2"/>
    <texture file="./table_arena/textures/desktop_random_textures/white-plaster.png" type="2d" name="tex-white-plaster"/>
    <material name="table_white-plaster" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-white-plaster"/>
    <texture file="./table_arena/textures/desktop_random_textures/wood_grain_1.png" type="2d" name="tex-wood_grain_1"/>
    <material name="table_wood_grain_1" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-wood_grain_1"/>
    <texture file="./table_arena/textures/desktop_random_textures/yellow-plaster.png" type="2d" name="tex-yellow-plaster"/>
    <material name="table_yellow-plaster" reflectance="0.0" shininess="0.0" specular="0.2" texrepeat="1 1" texture="tex-yellow-plaster"/>
  </asset>

  <worldbody>
    <geom name="floor" size="3 3 0.01" type="plane" material="floorplane"/>

    <geom pos="-1.25 2.25 1.5" quat="0.6532815 0.6532815 0.2705981 0.2705981"
          size="1.06 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_leftcorner_visual" material="walls_mat"/>
    <geom pos="-1.25 -2.25 1.5" quat="0.6532815 0.6532815 -0.2705981 -0.2705981"
          size="1.06 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_rightcorner_visual" material="walls_mat"/>
    <geom pos="1.25 3 1.5" quat="0.7071 0.7071 0 0"
          size="1.75 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_left_visual" material="walls_mat"/>
    <geom pos="1.25 -3 1.5" quat="0.7071 -0.7071 0 0"
          size="1.75 1.5 0.01" type="box" conaffinity="0" contype="0" group="1" name="wall_right_visual" material="walls_mat"/>
    <geom pos="-2 0 1.5" quat="0.5 0.5 0.5 0.5" size="1.5 1.5 0.01"
          type="box" conaffinity="0" contype="0" group="1" name="wall_rear_visual" material="walls_mat"/>
    <geom pos="3 0 1.5" quat="0.5 0.5 -0.5 -0.5" size="3 1.5 0.01"
          type="box" conaffinity="0" contype="0" group="1" name="wall_front_visual" material="walls_mat"/>

    <body name="table" pos="-0.15 0 0">
      <geom name="table_collision" type="box" size="0.4 0.8 0.03" pos="0 0 0.89" friction="1 0.005 0.0001" group="3"/>
      <geom name="table_visual" type="box" size="0.4 0.8 0.03" pos="0 0 0.89" contype="0" conaffinity="0" group="1" material="table_ceramic"/>
      <site name="table_top" pos="0 0 0.92" size="0.002 0.002 0.002" rgba="0 0 0 0"/>

      <geom pos="0.32 0.72 0.43" type="cylinder" size="0.04 0.43" contype="0" conaffinity="0" group="1" material="table_legs_metal"/>
      <geom pos="0.32 -0.72 0.43" type="cylinder" size="0.04 0.43" contype="0" conaffinity="0" group="1" material="table_legs_metal"/>
      <geom pos="-0.32 0.72 0.43" type="cylinder" size="0.04 0.43" contype="0" conaffinity="0" group="1" material="table_legs_metal"/>
      <geom pos="-0.32 -0.72 0.43" type="cylinder" size="0.04 0.43" contype="0" conaffinity="0" group="1" material="table_legs_metal"/>
    </body>

    <camera name="back" pos="-1.3 0.0 2.2" quat="0.6532815 0.2705981 -0.2705981 -0.6532815" fovy="45" />
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular=".3 .3 .3" />

    <body name="target_right" pos="-0.10 -0.16 1.02" quat="0 1 0 0" mocap="true">
      <geom name="target_right" type="box" size=".03 .03 .03" contype="0" conaffinity="0" rgba=".6 .3 .3 0" />
    </body>
    <body name="target_left" pos="-0.10 0.16 1.02" quat="0 1 0 0" mocap="true">
      <geom name="target_left" type="box" size=".03 .03 .03" contype="0" conaffinity="0" rgba=".3 .3 .6 0" />
    </body>
  </worldbody>

  <sensor>
    <framepos objtype="body" name="assembly_peg_pos" objname="{names.peg_body}" />
    <framepos objtype="body" name="assembly_socket_pos" objname="{names.socket_body}" />
  </sensor>
</mujoco>
"""


def build_dual_socket_arena_xml(
    primary: AssemblyGeometryNames,
    secondary: AssemblyGeometryNames,
) -> str:
    """Arena with one peg + two same-family sockets (distinct body/site names)."""
    base = build_arena_xml(primary)
    # Insert secondary socket include after primary socket include.
    needle = f'<include file="{primary.socket_asset_xml}" />'
    insert = (
        f'{needle}\n'
        f'  <include file="{secondary.socket_asset_xml}" />'
    )
    if needle not in base:
        raise RuntimeError("primary socket include not found in arena template")
    out = base.replace(needle, insert, 1)
    out = out.replace(
        f'Arena_Allegro_Bimanual_Assembly_{primary.family_id}',
        f'Arena_Allegro_Bimanual_Assembly_{primary.family_id}_dual',
        1,
    )
    out = out.replace(
        f"parameterized family={primary.family_id}; default round_8mm arena left untouched",
        f"S0.3b dual-socket family={primary.family_id}; default arena left untouched",
        1,
    )
    return out


def write_dual_socket_family_assets(
    spec: GeometryFamilySpec,
    *,
    secondary_key: str = "b",
    secondary_pos_xyz: tuple[float, float, float] = (-0.10, -0.02, 0.92),
    xmls_dir: Path | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Write secondary socket + dual-socket arena for S0.3b (same family, two holes)."""
    xmls = Path(xmls_dir) if xmls_dir is not None else XMLS_DIR
    primary = names_for_family(spec.family_id)
    secondary = names_for_socket_instance(spec.family_id, secondary_key)
    # Ensure primary family assets exist for non-default families.
    write_formal_family_assets(spec, xmls_dir=xmls, overwrite=False)

    sock_path = xmls / secondary.socket_asset_xml
    arena_path = xmls / f"arena_arm_hand_bimanual_assembly__{primary.family_id}__dual.xml"
    if sock_path.exists() and not overwrite:
        pass
    else:
        sock_path.write_text(
            build_socket_asset_xml(spec, secondary, body_pos_xyz=secondary_pos_xyz),
            encoding="utf-8",
        )
    if arena_path.exists() and not overwrite:
        pass
    else:
        arena_path.write_text(build_dual_socket_arena_xml(primary, secondary), encoding="utf-8")
    return {
        "family_id": primary.family_id,
        "primary_socket_site": primary.socket_site,
        "secondary_socket_site": secondary.socket_site,
        "secondary_instance_id": secondary.socket_site,
        "secondary_socket_xml": str(sock_path),
        "dual_arena_xml": str(arena_path),
        "secondary_pos_xyz": list(secondary_pos_xyz),
    }


def write_formal_family_assets(
    spec: GeometryFamilySpec,
    *,
    xmls_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    xmls = Path(xmls_dir) if xmls_dir is not None else XMLS_DIR
    names = names_for_family(spec.family_id)
    out: dict[str, Any] = {
        "family_id": names.family_id,
        "reused_official_8mm": names.is_default_8mm_round,
        "peg_xml": str(xmls / names.peg_asset_xml),
        "socket_xml": str(xmls / names.socket_asset_xml),
        "arena_xml": str(arena_xml_path(names.family_id, xmls_dir=xmls)),
    }
    if names.is_default_8mm_round:
        out["note"] = "default round_8mm assets/arena left unchanged"
        return out

    peg_path = xmls / names.peg_asset_xml
    sock_path = xmls / names.socket_asset_xml
    arena_path = arena_xml_path(names.family_id, xmls_dir=xmls)
    if peg_path.exists() and not overwrite:
        out["peg_skipped"] = True
    else:
        peg_path.write_text(build_peg_asset_xml(spec, names), encoding="utf-8")
        out["peg_written"] = True
    if sock_path.exists() and not overwrite:
        out["socket_skipped"] = True
    else:
        sock_path.write_text(build_socket_asset_xml(spec, names), encoding="utf-8")
        out["socket_written"] = True
    if arena_path.exists() and not overwrite:
        out["arena_skipped"] = True
    else:
        arena_path.write_text(build_arena_xml(names), encoding="utf-8")
        out["arena_written"] = True
    return out
