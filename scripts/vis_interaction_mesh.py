#!/usr/bin/env python3
"""Visualize interaction mesh (21 hand + 50 object + graph edges) from sidecar npz.

Headless-friendly: writes interactive HTML (Three.js) and/or static PNG.
Download the HTML to your laptop and open in Chrome/Firefox to rotate/zoom.

Example:
  python scripts/vis_interaction_mesh.py \\
    --sidecar-dir /tmp/interaction_sidecar_21pt/episode_000 \\
    --object tray \\
    --out /tmp/tray_mesh.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from interaction_retarget.constants import LEFT_HAND_ROOT, PEG_MESH_PATH, RIGHT_HAND_ROOT, TRAY_MESH_PATH
from interaction_retarget.sim.hand_geom import hand_collision_segments_world
from interaction_retarget.sim.replay import make_assembly_env, raw_flat_to_dict, replay_episode
from interaction_retarget.transforms import object_to_world
from interaction_retarget.io.npz import load_interaction_npz
from interaction_retarget.vis.mesh import (
    object_mesh_edge_segments,
    object_mesh_edge_segments_world,
    write_interaction_html,
    write_interaction_png,
    write_world_grasp_scene_html,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sidecar-dir",
        type=Path,
        required=True,
        help="Episode dir containing interaction_sidecar.npz",
    )
    p.add_argument(
        "--object",
        choices=("tray", "peg", "both"),
        default="both",
        help="Which hand-object interaction to visualize",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (.html or .png). Default: <sidecar-dir>/vis_<object>.html",
    )
    p.add_argument("--png", action="store_true", help="Also write a static PNG alongside HTML")
    p.add_argument(
        "--png-only",
        action="store_true",
        help="Skip HTML; write PNG only (needs matplotlib)",
    )
    p.add_argument(
        "--show-object-mesh",
        action="store_true",
        help="Overlay object mesh wireframe",
    )
    p.add_argument(
        "--world-scene",
        action="store_true",
        help="World-frame grasp scene: hand skeleton + table + scaled mesh (recommended)",
    )
    return p.parse_args()


def _default_out(sidecar_dir: Path, prefix: str, png_only: bool, world: bool) -> Path:
    ext = ".png" if png_only else ".html"
    tag = f"vis_{prefix}_world" if world else f"vis_{prefix}"
    return sidecar_dir / f"{tag}{ext}"


def _render_one(
    npz_path: Path,
    prefix: str,
    out: Path,
    *,
    png: bool,
    png_only: bool,
    show_mesh: bool,
) -> None:
    snap = load_interaction_npz(npz_path, prefix=prefix)
    mesh_path = TRAY_MESH_PATH if prefix == "tray" else PEG_MESH_PATH
    mesh_segs = object_mesh_edge_segments(mesh_path) if show_mesh else None
    title = f"{prefix} interaction mesh @ grasp frame {snap['grasp_frame']}"

    if not png_only:
        html_path = out if out.suffix == ".html" else out.with_suffix(".html")
        write_interaction_html(
            html_path,
            hand_obj=snap["hand_obj"],
            object_samples_obj=snap["object_samples_obj"],
            edges=snap["edges"],
            title=title,
            contact_centers_obj=snap["contact_centers_obj"],
            mesh_segments_obj=mesh_segs,
        )
        print(f"[html] {html_path}")

    if png or png_only:
        png_path = out if out.suffix == ".png" else out.with_suffix(".png")
        try:
            write_interaction_png(
                png_path,
                hand_obj=snap["hand_obj"],
                object_samples_obj=snap["object_samples_obj"],
                edges=snap["edges"],
                title=title,
            )
            print(f"[png]  {png_path}")
        except ModuleNotFoundError:
            print("[png]  skipped (pip install matplotlib)")


def _hand_mesh_at_grasp_frame(
    meta: dict,
    zarr_path: Path,
    grasp_frame: int,
    *,
    side: str,
) -> np.ndarray:
    from dexjoco.tasks import CONFIG_MAPPING
    from dexjoco.tasks.state_restorers import restore_initial_state
    from interaction_retarget.io.zarr_io import load_zarr_episode

    actions, _, initial_state = load_zarr_episode(zarr_path)
    env = make_assembly_env(seed=int(meta["episode_index"]))
    raw = env.unwrapped
    model = raw._model
    config = CONFIG_MAPPING["bimanual_assembly"]()
    root = LEFT_HAND_ROOT if side == "left" else RIGHT_HAND_ROOT
    try:
        env.reset()
        if initial_state is not None:
            restore_initial_state(env, "bimanual_assembly", config, initial_state)
        for i, action in enumerate(actions):
            raw.step(raw_flat_to_dict(action))
            if i == grasp_frame:
                return hand_collision_segments_world(model, raw._data, root)
        return np.zeros((0, 2, 3), dtype=np.float64)
    finally:
        env.close()


def _render_world_scene(
    sidecar_dir: Path,
    npz_path: Path,
    prefix: str,
    out: Path,
) -> None:
    from interaction_retarget.sim.replay import replay_episode
    from interaction_retarget.io.zarr_io import load_zarr_episode

    meta = json.loads((sidecar_dir / "meta.json").read_text(encoding="utf-8"))
    snap = load_interaction_npz(npz_path, prefix=prefix)
    zarr_path = Path(meta["zarr_path"])
    actions, _, initial_state = load_zarr_episode(zarr_path)
    trace = replay_episode(
        actions,
        seed=int(meta["episode_index"]),
        initial_state=initial_state,
    )
    grasp_frame = int(snap["grasp_frame"])
    step = trace.steps[grasp_frame]
    mesh_path = TRAY_MESH_PATH if prefix == "tray" else PEG_MESH_PATH

    if prefix == "tray":
        hand_w = step.left_hand_world
        obj_pos, obj_quat = step.tray_pos, step.tray_quat
        rest_z = float(trace.steps[0].tray_z)
        hand_side = "left"
    else:
        hand_w = step.right_hand_world
        obj_pos, obj_quat = step.peg_pos, step.peg_quat
        rest_z = float(trace.steps[0].peg_z)
        hand_side = "right"

    contacts_obj = np.asarray(snap.get("contact_centers_obj", np.zeros((0, 3))), dtype=np.float64)
    contacts_w = object_to_world(contacts_obj, obj_pos, obj_quat) if contacts_obj.size else np.zeros((0, 3))
    samples_w = object_to_world(snap["object_samples_obj"], obj_pos, obj_quat)
    mesh_segs_w = object_mesh_edge_segments_world(mesh_path, obj_pos, obj_quat)
    hand_mesh_w = _hand_mesh_at_grasp_frame(meta, zarr_path, grasp_frame, side=hand_side)
    table_z = rest_z - 0.008
    cx, cy = float(obj_pos[0]), float(obj_pos[1])
    half = 0.18
    table_quad = np.asarray(
        [
            [cx - half, cy - half, table_z],
            [cx + half, cy - half, table_z],
            [cx + half, cy + half, table_z],
            [cx - half, cy + half, table_z],
        ],
        dtype=np.float64,
    )

    html_path = out if out.suffix == ".html" else out.with_suffix(".html")
    write_world_grasp_scene_html(
        html_path,
        hand_world=hand_w,
        object_samples_world=samples_w,
        contact_centers_world=contacts_w,
        edges=snap["edges"],
        mesh_segments_world=mesh_segs_w,
        hand_mesh_segments_world=hand_mesh_w,
        table_quad_world=table_quad,
        title=f"{prefix} grasp scene (world) @ frame {grasp_frame}",
    )
    print(f"[world html] {html_path}")


def main() -> None:
    args = _parse_args()
    sidecar_dir = args.sidecar_dir.expanduser()
    npz_path = sidecar_dir / "interaction_sidecar.npz"
    if not npz_path.exists():
        raise SystemExit(f"Not found: {npz_path}")

    targets = ["tray", "peg"] if args.object == "both" else [args.object]
    for prefix in targets:
        if args.out is not None and len(targets) == 1:
            out = args.out
        else:
            out = _default_out(sidecar_dir, prefix, args.png_only, args.world_scene)
        if args.world_scene:
            if args.png or args.png_only:
                print("[png]  skipped in --world-scene mode")
            _render_world_scene(sidecar_dir, npz_path, prefix, out)
            continue
        _render_one(
            npz_path,
            prefix,
            out,
            png=args.png,
            png_only=args.png_only,
            show_mesh=args.show_object_mesh,
        )


if __name__ == "__main__":
    main()
