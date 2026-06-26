"""Export interaction mesh (hand + object + graph edges) for 3D viewing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from interaction_retarget.constants import INDUSTREAL_MESH_SCALE
from interaction_retarget.mesh.sampling import load_object_mesh


def object_mesh_edge_segments(mesh_path: Path, *, max_edges: int = 4000) -> np.ndarray:
    """Object-frame mesh wireframe segments (E, 2, 3) for background context."""
    mesh = load_object_mesh(mesh_path, scale=INDUSTREAL_MESH_SCALE)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if edges.shape[0] > max_edges:
        rng = np.random.default_rng(0)
        pick = rng.choice(edges.shape[0], size=max_edges, replace=False)
        edges = edges[pick]
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    segs = np.stack([verts[edges[:, 0]], verts[edges[:, 1]]], axis=1)
    return segs


def object_mesh_edge_segments_world(
    mesh_path: Path,
    obj_pos: np.ndarray,
    obj_quat_wxyz: np.ndarray,
    *,
    max_edges: int = 4000,
) -> np.ndarray:
    from interaction_retarget.transforms import object_to_world

    segs_obj = object_mesh_edge_segments(mesh_path, max_edges=max_edges)
    flat = segs_obj.reshape(-1, 3)
    flat_w = object_to_world(flat, obj_pos, obj_quat_wxyz)
    return flat_w.reshape(segs_obj.shape)


def write_interaction_html(
    out_path: Path,
    *,
    hand_obj: np.ndarray,
    object_samples_obj: np.ndarray,
    edges: list[tuple[int, int]],
    title: str,
    contact_centers_obj: np.ndarray | None = None,
    mesh_segments_obj: np.ndarray | None = None,
) -> Path:
    """Write a self-contained interactive 3D HTML (Three.js, no server needed)."""
    hand = np.asarray(hand_obj, dtype=np.float64).tolist()
    obj_pts = np.asarray(object_samples_obj, dtype=np.float64).tolist()
    edge_list = [[int(a), int(b)] for a, b in edges]
    contacts = (
        np.asarray(contact_centers_obj, dtype=np.float64).tolist()
        if contact_centers_obj is not None and contact_centers_obj.size
        else []
    )
    mesh_segs = (
        np.asarray(mesh_segments_obj, dtype=np.float64).reshape(-1, 2, 3).tolist()
        if mesh_segments_obj is not None and mesh_segments_obj.size
        else []
    )
    payload = {
        "title": title,
        "hand": hand,
        "object_samples": obj_pts,
        "edges": edge_list,
        "contacts": contacts,
        "mesh_segments": mesh_segs,
        "hand_skeleton": [],
        "hand_mesh_segments": [],
        "table_quad": [],
        "world": False,
        "num_hand": len(hand),
    }

    html = _HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _hand_skeleton_segments(hand_pts: np.ndarray) -> list[list[list[float]]]:
    from interaction_retarget.constants import HAND_SKELETON_EDGES

    segs: list[list[list[float]]] = []
    for i, j in HAND_SKELETON_EDGES:
        segs.append([hand_pts[i].tolist(), hand_pts[j].tolist()])
    return segs


def write_world_grasp_scene_html(
    out_path: Path,
    *,
    hand_world: np.ndarray,
    object_samples_world: np.ndarray,
    contact_centers_world: np.ndarray,
    edges: list[tuple[int, int]],
    mesh_segments_world: np.ndarray,
    hand_mesh_segments_world: np.ndarray,
    table_quad_world: np.ndarray,
    title: str,
) -> Path:
    """World-frame grasp snapshot: hand collision mesh + object + table plane."""
    hand = np.asarray(hand_world, dtype=np.float64).tolist()
    obj_pts = np.asarray(object_samples_world, dtype=np.float64).tolist()
    contacts = np.asarray(contact_centers_world, dtype=np.float64).tolist()
    mesh_segs = np.asarray(mesh_segments_world, dtype=np.float64).reshape(-1, 2, 3).tolist()
    hand_mesh = np.asarray(hand_mesh_segments_world, dtype=np.float64).reshape(-1, 2, 3).tolist()
    table = np.asarray(table_quad_world, dtype=np.float64).tolist()
    payload = {
        "title": title,
        "hand": hand,
        "object_samples": obj_pts,
        "edges": [[int(a), int(b)] for a, b in edges],
        "contacts": contacts,
        "mesh_segments": mesh_segs,
        "hand_mesh_segments": hand_mesh,
        "hand_skeleton": _hand_skeleton_segments(hand_world),
        "table_quad": table,
        "world": True,
        "num_hand": len(hand),
    }
    html = _HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def write_interaction_png(
    out_path: Path,
    *,
    hand_obj: np.ndarray,
    object_samples_obj: np.ndarray,
    edges: list[tuple[int, int]],
    title: str,
) -> Path:
    """Static 3D fallback image (matplotlib Agg, headless-safe)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    vertices = np.concatenate([hand_obj, object_samples_obj], axis=0)
    hand_n = hand_obj.shape[0]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(hand_obj[:, 0], hand_obj[:, 1], hand_obj[:, 2], c="crimson", s=40, label="hand (21)")
    ax.scatter(
        object_samples_obj[:, 0],
        object_samples_obj[:, 1],
        object_samples_obj[:, 2],
        c="steelblue",
        s=12,
        alpha=0.7,
        label="object (50)",
    )
    for i, j in edges:
        p0, p1 = vertices[i], vertices[j]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], c="gray", alpha=0.25, linewidth=0.6)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlabel("x (object frame)")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # equal aspect
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = float(np.max(maxs - mins) * 0.55 + 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Interaction Mesh</title>
  <style>
    body { margin: 0; overflow: hidden; font-family: sans-serif; background: #111; color: #eee; }
    #info {
      position: absolute; top: 8px; left: 8px; z-index: 1;
      background: rgba(0,0,0,0.55); padding: 8px 12px; border-radius: 6px; font-size: 13px;
      max-width: 420px; line-height: 1.4;
    }
  </style>
</head>
<body>
<div id="info"></div>
<script type="importmap">
  {"imports": {"three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
               "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DATA = __PAYLOAD__;
const world = !!DATA.world;
document.getElementById('info').innerHTML =
  `<b>${DATA.title}</b><br/>` +
  (DATA.world ? `World frame @ grasp · ` : `Object frame · `) +
  `Drag rotate · scroll zoom · right-drag pan<br/>` +
  `<span style="color:#f96">●</span> hand ${DATA.num_hand} &nbsp; ` +
  `<span style="color:#5ec8ff">●</span> object ${DATA.object_samples.length} &nbsp; ` +
  `<span style="color:#c084fc">—</span> topo ${DATA.edges.length}` +
  (DATA.contacts.length ? `<br/><span style="color:#ffb020">●</span> contact (debug, 不在 71 顶点里) ${DATA.contacts.length}` : '') +
  (DATA.mesh_segments.length ? `<br/><span style="color:#b0b8c8">—</span> object mesh wire` : '') +
  (DATA.world ? `<br/><span style="color:#8a8">▭</span> table plane` : '') +
  ((DATA.hand_mesh_segments || []).length ? `<br/><span style="color:#ff9933">—</span> hand collision mesh` : '');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x141820);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.001, 10);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const light = new THREE.DirectionalLight(0xffffff, 1.1);
light.position.set(1, 2, 3);
scene.add(light, new THREE.AmbientLight(0xffffff, 0.45));

function addSpheres(points, color, size) {
  const geo = new THREE.SphereGeometry(size, 10, 10);
  const mat = new THREE.MeshPhongMaterial({ color });
  points.forEach(p => {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(p[0], p[1], p[2]);
    scene.add(m);
  });
}

function addEdges(edges, allPts, color) {
  const positions = [];
  edges.forEach(([a,b]) => {
    const p0 = allPts[a], p1 = allPts[b];
    positions.push(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]);
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.58 });
  scene.add(new THREE.LineSegments(geo, mat));
}

function addMeshSegments(segs, color) {
  const positions = [];
  segs.forEach(seg => {
    positions.push(seg[0][0], seg[0][1], seg[0][2], seg[1][0], seg[1][1], seg[1][2]);
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.15 });
  scene.add(new THREE.LineSegments(geo, mat));
}

function addSegmentList(segs, color, opacity) {
  const positions = [];
  segs.forEach(seg => {
    positions.push(seg[0][0], seg[0][1], seg[0][2], seg[1][0], seg[1][1], seg[1][2]);
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  scene.add(new THREE.LineSegments(geo, mat));
}

function addTableQuad(quad) {
  if (!quad.length) return;
  const geo = new THREE.BufferGeometry();
  const positions = [];
  const idx = [0,1, 1,2, 2,3, 3,0];
  idx.forEach(i => {
    const p = quad[i];
    positions.push(p[0], p[1], p[2]);
  });
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const mat = new THREE.LineBasicMaterial({ color: 0x88aa88, transparent: true, opacity: 0.8 });
  scene.add(new THREE.LineSegments(geo, mat));
  const v = quad.map(p => new THREE.Vector3(p[0], p[1], p[2]));
  const meshGeo = new THREE.BufferGeometry().setFromPoints(v);
  meshGeo.setIndex([0,1,2, 0,2,3]);
  meshGeo.computeVertexNormals();
  const meshMat = new THREE.MeshPhongMaterial({ color: 0x445544, transparent: true, opacity: 0.25, side: THREE.DoubleSide });
  scene.add(new THREE.Mesh(meshGeo, meshMat));
}

const handR = world ? 0.006 : 0.004;
const objR = world ? 0.0035 : 0.0025;
const conR = world ? 0.0055 : 0.0035;
const allPts = DATA.hand.concat(DATA.object_samples);
addSpheres(DATA.hand, 0xff4455, handR);
addSpheres(DATA.object_samples, 0x5ec8ff, objR);
if (DATA.contacts.length) addSpheres(DATA.contacts, 0xffb020, conR);
addEdges(DATA.edges, allPts, 0xc084fc);
if (DATA.mesh_segments.length) addSegmentList(DATA.mesh_segments, 0xb0b8c8, world ? 0.7 : 0.4);
if (DATA.hand_mesh_segments.length) addSegmentList(DATA.hand_mesh_segments, 0xff9933, 0.9);
if (DATA.hand_skeleton.length) addSegmentList(DATA.hand_skeleton, 0xff8899, 0.75);
if (DATA.table_quad.length) addTableQuad(DATA.table_quad);

// frame object at origin
const box = new THREE.Box3();
allPts.forEach(p => box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2])));
const center = box.getCenter(new THREE.Vector3());
const size = box.getSize(new THREE.Vector3()).length();
camera.position.copy(center).add(new THREE.Vector3(size * 0.9, size * 0.6, size * 0.9));
controls.target.copy(center);

window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();
</script>
</body>
</html>
"""
