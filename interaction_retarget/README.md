# interaction_retarget（一阶段 · 已完成部分）

从 `bimanual_assembly` perfect demo（zarr）离线导出 **δ\* 原料**：物体系 hand–object interaction mesh。

完整计划见 [`docs/phase1_grasp_plan.md`](../docs/phase1_grasp_plan.md)。

## 目录结构

```
interaction_retarget/
  constants.py          # 任务常量、路径、阈值
  transforms.py         # 物体系 ↔ world
  laplacian.py          # Delaunay + Laplacian（核心表示）
  distill_grasp.py      # Step2：many ep → canonical δ*

  io/                   # 数据读写
    zarr_io.py          # replay.zarr
    sidecar.py          # 逐 ep sidecar 导出
    npz.py              # sidecar/canonical npz 加载

  sim/                  # MuJoCo 仿真与抓取检测
    replay.py
    contact.py
    grasp_timing.py
    hand_geom.py

  mesh/                 # 物体表面采样
    sampling.py

  vis/                  # 3D 可视化 HTML
    mesh.py
```

## 已完成

| 模块 | 作用 |
|------|------|
| `io/zarr_io.py` | 读取 `replay.zarr`（action / state） |
| `sim/replay.py` | sim replay + 每帧 kinematics / contact |
| `sim/contact.py` | 左–tray / 右–peg MuJoCo 接触检测 |
| `sim/grasp_timing.py` | grasp 稳定帧、lift 起点（规则） |
| `mesh/sampling.py` | 物体 mesh 接触加权采样（holosoma 思路） |
| `laplacian.py` | Delaunay 邻接 + Laplacian 坐标 |
| `io/sidecar.py` | 汇总导出 npz / meta / manifest |
| `distill_grasp.py` | 100 ep → canonical δ\*（tray/peg 各一条） |
| `vis/mesh.py` | 3D interaction mesh HTML 可视化 |

脚本：

- `scripts/build_interaction_sidecar.py` — 批量导出 sidecar
- `scripts/vis_interaction_mesh.py` — 从 sidecar 生成交互式 3D HTML

## 表示（对齐 TopoRetarget）

每只手–物体一对：

- **21 手点**：Allegro `palm + 4指×5 link`（物体系）
- **50 物点**：物体 mesh 表面采样（`N_o=50`）
- **71 顶点**：手 + 物 → Delaunay → Laplacian 坐标

接触验稳（spider 思路）单独用 **4 指尖 geom**，与 21 点 mesh 分开。

## 环境

```bash
cd ~/dexjoco
conda activate dexjoco
export MUJOCO_GL=egl
export PYTHONPATH=~/dexjoco:~/dexjoco/dexjoco
pip install trimesh   # 若未装
```

## 导出 sidecar

```bash
python scripts/build_interaction_sidecar.py \
  --zarr-root /mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly
# 默认写到 /mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly

# 调试
python scripts/build_interaction_sidecar.py ... --max-episodes 1
python scripts/build_interaction_sidecar.py ... --episodes 0 3 7
```

输出结构：

```
out-dir/
  manifest.json
  episode_000/
    interaction_sidecar.npz   # tray/peg 各一套 21+50+71
    meta.json                 # timing、body 名、grasp 帧
```

`npz` 主要字段（前缀 `tray_` / `peg_`）：

- `*_hand_points_obj` `(21, 3)`
- `*_object_samples_obj` `(50, 3)`
- `*_interaction_vertices_obj` `(71, 3)`
- `*_laplacian_coords` `(71, 3)`
- `*_adjacency` — 邻接表（padding 矩阵）

## 3D 可视化（无屏服务器）

**推荐 world 场景**（手骨架 + 桌面 + 缩放后 mesh）：

```bash
python scripts/vis_interaction_mesh.py \
  --sidecar-dir /mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly/episode_000 \
  --object both \
  --world-scene
```

生成 `vis_tray_world.html` / `vis_peg_world.html`。

物体系拓扑（不含桌面）：

```bash
python scripts/vis_interaction_mesh.py \
  --sidecar-dir .../episode_000 \
  --object both \
  --show-object-mesh
```

scp 到本机浏览器打开。

**读图**：
- 红=21 手点，亮蓝=50 物面采样，紫实线=拓扑边
- 银灰实线=物体 mesh 线框，橙实线=手 collision，黄=接触中心（debug）

## grasp timing 规则（约 30 Hz）

- **grasp**：连续 10 帧 contact + 指速 ≤ 0.02 + 物体仍在桌面（相对 reset 高度 ≤ 1.5 cm）
- **lift**：相对 **replay 初始 rest z** 抬高 ≥ 2 cm 且仍有 contact 的首帧（与 grasp 独立，可在 grasp 之后）

阈值见 `constants.py`。

## distill canonical δ*（Step 2）

```bash
python scripts/distill_grasp.py \
  --sidecar-dir /mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly

# 可选：去掉 manifest 里 fallback 的 episode
python scripts/distill_grasp.py --exclude-fallback
```

输出（与 sidecar 同目录）：

- `canonical_tray_grasp.npz` / `canonical_peg_grasp.npz`
- `canonical_*_grasp.json`（质量报告）
- `canonical_grasp_summary.json`

**归纳规则**：21 手点逐点中位；物面 50 点用全部 episode 接触中心 pooled 后加权重采样（固定 seed）；再 Delaunay → Laplacian。

## 未做（后续步骤）

1. ~~`distill_grasp.py` — 100 ep → canonical δ\*~~ ✅
2. `laplacian_ik.py` — δ\* → q_grasp
3. `grasp_repair.py` — contact 修 + 验稳
4. `validate_grasp_openloop.py` — random init 开环 grasp
5. lift clip + OpenTrack 式 tracking

## 参考

- **TopoRetarget**：21 手 + 50 物 interaction mesh
- **holosoma**：mesh 采样、Laplacian
- **spider**：指尖 contact 检测
- **pyroki**：后续 IK（未接入）
