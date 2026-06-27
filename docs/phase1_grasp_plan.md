# 一阶段：双物体稳定抓取 + 提起

目标：random init 下 L 抓 tray、R 抓 peg，**一次抓稳**，再 **lift 到物体系目标**。插孔二阶段再做。

demo 只用于 **离线总结技能**，推理 **不查 demo 模板**。

---

## 核心表示

| 符号 | 含义 |
|------|------|
| **δ\*** | 物体系手–物拓扑：各指相对物体应落在哪里（pairwise / Laplacian） |
| **δ_lift\*** | grasp 稳定后的 lift clip（物体系，非 world 单点） |

一阶段 **不需要点云**；MuJoCo body/site + contact 即可。

---

## 五个 refs 借什么

| Repo | 借什么 | 一阶段 |
|------|--------|--------|
| **pyroki** | `examples/09_hand_retargeting.py`：pairwise/Laplacian cost + IK | 抓 |
| **holosoma** | `interaction_mesh_retargeter.py`：物体系 interaction mesh | 抓 |
| **spider** | `detect_contact.py` + mjwp contact guidance / 物理 rollout | 抓 + 验稳 |
| **OpenTrack** | clip ref tracking（`traj_handler` + PPO） | **仅 lift 段** |
| **DexGrasp-Anything** | — | 跳过 |

> 五个 repo **都不做「100 条轨迹取平均」**。δ\* 总结 **相对几何**；lift 用 **一条 canonical clip** 或 **clip 库**。

---

## 离线（100 条 perfect demo）

1. zarr replay → 每 ep 抽 grasp 稳定帧的 **物体系指–物相对几何**
2. 100 条对齐 → **canonical δ\***（pairwise 中位 / 聚类代表帧，不是 world 轨迹平均）
3. 100 条 lift 段 → 物体系 **canonical lift clip**（方差大则 clip 库）
4. 输出 sidecar / npz，不改 LeRobot 原数据

---

## 推理（任意物体位姿）

```
当前 T_world_obj + δ*  →  pyroki 式 IK  →  q_grasp
                              ↓
                    spider 式 contact 修 + sim 验稳
                              ↓
                         开环执行 grasp（不用 OpenTrack）
                              ↓
              T_world_obj × δ_lift*(t)  →  OpenTrack 式跟 lift
```

> reach 用 pyroki trajopt；grasp 构型用 pyroki 09；contact/抓稳用 dexjoco sim + spider，三块拼起来才是完整 answer。

- **不做 regrasp**
- **不查最近邻 demo**

---

## 工程落点（dexjoco 新建）

```
interaction_retarget/
  io/                     # sidecar 导出
  grasp/
    distill.py            # 100 ep → canonical δ*
    ik.py                 # δ* → q（参考 pyroki/holosoma）
    repair.py             # contact 修 + 验稳（参考 spider）
scripts/
  build_interaction_sidecar.py
  validate_grasp_openloop.py
```

demo 读取：**zarr + dexjoco env replay**（`scripts/replay_demos_zarr.py` 或同等逻辑）。

`resfit_dexjoco`（ForceVLA+残差 TD3）**一阶段不用**。

---

## 实施顺序

| # | 工作 | 完成标准 |
|---|------|----------|
| 1 | sidecar 导出 δ* | 100 ep 可读 |
| 2 | distill canonical δ* | tray / peg 各一条 |
| 3 | grasp/ik + grasp/repair | random init → q_grasp |
| 4 | 开环 grasp 验证 | L/R contact 不断、不掉物 |
| 5 | distill lift clip | 物体系 clip 就绪 |
| 6 | lift ref tracking | 提起达标 |

**MVP 截止：步骤 1–4**（双物体一次抓稳）。

---

## 命令环境（参考）

```bash
cd ~/dexjoco
conda activate dexjoco   # 或 lerobot
export MUJOCO_GL=egl
export PYTHONPATH=~/dexjoco:~/dexjoco/dexjoco
```
