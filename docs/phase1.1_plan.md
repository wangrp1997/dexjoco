# OIG 支线：Object-frame Interaction Grasp（通用抓取框架）

在 dexjoco 上验证的是特例；通用抓取收成 **「物体系 Interaction 原型 + 在线几何求解」** 框架。

---

## 名字（对外）

**OIG：Object-frame Interaction Grasp**  
或 slogan：**Many Demos Distill, One Skill Deploy**

核心：**offline 多 demo 归纳相对几何；online 只带少量固定原型，不查轨迹库。**

---

## 通用抽象（任务无关）

| 层 | 内容 |
|----|------|
| **表示** | 物体系 interaction mesh：N_h 手点 + N_o 物面点 → Laplacian δ |
| **技能单元** | 一个 **Grasp Prototype** = {δ*, 拓扑, 可选接触约束} |
| **推理输入** | 当前物体 mesh/body 系、T_world_obj、机器人模型 |
| **推理输出** | q_grasp（+ 可选 lift clip） |

物体对称与否、单臂双臂，只改 N_h、物体 mesh、约束，**不改框架**。

---

## 离线：Demo → Prototype Library

```
K 条 demo（同任务、同语义抓法）
  → replay / mocap / sim
  → 每条抽 grasp 帧，转到物体系
  → Laplacian / 顶点聚类
  → 1～M 个 Prototype（M 通常很小）
```

**归纳规则（通用）：**

1. **单模态**（抓取语义唯一）→ M=1，中位/代表帧  
2. **多稳定朝向 / 多抓法** → K-means 得 M=2~4，每簇一条 δ*  
3. **新物体** → 重新跑离线，**不**把旧物体 δ* 硬迁移  

100 条是 **估计 prototype 的样本量**，不是 online 的检索库。

---

## 在线：任意初始位姿

```
感知/仿真 → T_world_obj +（可选）稳定朝向 id
       ↓
选 Prototype（M=1 则跳过；M>1 用朝向/视觉/几何规则）
       ↓
Laplacian IK（pyroki/holosoma 式）→ q_init
       ↓
Contact repair + 物理验稳（spider 式）
       ↓
开环 grasp →（可选）跟 lift clip
```

**泛化来源：**

- 位姿变 → 物体系 δ* 固定，**IK 现算** world 构型  
- 不对称 → body 系随物体转，**不是** world 构型模板  
- 多抓法 → **多 prototype**，不是一条 median 糊掉  

**一阶段可不学网络**；M>1 时 prototype 选择可用轻量规则或小网络，仍 **不 replay demo 轨迹**。

---

## 和 dexjoco 任务的关系

| dexjoco 现状 | 通用版 |
|--------------|--------|
| tray/peg 各 1 个 δ* | 每物体/每抓法语义 1~M 个 Prototype |
| 对称、语义单一 | 不对称 → 先验定义 body 系 + 检查聚类是否单模态 |
| MuJoCo contact | 可换 sim / 真机 contact 验稳 |
| 双臂 Allegro | N_h、URDF 可配置 |

dexjoco = **OIG 在 industrireal 双物体上的第一个 benchmark**。

---

## 研究贡献（可写进 paper）

1. **范式**：many demos offline distill → few prototypes online deploy（对比 demo retrieval / E2E policy）  
2. **表示**：物体系 interaction mesh 作 **跨位姿** 抓取目标（对比 world 轨迹模仿）  
3. **扩展**：单 prototype → 小库 + 选择，覆盖不对称与多稳定朝向  
4. **验证阶梯**：dexjoco 对称物 → 单臂不对称物 → 新物体 re-distill  

---

## 实施路线（通用，不只 dexjoco）

| 阶段 | 内容 |
|------|------|
| **A** | dexjoco：distill δ* → IK → 开环 grasp（MVP） |
| **B** | 加 **prototype 质量指标**（物体系方差、多模态检测） |
| **C** | 换 **不对称单物体**（M=1 或 M>1 自动聚类） |
| **D** | **新物体协议**：少量 demo → distill → 零改动 online 栈 |
| **E** | lift / 操作扩展（δ_lift* clip，仍物体系） |

---

## 数据集与 Benchmark 对标（支线调研）

OIG **不依赖大规模 end-to-end 训练数据**；offline 需少量同语义 demo（10–100 条/物体），评测需 **跨位姿 / 跨物体 / 验 contact** 的基准。下表按用途分，不必全用。

### 常用数据（按角色）

| 类型 | 代表数据集 | 和 OIG 的关系 |
|------|------------|---------------|
| **灵巧手–物 mocap / HOI** | [GRAB](https://grab.is.tue.mpg.de/)、[DexYCB](https://dex-ycb.github.io/)、[OakInk](https://oakink.net/)、[HOI4D](https://hoi4d.github.io/)、[ARCTIC](https://arctic.is.tue.mpg.de/) | 人手交互轨迹；可转 robot hand retarget，作 **offline distill 原料** 或 qualitative 对比 |
| **灵巧抓取合成/仿真** | [DexGraspNet](https://dexgraspnet.github.io/)、[MultiGripperGrasp](https://multigrippergrasp.github.io/)、[BODex](https://pku-epic.github.io/BODex/) | 多指 grasp 姿态与成功率；对标 **几何/优化抓取**，测 pose 泛化 |
| **单物体多 demo（自采）** | dexjoco 100 ep zarr、LeRobot 录制的 perfect demo | **主战场**：many → one prototype，不对外部大数据做 supervised training |
| **真机灵巧操作** | [ALOHA / Mobile ALOHA](https://mobile-aloha.github.io/)、[DROID](https://droid-dataset.github.io/)（以夹爪为主）、各 lab Allegro/Shadow 小集 | 验证 **sim→real**；OIG 一阶段可先在 sim contact 验稳 |
| **mesh / 物体库** | [YCB](https://www.ycbbenchmarks.com/)、[Objaverse](https://objaverse.allenai.org/)（需筛选） | 新物体 **re-distill** 协议（阶段 D）；不直接当 grasp 标签库 |

> **说明**：GRAB / DexYCB 等体量很大，OIG 用法是 **子集 + 物体系归纳**，不是「全量预训练一个 policy」。

### Benchmark 阶梯（建议验证顺序）

| 层级 | Benchmark | 测什么 |
|------|-----------|--------|
| **L0** | dexjoco `bimanual_assembly`（IndustReal tray/peg） | 双物体、对称物、sim contact；**当前 MVP** |
| **L1** | DexGraspNet / BODex 子集（单 Allegro/Shadow + YCB 物体） | 单物体、多初始位姿、**grasp success / 不掉落** |
| **L2** | 自选 **不对称物体**（带稳定多朝向，5–10 物体 × 50 demo） | prototype 单模态 vs 小库 M>1、朝向选择 |
| **L3** | **新物体 hold-out**（训练未见物体，仅 20 demo re-distill） | 「新物体协议」：offline 重 distill、online 栈不变 |
| **L4**（可选） | GRAB / DexYCB 转 retarget 的 sim 复现 | 与 **human-demo retarget** 路线比质量，非必须 |

**统一指标（建议报告）**：grasp success、contact 连续帧、lift 成功率、物体系 Laplacian 误差、**所需 demo 条数**、推理是否查轨迹库（OIG 应为 **否**）。

### 算法对标（按范式分组）

| 范式 | 代表方法 | 和 OIG 的差异（写 paper 时用） |
|------|----------|-------------------------------|
| **物体系 interaction + 优化** | TopoRetarget / [holosoma](https://github.com/amazon-far/holosoma) / [pyroki](https://github.com/chungmin99/pyroki) hand retarget | 最接近；通常 **逐 demo  retarget**，OIG 强调 **many→few prototype、online 不查库** |
| **接触引导 / 物理验稳** | [SPIDER](https://spider-mjx.github.io/)（mjwp contact） | 可作 OIG 的 repair / rollout 模块；对比「仅 IK 无 contact 修」 |
| **学习式灵巧抓取** | DexGrasp-Anything、UniDexGrasp、SceneGrasp++ | 大数据训练、隐式表示；对比 **小 demo 量 + 显式 δ\*** |
| **模仿 / 检索** | ACT、Diffusion Policy、**最近邻 demo replay** | 轨迹级模仿；OIG 对比 **不 replay 100 条、只 deploy prototype** |
| **平行夹爪 grasp（基线）** | GraspNet / Contact-GraspNet | 非灵巧；作 **「非 interaction-mesh 基线」**，说明多指必要性 |
| **全局 motion retarget** | OmniRetarget | 全身 loco-manipulation；OIG 聚焦 **桌面灵巧 grasp 技能单元** |

**支线任务产出（调研一节即可）**：选定 L1–L2 各 1 个外部 benchmark + 2–3 个对标算法，填同一套指标表；dexjoco 作 L0 anchor，不强行刷全大数据 leaderboard。

---

## 真机落点：教一次就会抓（主故事）

**不必和 DexGraspNet 等刷榜路线硬比。** 我们的主叙事是：

> **One Video Teach, Object-frame Grasp on Real Robot**  
> 人教一次 → 机器人学会抓 → **同一物体、不同摆放/接近** 仍能抓起。

### 和上文 Benchmark 的关系

| | 支线调研（上一节） | **主落点（本节）** |
|--|-------------------|-------------------|
| 目的 | 了解领域数据与算法地图 | **真机可演示的系统** |
| 对比对象 | 可选外部 benchmark | **不 replay 视频轨迹、不查 demo 库** |
| 成功标准 | leaderboard / sim 指标 | **同物异位姿 grasp success（真机）** |
| 数据 | 大数据集子集 | **1 段人类抓取视频**（或极少量） |

### 流程（视频 → 真机）

```
人拍一段抓取视频
  → 手/物体估计（HOI、mesh、位姿）
  → 抽 grasp 帧，建物体系 interaction mesh / δ*
  → retarget 到 Allegro（GeoRT / pyroki / holosoma 式）
  → 真机：当前物体位姿 + δ* → IK + contact 验稳 → 开环 grasp
```

**视频学习调研**（GRAB、DexYCB、hand-object from video 等）用在 **offline 提取拓扑与相对几何**，不是训一个 end-to-end 大 policy。

### dexjoco 100 条 zarr 的角色

- **sim 脚手架**：把 sidecar → distill → IK → contact 管线跑通、调参。
- **真机目标**：100 条 eventually 换成 **1 段视频 → 1 个 Grasp Prototype**（many distill 是 sim 里求稳，不是最终产品形态）。

### 最低验收（真机）

- 固定或少量物体，桌面 **随机位姿 / 不同接近方向**。
- **一次视频 teach** 后，多次 trial grasp 成功率可报告。
- **不**要求 joint 角复现视频；要求 **物体系抓取关系** 成立 + contact 验稳。

### 和 OIG slogan 的统一

- sim 阶段：**Many Demos Distill, One Skill Deploy**（100 ep 压噪声）。
- 真机阶段：**One Video Teach, One Prototype Deploy**（教一次，online 仍不查轨迹库）。

---

## 假设与限制

**当前 pipeline 里较硬的假设：**
- 物体 **mesh / body 系** 已知（或 sim 里精确可读）→ 物体系 δ*、表面采样、contact snap 都靠这个
- 机器人 **URDF + 碰撞几何** 已知 → IK、contact 检测
- grasp **语义单一**（同一种抓法）
- 桌面场景、物体可跟踪位姿（sim 或标定+感知）

**真机「教一次」会松一点，但不会全无：**
- mesh 可来自 **一次性扫描 / CAD / 视觉重建**，不必每帧 perfect，但要有 **大致形状 + 稳定 body 系**
- 位姿靠 **ArUco / 6D pose / 手眼标定**，不是 magic
- contact 真机用 **指尖力/电流/触觉**，不必靠 mesh 精确碰撞（sim 里才 heavily 依赖 mesh）

---

**一句话**

通用抓取不是「再训一个大 policy」，而是：**用 demo 离线提炼少量物体系 interaction 原型，online 靠几何 IK + contact 适配任意位姿**；dexjoco sim 是脚手架，**真机落点是同一物体不同摆放能抓**；视频 teach 是数据入口，不是和大数据 grasp 榜同赛道竞技。