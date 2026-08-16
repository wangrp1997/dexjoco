# Search Log

## 2026-08-16：Active contact sequence / contact manifold

- 检索到 `Efficient Active Pose Estimation via Contact Manifold Exploration`（arXiv:2505.19215）。
- 只借用“固定 primitive contact sequence 是否增加状态信息”的问题，不迁移其控制算法。
- 本地转化为 `Active Contact Probe Information P0`；episode-held-out sequence accuracy
  低于 static，方向已停止。

- 日期：2026-08-15
- 状态：`first_pass_complete`
- 范围：控制接口、接触状态、阶段可行域、跨几何装配语义。

## 查询

### P2：控制接口

- `contact-rich manipulation variable impedance action space insertion`
- `learned compliance stiffness damping action chunk contact manipulation`
- `passivity variable impedance connector insertion`

### P1/P3：接触状态与因果标签

- `assembly contact formation constraints force history generalized insertion`
- `contact information flow compliance contact mode detection assembly`
- `differentiable contact features uncertain pose insertion`

### P4/P5：阶段可行域

- `manipulation learned initiation set skill precondition recoverability assembly`
- `handoff state feasibility insertion success precondition`

### P6：跨几何语义

- `generalist assembly policies diverse geometries object environment contact representation`
- `geometric mating constraints robot assembly generalization`

## 论文与官方实现

### Variable Impedance Control in End-Effector Space（VICES，IROS 2019）

- 原始论文：`https://arxiv.org/abs/1906.08880`
- 官方项目：`https://stanfordvl.github.io/vices/`
- 官方实现线索：robosuite `vices_iros19` branch。
- 关键点：把 end-effector stiffness/damping 作为策略动作的一部分，而不是只预测
  pose；论文报告该动作空间在接触更强的任务上更有优势。
- 本地差异：DexJoCo 当前 policy action 只给 pose+finger position；机械臂底层
  operational-space controller 的 `pos_gains`、`ori_gains` 和 damping 固定。
- 保留用途：支持“动作接口 P0”，不支持直接启动 RL。

### Variable Impedance Skill Learning for Contact-Rich Manipulation（RA-L 2022）

- 论文/实现：`https://github.com/yquantao/learning_impedance_actions`
- 关键点：skill action 同时包含 Cartesian motion 与 stiffness，并在多种 peg shape
  上做 insertion adaptation。
- 风险：使用 FSM、spiral alignment 和预定义阶段收集数据，与本地已禁的手工相位和
  解析搜索有重叠。
- 处理：只保留“stiffness 是可学习动作维度”的证据，不迁移 FSM/spiral/RL 配方。

### Diffusion Contact Model for Variable Impedance（2024）

- 原始论文：`https://arxiv.org/abs/2403.13221`
- 关键点：从 variable stiffness 输入预测复杂 contact trajectory，用于降低 stiffness
  tuning 所需真实试验。
- 排除原因：核心仍是 contact outcome model + stiffness optimization，与本地
  counterfactual outcome prediction 相近；不能作为完整新方案。

### Compliance-Enabled Contact Formations（ICRA 2023）

- 原始论文：`https://arxiv.org/abs/2303.05565`
- 关键点：不估计精确接触点，而以“约束了哪些物体自由度”的 contact formation
  描述插入过程，并用 compliance 维持/增加约束。
- 正面价值：提示标签可以从 contact identity 改为 constraint/DOF semantics。
- 排除原因：其控制算法是固定 force-modulation/contact-formation path，与本地
  funnel、阶段机、force gate 和解析 servo 高度重叠。
- 保留用途：仅作为跨几何标签语义的参考，不迁移其控制流程。

### Contact Information Flow and Design of Compliance（2021）

- 原始论文：`https://arxiv.org/abs/2110.12435`
- 关键点：机械顺应性会改变接触模式检测的信息增益和确定速度。
- 研究意义：Stage-2R 的不可预测性可能不仅是 sensor/model 问题，也可能由当前高刚度
  控制闭环主动压低了可辨识信息。
- 本地差异：过去项目增加过 FT/history，但没有把 controller compliance 当作实验变量。

### AutoMate（RSS 2024）

- 原始论文：`https://arxiv.org/abs/2407.08028`
- 官方项目：`https://bingjietang718.github.io/automate/`
- 关键点：100 种 assembly geometry、specialist/generalist policy、SDF/几何感知训练。
- 本地差异：Embodied 只完成多几何 plumbing，没有验证策略所用 object-target relation
  representation；BotYard 当前也主要做 insert-focus sampling，不等于跨几何语义。
- 保留用途：支持多几何 P0 和 held-out geometry 评测设计，不直接搬其大规模 RL。

### UniCORN / HAMNET（RSS 2025）

- 官方项目：`https://unicorn-hamnet.github.io/`
- 关键点：学习任意 object-environment geometry 之间的 contact affordance，而不是只编码
  单个物体或固定任务坐标。
- 本地差异：当前固定 tip/socket 几何、CoFiT 25D 和 target-hole metadata 都未验证这种
  object-environment relation 是否跨 geometry 保留物理语义。
- 保留用途：支持“接触可供性表示 P0”，不直接采用其非抓持 manipulation policy。

### GeoManip（2025）

- 原始论文：`https://arxiv.org/abs/2501.09783`
- 关键点：以 object-part geometric constraints 作为任务接口。
- 排除原因：其 solver/constraint execution 属于解析几何规划；若用于 DexJoCo 控制会重回
  analytic servo/projection。仅可参考 task descriptor 结构。

### Learned skill precondition/effect models

- 代表工作：`Search-Based Task Planning with Learned Skill Effect Models for Lifelong
  Robotic Manipulation`，`https://arxiv.org/abs/2109.08771`。
- 排除原因：本地 root-paired preference、critic P0、counterfactual selector 和 skill
  graph 已覆盖“当前状态能否进入/成功执行某 skill”的核心问题；换成 initiation set
  不能自动解决不可辨识性。

## 排除项

| 检索方向 | 排除理由 |
|---|---|
| Reactive Diffusion Policy | 本地已有对应参考、reactive/chunk 路线和 residual/gate 禁令；不是新科学轴 |
| PAC-ACT / ACT 后训练 | SAC/RLPD、ChunkBC、PrivHI、BotYard 后训已覆盖；仍是 policy optimization |
| Contact formation fixed controller | 与 funnel、phase machine、force gate、servo 同构 |
| Learned initiation set | 与 root outcome/preference/critic/selector 同构 |
| Differentiable contact pose estimator | 本地已有 privileged exact geometry；当前主要问题不是缺 peg/socket pose estimator |
| GeoManip constraint solver | 解析 trajectory optimization/action projection，违反非裁缝边界 |
| AutoMate 全量 RL 复现 | 算力高且先验问题未过；只借鉴 geometry protocol |

## 第一轮结论

第一轮检索没有支持“再换一个策略模型”。目前只保留两个可做低算力 P0 的科学轴：

1. **Controller Compliance Causal P0**：当前固定高刚度 pose interface 是否压低了接触
   可恢复性和状态可辨识性？
2. **Cross-Geometry Contact-Affordance P0**：object-target relation 表示能否跨 geometry
   预测接触约束/可行方向，而不是记固定 tip/socket 坐标？

二者都不是训练方案；必须先通过 `CANDIDATE_SHORTLIST.md` 中的 oracle/causal P0。
