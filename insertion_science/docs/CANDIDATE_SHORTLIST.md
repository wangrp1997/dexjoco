# Candidate Shortlist

- 日期：2026-08-15
- 状态：`two_p0_candidates_no_policy_training`

## Candidate A：Controller Compliance Causal P0

### 科学问题

过去方法几乎都在固定 operational-space gains 下输出 pose/action44。它们增加了图像、
force、history、finger action、world model 和候选 ranking，却没有检验**控制闭环刚度本身**
是否决定接触信息、掉杆、jam 和动作后果的可预测性。

### 假设

在相同 snapshot 和相同 pose/finger command 下，改变 Cartesian stiffness/damping 会产生
系统性的 peg retention、contact load、drift 和 `insert_ok` 分化；至少存在一个受约束的
compliance 区间同时改善安全性或可恢复性，并使 action→outcome 关系更稳定。

### 与旧方法的本质差异

- ForceVLA：改变 observation，本候选改变 plant/controller action interface。
- ResFit/PAS/PGR：在 pose action 上加 residual/servo，本候选不添加第二动作策略。
- Stage-2R：在固定 gains 下预测后果，本候选把 gains 作为因果干预变量。
- DCM/counterfactual selector：当前 P0 不训练 outcome model，只测物理干预是否存在。

### 最小 P0

- 只复用已保存的 matched snapshots；不扩 episode，不训练网络。
- 固定 3–4 个代表性 root phase，每个 root 使用完全相同的 pose/finger action。
- 对双臂 Cartesian position/orientation gain 做预注册的小网格，例如 baseline、0.5×、
  0.25×；damping ratio 只取临界附近的少数安全值。
- 记录 peg retention、o2h drift、wrist load、contact impulse、tip/lat progress、数值稳定性。
- 与 hold action 和 baseline fixed-gain action 配对比较。

### 通过线

- 至少两个 held-out root/episode 上出现方向一致、超出 restore/noise tolerance 的物理效应；
- 不能只降低 force 同时显著恶化 retention/progress；
- matched restore 必须 bit/near-bit reproducible；
- 若效果仅存在单 root 或依赖事后选 gain，立即停止。

### 算力

CPU/单 GPU MuJoCo 小规模 matched rollout；预计远低于一次策略训练。

## Candidate B：Cross-Geometry Contact-Affordance P0

### 科学问题

当前语义多数是固定 peg/socket 的 tip、axis、lateral error。多几何资产已经存在，但尚未
证明任何表示能跨截面、尺寸和 socket instance 表达“哪些相对运动受到接触约束、哪些方向
仍可行”。

### 假设

基于 object-target surface/contact affordance 的关系表示，相比固定 task-frame 标量，能在
held-out geometry 上预测局部接触约束或可行微运动方向；若连这个 oracle P0 都失败，则不应
训练所谓 generalist insertion policy。

### 与旧方法的本质差异

- CoFiT：固定 25D socket-frame/contact-topology，未验证 held-out geometry relation。
- Target-hole metadata：只完成 ID/site/pose plumbing，没有预测物理约束。
- GNNE/trajectory retrieval：从状态找动作，本候选只验证跨几何物理表示，不检索动作。
- AutoMate：不复现其大规模 RL，只借用多几何 held-out protocol。

### 最小 P0

- 使用现有正式 4–8 个 geometry families；不训练 policy。
- 从几何和 simulator contact 导出 object-target surface relation 与局部约束标签。
- 任务是预测小 matched perturbation 下哪些 twist directions 被阻挡、可滑动或导致 jam。
- 比较固定 tip/lat/axis、raw point relation、contact-affordance relation。
- family-held-out + socket-instance-held-out；禁止 episode/geometry ID 泄漏。

### 通过线

- contact-affordance representation 在所有 held-out family fold 上稳定优于固定几何 baseline；
- shuffled object-target pairing 必须明显下降；
- 若只有同 family/同尺寸有效，停止“通用语义”方向。

### 算力

优先解析导出和小扰动标签；只允许轻量线性/小 MLP probe，不训练控制策略。

## 当前优先级

1. ~~Candidate A（Controller Compliance）~~：已放弃动态 compliance。
2. ~~Candidate B（Cross-Geometry Contact-Affordance）~~：**P0 失败，停止该方向**。
3. ~~Candidate C（Dense Intermediate Event Label）~~：**P0 失败，停止该方向**。
4. ~~Candidate D（Active Contact Probe Information）~~：**P0 失败，停止该方向**。
5. handoff/阶段接口审查：最终门失败，已停止该方向。

当前没有剩余已准入候选；Insertion Science 当前轮暂停。禁止把“候选耗尽”表述为
仿真插孔不可解，也禁止回到换模型、扩数据或抢救旧项目。
