# Scientific Problem Map

## 目标

把过去项目的失败从“某个模型没训好”还原为可检验的科学问题。任何新候选必须
明确解决至少一个未决问题，并证明没有重复既有方法族。

## P1：决策状态是否充分

### 已有证据

- Embodied Stage-2R 加入 object-in-hand、finger q/qdot、wrist state/FT 和 action 后，
  仍不能稳定预测 relative-to-hold 物理后果。
- `reach_insert_rl` 的 force-history P0 没有稳定优于 no-force。
- counterfactual/listwise 方法离线可改善，部署 handoff 上退化。

### 不能据此声称

- 不能声称所有历史或触觉都无用；ForceVLA-both 仍是强端到端基线之一。
- 不能把小样本线性不可预测等同于信息论不可观测。

### 未决问题

- 不同结果是否源自未记录的接触微状态、仿真积分敏感性，还是任务标签过粗？
- 哪个最小历史窗口或事件表示能够在 episode-held-out 下增加可辨识信息？
- 若最充分的 simulator state 也不能区分，是否应改研究目标而非继续预测？

## P2：动作接口是否包含必要控制权

### 已有证据

- wrist12 冻结手指，无法解决滑移、抓力和手内位姿变化。
- matched finger intervention 已证明手指动作可以改变物理结果。
- ForceVLA-both 相比多个无力或单力输入基线有更高成功率。

### 未决问题

- finger action 的有效作用是稳定抓持、改变 hand-object pose，还是改变插入接触顺应性？
- wrist 与 finger 是否必须联合规划，还是可以通过物理上有定义的控制分解？
- 当前 action44 的位置控制接口是否适合表达强接触顺应行为？

## P3：监督目标是否与真实任务因果一致

### 已有证据

- future trajectory oracle 不优于 Direct，说明未来几何轨迹不必然决定可执行动作。
- analytic funnel、DART 和平滑扰动可能产生不保持真实抓持动力学的标签。
- 长时 counterfactual 中 `insert_ok` 分化稀疏，终端标签难以支持稳定 ranking。

### 未决问题

- 哪种中间物理事件既足够稠密，又与最终插入存在可验证的因果关系？
- 是否存在比 binary `insert_ok` 更稳定、跨几何一致的接触状态转移定义？
- 如何验证标签不是固定 peg/socket 几何上的解析捷径？

## P4：成功是否位于训练分布支持内

### 已有证据

- DexVOC 在 oracle/hard handoff 成功，soft dynamics 轻微变化后出现成功 cliff。
- set-listwise 离线提升但部署失败，表明候选根与真实 handoff 分布不一致。
- PoseInsert 同分布最好 `14/100`，11 条 holdout 最好 `2/11`。

### 未决问题

- 失败来自状态支持缺口、动作支持缺口，还是闭环误差累积？
- 成功轨迹在状态—动作空间中是否形成连续可达区域，还是孤立窄通道？
- 数据应围绕哪些物理边界采集，才能增加支持而非简单堆 episode？

## P5：阶段耦合是否是主要瓶颈

### 已有证据

- 多个端到端 VLA 在抓取、保持、运输、对孔之前就失败。
- LAI、PoseInsert、DexVOC ideal handoff、PrivHI 表明固定 handoff 后存在局部插入能力。
- BotYard 阶段指标显示抓取率与 align/insert 指标存在明显断层。

### 未决问题

- 进入插入阶段所需的 hand-object-target state distribution 应如何定义？
- 上游 grasp/transport 的哪些误差是下游可恢复的，哪些会使插入任务不可行？
- 是否需要联合优化阶段接口，而不是独立策略或手工相位切换？

## P6：任务语义和几何泛化是否真实

### 已有证据

- 多数成功和诊断依赖固定 peg/socket、解析 tip/axis/lateral 几何。
- Embodied 项目建立了多几何 plumbing，但没有进入策略泛化验证。
- 当前结果不足以说明策略理解 object、target 和 mating relation。

### 未决问题

- 最小可部署任务描述应包含哪些几何、拓扑或装配约束？
- 哪些表示能跨尺寸、截面和 socket instance 保持相同物理语义？
- held-out geometry 的成功判据和训练标签如何避免固定坐标泄漏？

## 候选准入原则

候选必须同时回答：

1. 对应哪个未决问题；
2. 旧项目为什么没有回答它；
3. 新变量、新干预或新目标是什么；
4. 最小 P0 如何在不长训的情况下证伪；
5. P0 失败后为什么不允许靠增大模型或数据量抢救。

