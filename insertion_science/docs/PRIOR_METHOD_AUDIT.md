# DexJoCo 插孔既有方法审计（2026-08-15）

## 目的

本文件只记录已经检查到的代码、实验和结论，不提出下一算法。

在完成本审计前提出的“主动接触辨识 + 技能检索”草案撤回。`reach_insert_rl`
已经覆盖 force-history disambiguation、非参数 trajectory retrieval、候选动作
counterfactual outcome、set-listwise selector 和 long-horizon outcome；未证明新差异前，
不得把相近方案重新命名为创新。

## 审计范围

已检查：

- 当前仓库 `/home/wangrenpeng/dexjoco` 的 Git 历史、全部项目目录、基线配置、
  评测入口、当前工作树和 `/mnt/hdd/dexjoco/outputs` 全部一级项目根。
- 独立仓库 `/home/wangrenpeng/reach_insert_rl`、`dex_voc_insert`、
  `botyard_vla`、`contact_funnel`、`lai` 的 Git 历史、分支、stash、工作树、
  状态文件和 HDD 评测结果。
- 相关仓库的本地/远端分支。除 `botyard_vla` 外，没有未合并的算法分支；
  `botyard_vla/sim_v3` 存在尚未提交的 insert-focus 数据与阶段评测主线。
- 当前仓库中的 ACT、Diffusion、multi-task DiT、π0.5、ForceVLA、GR00T、
  T-Rex、ResFit、DexQuery、PoseInsert、PPO tracking、interaction retarget、
  skill replay、skill graph、regrasp/datagen 和恢复策略项目。

尚不能声称检查了用户主目录中与 DexJoCo 完全无文本/路径关联的任意仓库；本次用
`bimanual_assembly`、`DexJoCo`、`insert`、`assembly`、`grasp`、`reach`、
`contact`、`recovery` 等标记扫描后，相关仓库集合为上述六个主仓库。

## 统一分类

- **硬失败**：有闭环或严格 held-out 结果，未达到自身门槛。
- **局部阳性**：某一阶段或小评测集成功，但不能外推为端到端通用方法。
- **未完成**：代码/数据或 P0 存在，但尚无足够闭环结论。
- **解析/特权基线**：依赖 oracle geometry、规则相位、重标定、对象辅助、
  snapshot branching 或轨迹回放；不能与纯策略结果混为一类。

## DexJoCo 端到端学习基线

| 方法 | 主要输入/策略 | 硬结果 | 状态 |
|---|---|---:|---|
| ACT | 图像/本体，action chunk | `0/50`，另一个 legacy run `0/50` | 硬失败 |
| Diffusion | 图像/本体，生成式 chunk | `0/50` | 硬失败 |
| multi-task DiT | 多任务生成式策略 | `0/50` | 硬失败 |
| GR00T | VLA；含 replan03 | `0/50`、`0/50` | 硬失败 |
| T-Rex | VLA + tactile VQ 路线 | 4 个 checkpoint 均 `0/50` | 硬失败 |
| DexQuery | 子任务 query + phase eval | `0/50` | 硬失败 |
| π0.5 | 图像/本体 action chunk | seed `0/50, 4/50, 2/50` | 弱阳性，低成功率 |
| ForceVLA wrist | π0.5 + wrist force | `1/50, 2/50, 4/50` | 弱阳性 |
| ForceVLA finger | π0.5 + finger force | `4/50, 1/50, 2/50` | 弱阳性 |
| ForceVLA both | π0.5 + wrist/finger force | `5/50, 7/50, 5/50` | 当前大规模端到端最佳族 |
| ResFit | frozen ForceVLA + TD3 residual | `0/10` @5k，`0/10` @50k | 硬失败；zero residual 为 `1/10` |

结果来自 `/mnt/hdd/dexjoco/outputs/{act,diffusion,multi_task_dit,pi0.5,forcevla,groot,trex,resfit_dexjoco,dexquery}`。

## PoseInsert、LAI 与解析插入

### PoseInsert action44

- 同分布批评测最好 `14/100`：
  `/mnt/hdd/dexjoco/outputs/poseinsert_sim/batch_eval_action44_restart.log`。
- 11 条 holdout 最好 `2/11`：
  `/mnt/hdd/dexjoco/outputs/poseinsert_sim/holdout_eval_restart.log`。
- 说明 action44 插入策略存在局部能力，但泛化很弱；不能写成端到端成功。

### LAI

- `teacher_phase_reverify` 指定 5 个 episode 中为 `4/5`，ep27 失败：
  `/home/wangrenpeng/lai/runs/teacher_phase_reverify/eval_5.log`。
- 评测从 demo replay 到 `peg_lift_end` 后开始，不解决自主 reach/grasp。
- 虽然命令使用 `converge_blend=0`，runner 仍包含阶段执行、INSERT lateral freeze、
  jam 检测、retreat/hold 语义逻辑，并接入 PoseInsert/recalibrating controller。
- 因而分类为“固定 handoff 后的混合学习/解析局部阳性”，不能与纯 network-only
  端到端 `4/5` 等同。

### hybrid insert / skill graph

- π0.5 hybrid 小评测为 seed0 `2/5`、seed1 `0/5`。
- `skill_graph` 的 assembly graph 使用 PoseInsert 和规则 failure monitor/regrasp；
  π0.5 skill-graph 小评测 `0/3`。
- 这些路线证明阶段编排接口存在，但没有稳定端到端增益。

## `reach_insert_rl` 方法历史

当前可靠基准是 `PrivHI`：holdout `3/3`，expand15 `6/15`。其余主要路线：

| 方法族 | 关键结果 | 结论 |
|---|---|---|
| privileged SAC/RLPD/1-step BC | compounding，闭环低成功 | 耗尽 |
| PrivChunkBC | 局部 `2/3`，扩评 `5/15` | 阳性但低于 PrivHI |
| PrivHI | holdout `3/3`，expand15 `6/15` | 当前该仓库主基准 |
| DualPhase full-demo | holdout `0/3` @2.5k/5k | 硬失败 |
| RIVA/GGAM/ALATCH/GNNE/ACH | 未超过主基准或有害 | 耗尽 |
| DART recovery fine-tune | holdout `1/3`，破坏已有成功 | 硬失败 |
| contact residual / PGR / PAS | 过拟合或属于禁用 gate/residual | 禁重复 |
| CoFiT funnel flow | `0/3` @2.5k/5k | 数据族证伪 |
| physics branch MPC/MPPI | expand15 `5/15 < 6/15` | 昂贵且无增益 |
| task-frame trajectory retrieval | 74 成功 demo，holdout `0/3` | 非参数检索证伪 |
| residual VQ contact options | tokenizer 通过，策略 holdout `1/3` | 离散 option 未过门 |
| temporal reachability | predicted action improvement 为负 | 预测目标证伪 |
| distributional/joint future plans | best-of-K oracle 有益，deployable selector 为负 | 可达候选≠可选候选 |
| force-history disambiguation | wrist H32 相对 no-force 约无提升 | 动态力历史未消歧 |
| counterfactual outcome MLP/tree | 离线 tip progress 有小增益 | 仅离线局部阳性 |
| set-listwise ranking | 离线 `30.0%→38.89%`，strict deploy `1/3 < nominal 2/3` | 部署失败 |
| long-horizon counterfactual | 96 roots 仅 4 个 insert 分化；仅 1/4 fold 正增益 | 稀疏终端标签路线停止 |

正式状态：`/mnt/hdd/dexjoco/outputs/reach_insert_rl/innovate/state.json`。

## `dex_voc_insert` 与 Contact Funnel

- Stage A + oracle demo handoff 的 DiffChunk 达到 `4/4`，只证明理想 handoff 插入段。
- B3 `alpha=1.0` 可成功；`alpha=0.98/0.95` 成功跌为 0，存在明显 soft-dynamics cliff。
- PPO Stage B 历史峰值约 `insert=0.05`。
- pose-relative ChunkBC 最佳 tip 约 `129.6 mm`，`insert_ok=0`。
- CoFiT external-wrench PD 数值不稳定；bounded SE(3) homotopy 只是对象辅助基线。
- Reach CoFiT 数据 45,741 条，2.5k/5k 均 `0/3`，正式停止该数据族。

状态：`/mnt/hdd/dexjoco/outputs/dex_voc_insert/innovate/state.json`；
CoFiT：`/home/wangrenpeng/contact_funnel/PROGRESS.md`。

## BotYard-VLA

- V1/V2 自研 Flow/空间 conditioner 的 assembly 后训主要闭环为 `0/10`；
  hybrid diagnostic 小样本 `1/3`。
- π0.5 差异实验 30 ep：B0/B1/J5/J5A/J5AR 最终均约 `0–1/30`。
- 阶段评测：B0 succeed `10%`、peg grasp `97%`、tray grasp `73%`、align `10%`；
  J5A succeed `0%`、peg grasp `43%`、tray grasp `13%`、align `0%`。
- `sim_v3` 当前工作树存在尚未提交的 insert-focus sampling、阶段评测和训练配置。
  这是**正在进行的路线**，不得提前归类为已失败，也不得在新项目里平行重复。

文档：`/home/wangrenpeng/botyard_vla/docs/V3_INSERT_MAINLINE.md`。

## 恢复与 embodiment 路线

### Recovery Trajectory Policy

- 数据量门通过：27 frontier roots、7 success episodes、69 success trajectories。
- future-state trajectory oracle 相对 Direct 平均 `-15.7%`，24 folds 对 Direct 胜率 0%。
- 停止 future trajectory → action decoder，不训练 Diffusion/Flow。

### Root-Paired Recovery

- 完成 pair 数据设计和清单，但没有启动正式训练。
- 后续被 Contact-CMDP/embodiment scope 审计降级；不能写成“模型已失败”。

### Contact-CMDP

- 121 trajectories、8,856 transitions、69 success / 52 failure。
- 在 critic 训练前被 scope falsification：37D obs 缺手内位姿/手指/任务语义，
  wrist12 动作冻结手指。
- 降级为固定物体、稳定抓持后的局部诊断工具。

### Embodied Grasp-Insertion

- H2 matched intervention 证明 finger action 能改变物理结果。
- Stage-2R 在完整 o2h/qdot/finger q/qdot/wrist FT + action 下仍不能稳定预测
  relative-to-hold 后果；36 samples、12 roots，decision tree B。
- 结论只是否定当前预测设定和性价比，不是否定仿真插入可实现。
- 项目于 2026-08-15 正式冻结，不得直接解冻追加策略训练。

## Retarget、tracking、replay 与 regrasp

- `interaction_retarget` 已完成 sidecar、canonical grasp、IK、contact repair、
  per-demo lift reference 和初始场景 pose retrieval；这些属于抓取/重放基础设施。
- 当前 skill retrieval 只按 peg/tray 初始 `xy+yaw` 距离检索 demo，且主要保存
  grasp/lift reference；没有证据证明其本身解决了 held-out insertion。
- `dex_track_assembly` 完成轨迹导出和 PPO/MJX tracking 训练基础设施，但现有日志没有
  可与 `insert_ok` 对齐的正式闭环成功率，因此分类为“未完成评测”，不是硬失败。
- `dexjoco_datagen` 和 `skill_graph/regrasp` 提供 failure-to-regrasp 生成、模板选择和
  graph recovery 接口；现有产物不足以宣称稳定恢复成功率。
- demo/qpos replay 只证明回放或诊断链路，不能作为策略泛化结果。

## 禁止重复的方法族

新方案在提出前必须逐条说明与以下路线的本质差异：

1. 单步 BC/SAC/RLPD、普通 action chunk BC。
2. residual、gate、servo、解析 action projection 或混合第二策略。
3. 仅调整 phase、reward、seat threshold、execution horizon 或 replan frequency。
4. future state/subgoal/trajectory → action decoder，除非先通过新的 oracle 硬门。
5. 静态/动态 force history 判别、contact-mode classifier 的简单重做。
6. task-frame nearest-neighbor trajectory/skill retrieval。
7. candidate action 的 counterfactual outcome predictor、listwise ranker 或在线 branch MPC。
8. DART/噪声 recovery augmentation、解析 funnel/counterfactual chunk 扩增。
9. 再换一个 Diffusion/Flow/VLA 主干但不改变数据真实性、状态可辨识性和控制接口。
10. 在固定 peg/socket 特权几何上取得结果后宣称通用 embodied insertion。

## 当前不能下的结论

- 不能说所有插孔学习方法都失败：ForceVLA、π0.5、PoseInsert、PrivHI、LAI 和
  ideal-handoff DexVOC 都有不同范围的阳性。
- 不能说触觉/力无用：ForceVLA-both 明显高于多个无力基线，但仍需严格同预算分析。
- 不能说静态 privilege 完全无信息；只能说 Stage-2R 的当前小样本线性后果预测未过门。
- 不能把 LAI `4/5`、DexVOC `4/4` 或 π0.5 hybrid `2/5` 当作端到端通用成功。
- 不能在 BotYard insert-focus 主线尚未完成时平行提出同类 insert reweight/LoRA 方案。

## 下一步门槛

只有在用户确认本审计没有漏项后，才能：

1. 从本表生成方法族覆盖矩阵；
2. 对未覆盖机制执行新的论文/官方实现检索；
3. 提出最多 2–3 个候选；
4. 每个候选先写与所有禁重复族的差异证明和最小 oracle/P0，不直接训练。
