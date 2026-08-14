# Embodied Grasp-Insertion：动机与计划

## 1. 项目定位

本项目研究灵巧手双臂装配中比策略结构更上游的问题：机器人是否具备完成插孔所需的**物体语义、手内状态、接触物理和动作可控性**。

项目不从“再训练一个 recovery policy”开始，而是先回答三个必要问题：

1. 机器人是否知道手里拿着什么，以及物体在手中的真实位姿和稳定程度；
2. 机器人是否知道目标孔是什么，以及物体与孔之间允许怎样配合；
3. 机器人是否能通过手指、抓力和双腕动作主动改变抓持与插入结果。

只有这三个问题通过严格 P0，才讨论闭环策略学习。否则，无论 BC、Diffusion、IQL、CQL 或更大的 Transformer，都只是在不充分的观测和动作接口上拟合数据。

## 2. 为什么需要独立新项目

现有 `recovery_trajectory_policy` 和 `contact_cmdp_recovery_policy` 聚焦“已经进入插入阶段以后如何恢复”。代码与数据审计表明，这个问题定义过窄：

- insert-only observation 是 37D，由 peg/socket 相对几何、孔轴/插销轴、双腕位姿和双腕 wrench 构成；
- observation 没有手指关节状态、各指接触、物体形状/身份、object-in-hand pose 或显式滑移状态；
- action 是 12D 双腕增量，手指在 handoff 后被冻结；
- recovery NPZ 只保存 37D observation、12D wrist action、tip distance 和 lateral error；
- 当前数据没有多物体、多孔型或 held-out geometry 任务定义。

这意味着现有系统虽然使用真实 MuJoCo 物理执行，却仍存在三个结构性缺口：

### 2.1 状态不可辨识

相同的腕位姿和腕力可能对应不同的手内物体位姿、不同手指接触分布和不同滑移趋势。若策略看不到这些变量，它无法判断当前应继续推进、卸载接触、调整抓力还是重新抓取。

### 2.2 动作不可控

当 peg 在手内不稳定时，wrist-only policy 无法改变手指构型、法向夹持力或接触位置。策略不能通过不存在的动作维度修复抓持。

### 2.3 任务语义被硬编码

当前几何特征直接提供固定 peg/socket 的相对误差和轴向关系。模型可能学会针对单一装配实例的坐标反馈，但这不等于理解：

- 当前拿的是哪种物体；
- 哪一端是插入端；
- 目标孔的截面、深度、公差和允许姿态；
- 哪些接触是 capture、rim、jam、seated 或 backout；
- 换一个物体或孔后，任务关系如何变化。

因此，本项目把研究对象从“恢复动作生成”前移到“可观测、可控、可泛化的装配状态表示”。

## 3. 核心研究假设

### H1：手内对象状态是插入成功的必要隐变量

object-in-hand 6D pose、手内相对运动、各指接触保持和滑移趋势，应能解释仅靠 peg/socket tip 几何无法解释的成功/失败分化。

### H2：抓持稳定性必须通过全手动作建立因果可控性

从同一 MuJoCo snapshot 出发，允许手指动作的干预应在部分不稳定根状态上改变 peg retention、手内位姿漂移和最终 `insert_ok`。若全手动作相对 wrist-only 没有稳定因果收益，则不应声称抓持控制是当前瓶颈。

### H3：插孔语义应表示为对象—目标的几何配合关系

输入不应只包含预计算 tip error，而应显式描述：

- object geometry / canonical frame；
- insertion feature，例如轴、端面、截面或关键点；
- target hole geometry / canonical frame；
- clearance、depth、symmetry 和合法姿态集合；
- 当前 object-in-hand belief 相对目标的任务状态。

真正的语义通过 held-out object/hole geometry 验证，而不是通过语言标签或固定 episode ID 验证。

### H4：物理表示必须能预测动作后的接触后果

表示的价值不以 reconstruction loss 判定，而以能否跨 episode/root 预测真实动作造成的：

- object-in-hand pose change；
- finger contact retention / redistribution；
- slip、peg loss 和 regrasp need；
- capture、rim contact、jam、partial insertion、seated、backout；
- 插入深度和真实 `insert_ok`。

## 4. 与历史失败族的区别

| 历史失败族 | 已知问题 | 本项目明确不重复 |
|---|---|---|
| 未来状态轨迹 → 动作解码 | 特权未来几何对动作预测无价值 | 不以未来 tip/geometry 序列作为动作条件 |
| Diffusion/Flow 动作或轨迹生成 | 生成模型不能弥补错误状态与监督 | P0 不训练生成策略 |
| Set-Listwise / 在线候选排名 | 离线选块无法稳定闭环部署 | 不做在线候选选择或 best-of-K |
| 在线 Physics Branch / MPPI | 昂贵且未超过 PrivHI | 物理只用于离线标签和 matched intervention |
| wrist-only Contact-CMDP | 有真实转移但没有手指可控性和对象语义 | 不在 37D/12D 接口上直接训练 critic/actor |
| gate / residual / servo | 手工逻辑掩盖策略能力 | 不通过规则切换、动作混合或特判过门 |

本项目改变的是**问题定义、状态表示、动作接口和数据干预设计**，不是更换网络架构。

## 5. 目标任务分解

完整任务按物理状态而非固定时间段分解：

1. **Acquire**：建立有效物体接触；
2. **Stabilize**：形成可承受后续运动和装配接触的抓持；
3. **Transport**：保持 object-in-hand pose 可控地接近目标；
4. **Pre-align**：根据对象—孔几何建立可进入接触的相对位姿；
5. **Capture**：识别并利用入口/rim 接触进入约束流形；
6. **Insert**：在保持抓持的同时推进并处理 jam；
7. **Verify**：确认 seated / `insert_ok`，排除假接近和 backout；
8. **Recover/Regrasp**：当抓持或接触模式不可继续时，通过全手动作恢复。

这些阶段是用于数据标注和诊断的物理语义，不允许直接变成手写 gate 或状态机控制器。

## 6. 表示设计

### 6.1 特权教师状态

P0 可使用 MuJoCo 真值建立可辨识性天花板：

- object mesh/shape ID 与尺度；
- object world pose、velocity；
- object-in-hand relative pose 与相对速度；
- 双手腕与全部手指关节位置/速度；
- 各指尖和掌部的接触点、法向、切向力与 slip；
- target geometry、hole frame、clearance、depth、symmetry；
- object insertion feature 相对 target mating feature 的位姿；
- assembly contact mode 与真实 `insert_ok`。

特权状态只用于确定问题是否可解和生成监督，不是最终部署输入。

### 6.2 可部署 belief 输入

后续学生候选输入：

- 双臂/全手 proprioception history；
- 腕部 wrench history；
- 指尖 tactile/contact history；
- wrist/scene vision；
- object/target geometry descriptor 或从视觉得到的几何表示；
- 任务指令或 object-target pairing ID，仅作为配对信息，不能替代几何。

输出不是单一“语义 token”，而是带不确定性的 belief：

- object-in-hand pose distribution；
- grasp stability / slip probability；
- contact mode distribution；
- object-target mating transform distribution；
- action-conditioned short-horizon physical consequence。

### 6.3 动作接口

P0 至少比较：

- wrist-only 12D；
- full wrist + finger 44D；
- 44D 加可选低层 torque/impedance 参数，但不得与主对照混用。

首轮只使用现有 44D position-delta 接口，避免同时更换控制器。只有 44D 已证明能改变抓持结果后，才研究阻抗或力控制。

## 7. 数据设计

### 7.1 数据来源

- 现有完整 demo 与 full-episode rollout；
- 从真实 rollout 保存的 MuJoCo snapshot；
- snapshot restore 后的 matched interventions；
- 后续多物体/多孔几何变体，但必须记录精确资产、尺寸、公差和随机种子。

### 7.2 Matched intervention

同一 snapshot 下只改变一个可解释因素：

- 手指闭合/张开方向和幅度；
- 单指或指组动作；
- 双腕 transport / insertion action；
- object-in-hand 初始微扰；
- 目标孔位姿或几何；
- 接触摩擦、质量等物理参数仅用于鲁棒性评测，不作为伪标签。

所有结果必须来自真实 MuJoCo rollout。禁止解析拼接 recovery、插值轨迹或用距离阈值代替接触结果。

### 7.3 每步字段

- episode、root、branch、asset、geometry 和 seed；
- 完整 snapshot 或可精确恢复的状态引用；
- full proprioception、finger state、wrist wrench、finger contacts；
- object/target pose、shape descriptor 与 mating features；
- 44D action 及动作来源；
- next state 与 contact-mode transition；
- slip、peg retained、regrasp need、jam、insert depth、`insert_ok`；
- 成功和失败分支全部保留。

### 7.4 Split

必须同时报告：

- episode-held-out；
- root-held-out；
- object-instance-held-out；
- geometry-family-held-out；
- 必要时 physics-parameter-held-out。

同一 snapshot、派生分支、同一对象实例的近重复状态不得跨 split。

## 8. P0-A：现有系统可解性审计

目标：不采新数据、不训练 policy，先量化现有 full-episode 数据是否包含所需字段。

输出：

- 85D full observation 和 44D action 的逐字段 schema；
- 手指状态、物体 pose、接触真值、目标 geometry 是否可从 snapshot 重建；
- 现有 100 episode 中物体/孔资产的真实多样性；
- 抓持丢失、手内漂移、jam 与成功的跨 episode 分布；
- 缺失字段及最小定向重放需求。

通过条件：能从现有 snapshot 精确恢复 full hand/object/contact state，并能构造 episode/root 原子 split。

停止条件：snapshot 不含必要状态、恢复不确定，或所谓多样性实际只是同一几何的位姿变化。

## 9. P0-B：Observability 天花板

目标：验证抓持与装配接触状态能否从部署可用历史中被识别。

任务：

1. object-in-hand relative pose / velocity 回归；
2. per-finger contact retention 与 slip 预测；
3. contact mode 分类；
4. regrasp-needed 与短期 peg-loss 风险预测；
5. object-target mating transform 估计。

对照：

- proprioception only；
- proprioception + wrist wrench；
- proprioception + finger contact/tactile；
- 加 vision/geometry descriptor；
- H1/H4/H8/H16 history；
- shuffled tactile、shuffled geometry 和跨 object mismatch。

硬门：

- episode/root-held-out 明显优于无 tactile、无 geometry 和 shuffle；
- 指标按 episode/object 等权，不由单一对象支配；
- 不确定性与误差相关，不能只给过度自信点估计；
- 若 privileged truth 可预测、部署输入不可预测，则先解决 sensing，不训练控制策略。

## 10. P0-C：全手动作因果可控性

目标：证明手指动作在当前 MuJoCo 与控制接口中确实能改变抓持和插入后果。

协议：

- 从不稳定但尚未掉杆的 matched roots 开始；
- wrist-only 与 full 44D 使用相同 snapshot、相同腕动作预算和 rollout 时长；
- 比较 hand action intervention、随机 hand action、demo hand replay 和 hold-finger；
- 记录 object-in-hand drift、接触保持、peg loss、过载、jam 和 `insert_ok`；
- 至少跨三个 episode/object seed，不能用单 episode 特判。

通过条件：存在可重复、跨 root 的手指动作因果效应，并且不是仅改善 tip distance。

停止条件：44D 动作无法稳定改变抓持结果，或收益只来自不公平的腕动作/时长差异。此时优先修复 actuator、控制频率、接触模型或手部接口。

## 11. P0-D：对象—孔几何语义

目标：证明模型学习的是配合关系，而不是固定坐标轨迹。

最小数据变化：

- 多个 peg 截面/尺度/长度；
- 多个 hole geometry / clearance / depth；
- 具有旋转对称与非对称插入约束的对象；
- object-target 正确配对和 mismatch 负对照。

任务：

- 预测合法 mating transform 集合；
- 预测当前相对位姿是否可 capture / insert；
- 预测接触动作后的 mode transition；
- 在 held-out geometry family 上评测，而非只 held-out episode。

通过条件：geometry-conditioned 表示稳定优于不含 geometry、只含 ID 和 shuffled-pair 对照。

停止条件：训练集对象少到无法形成 held-out geometry，或模型只靠资产 ID/episode 泄漏完成任务。

## 12. 策略准入门

只有以下条件全部满足，才允许新建控制模型：

1. Observability P0 证明部署输入能够估计抓持与接触 belief；
2. Controllability P0 证明全手动作能因果改善 peg retention / `insert_ok`；
3. Semantic P0 证明对象—孔几何条件在 held-out geometry 上有效；
4. snapshot、split、标签和 matched intervention 均通过真实性审计；
5. 相对既有失败族的差异轴已写入状态文件。

准入后的首个策略也必须是容量可控的简单基线，优先验证：

`belief state + object-target geometry -> full 44D action`

不直接上 Diffusion、Flow、VLA 或大 Transformer。先证明表示和动作接口有效，再研究策略规模。

## 13. 绝对禁止

1. 在 37D observation + wrist12 action 上再训练新恢复 actor；
2. 把腕力当作完整抓持状态，忽略手指接触；
3. 把预计算 tip/axis error 表述为模型学会插孔语义；
4. 只在同一 peg/socket 上按 episode 切分，却宣称跨物体泛化；
5. 用 object ID 替代几何，并把记忆结果表述为语义理解；
6. 手指动作未证明可控前训练 full-hand policy；
7. 使用解析 recovery、在线 MuJoCo branching、MPC、best-of-K、gate、residual 或 servo；
8. 用 tip distance 改善替代 `insert_ok`、peg retention 和接触模式改善；
9. 在 observability 失败后靠更大网络或更多 epoch 抢救；
10. 未按 snapshot/root/object 隔离派生数据；
11. 同时改变 sensing、action、reward 和 policy，导致无法归因；
12. 为了“有模型可训”跳过可解性审计。

## 14. 参考方向

以下工作只提供模块级启发，不直接复制为主方案：

- **ViHOPE**：视觉—触觉联合估计手内物体 6D pose，并显式利用形状补全；迁移点是 object-in-hand belief 与 shape/pose 联合监督。
- **TacSL**：提供接触力场/触觉模拟和插入任务，展示 peg-in-hand pose、socket pose 变化下的触觉策略评测；迁移点是触觉观测与随机化协议，而非其 RL 算法。
- **Tac-Man**：通过触觉变化维护稳定接触；迁移点是 contact stability 表示与 action-conditioned contact change，不采用其在线伺服结构作为部署方案。

正式实现前必须进一步阅读论文和官方代码，并在进展文件记录具体迁移模块、接口差异和拒绝复制的部分。

## 15. 预期贡献边界

若项目成功，贡献应表述为：

> 对象—孔几何条件、手内状态 belief 和全手接触干预共同构成灵巧插入的必要表示与可控接口，并通过跨对象/跨几何严格对照证明其对真实物理后果的预测与控制价值。

不能表述为：

- “换了一个更大的 policy”；
- “使用物理仿真所以理解物理”；
- “tip 更近所以学会插孔”；
- “固定对象成功所以具备通用装配语义”。
