# DexJoCo Project Vision

> 当前状态：2026-08-23 已停止完整姿态估计、IEKF-PBVS、主动协方差优化和检索-SQP 作为项目主线。权威路线见 [METHODOLOGY.md](METHODOLOGY.md)。本文件后半部分保留原 DexContactRAM 愿景，仅作为历史档案。

## Intent-Preserving Sensor-Compliant Bimanual Execution

Last updated: 2026-08-23

## 一句话定位

项目不再假设 ego RGB 能提供毫米级完整相对姿态。π0.5 负责理解、抓取、粗搬运和粗对准；进入预定义精细插入阶段后，上层任务状态机向独立小脑提供新的时变双腕 action chunk。小脑只使用本体、腕力、多指力、动作历史和可选粗视觉，对 chunk 进行重定时、接触响应补偿、双手约束释放、柔顺执行、硬力退让和成功/退出判断。

核心链路为：

`π0.5 抓取、搬运与粗对准 → 精细阶段接管契约 → 冻结时变双腕意图 → 视触觉补偿与重定时 → 双手内力释放 → 柔顺执行/退让 → 可部署成功或退出`

当前核心创新不再是估计完整物体姿态，而是：

1. 保留 π0.5 在自然 handoff 后仍然重要的时变双腕接近意图；
2. 用多指抓持稳定性和双腕力残差连续调节 chunk 相位与左右手运动分配；
3. 在不读取真实几何的条件下实现硬力停止、有界退让、chunk 耗尽处理和重新请求策略；
4. 将所有端到端结论建立在显式 handoff、可部署终止和完整自然任务上。

完整姿态估计路线停止的原因是冻结 held-out P1 三项全失败：横向 P90 `21.06 mm > 1.2 mm`，倾斜 P90 `0.2077 rad > 0.03 rad`，深度 P90 `30.55 mm > 3 mm`。该决定由代码硬门生成，不允许在当前项目内修改阈值复活。

# 历史 DexContactRAM 愿景档案

以下内容记录原始检索、信念估计和约束优化设想，不再定义当前研究结论或实现优先级。

## 问题边界

当前任务的工件、孔、尺寸和装配关系固定。需要泛化的不是新类别或新尺寸，而是 π0.5 每次产生的：

- 未见过的工件抓取姿态；
- 未见过的托盘抓取姿态；
- 不同的工件—孔粗对准状态；
- 不同的多指接触、滑移和插入偏差；
- 摩擦、质量、接触刚度和传感噪声扰动。

π0.5 已经使用 DexJoCo 双臂装配数据完成后训练。本项目不再次训练、微调或修改 π0.5，也不在 π0.5 动作上叠加残差。

## 科学问题

π0.5 具有强语义和长程操作能力，但无法稳定解决毫米级、接触丰富的插孔。精密阶段又不能依赖真实环境中不存在的物体位姿真值。

本项目研究：

> 能否把 RAM 的“检索结构化对象知识以增强规划”扩展为“检索结构化接触知识以增强灵巧手闭环控制”，并仅依靠真实可获得的触觉、本体和弱视觉观测，适应 π0.5 未见的抓取与粗对准结果？

## 对 RAM 的核心扩展

`~/retrieval-augmented-manipulation` 使用对象模板、特征对应、抓取点和功能平面，把 VLM 不具备的精确空间知识注入后续规划。

DexContactRAM 不简单复用其视觉执行栈，而是将静态对象 primitive 扩展为双臂灵巧手动态接触技能模板：

| RAM | DexContactRAM |
| --- | --- |
| 对象网格与规范坐标 | 工件—孔规范坐标与接触表面 |
| 单夹爪抓取点 | 多指接触区域、法向和手指协同 |
| 功能平面 | 工件轴、孔轴、孔口平面和插入方向 |
| 模板到实例的视觉迁移 | 模板到当前抓取/接触信念的状态实例化 |
| 空间约束与轨迹规划 | 接触模式先验与约束技能优化 |
| 静态对象知识 | 成功插入的多指接触演化知识 |

真正的创新不在“再训练一个小脑网络”，而在于：

1. 建立可检索的多指接触装配知识模板；
2. 用仿真真值蒸馏真实可用的触觉/本体信念估计；
3. 用检索先验降低接触优化的搜索难度；
4. 在接触反馈下持续变形成功技能并安全执行。

## 完整算法流程

### A. 离线：接触知识模板

固定 CAD 和 MuJoCo 装配模型提供规范模板：

- 工件和托盘表面；
- 工件轴、孔轴、孔口平面和插入深度；
- 每根手指的可接触区域和表面法向；
- 稳定手—物相对姿态分布；
- 接近、单侧接触、双侧接触和正常插入等接触模式；
- 各模式允许的运动方向、受约束方向和期望接触力范围。

模板是几何与接触知识，不直接输出控制动作。

### B. 离线：成功示范记忆

成功示范统一转换到工件—孔、右手—工件和左手—托盘坐标系，保存：

- 46D 本体状态和 44D 双臂灵巧手动作；
- 工件—孔相对位姿；
- 双手—物体相对位姿；
- 接触数量、抓持状态、插入接触和滑移；
- 每条轨迹的接触演化和到达孔口的过程。

这些数据构成正常技能记忆和检索先验，不用于普通端到端行为克隆。

### C. 离线：仿真真值蒸馏

MuJoCo 训练阶段可以访问：

- 物体真实位姿；
- 真实接触点、法向和接触力；
- 手—物滑移；
- 工件—孔真实相对状态。

真值只用于：

- 监督触觉/本体观测模型；
- 学习接触是否粘着、滑移或失效；
- 标定因子图/MHE 的观测噪声和鲁棒核；
- 生成接触模式和插入进度标签；
- 在控制器外部评估状态估计误差。

正式测试时，控制器禁止读取以上真值。

### D. 在线：π0.5 粗操作

π0.5 完成：

- 任务理解；
- 双手抓取；
- 抬升和搬运；
- 工件和孔的粗对准。

双手抓持后，小脑开始估计接触状态；达到可执行条件后接管精密阶段。物体掉落并需要重新抓取时才将控制权交还 π0.5。

### E. 在线：触觉/本体信念估计

小脑只使用真实可获得的观测：

- 双臂腕部位姿和关节编码器；
- 手指关节位置、速度和力矩；
- 手指接触位置、法向和接触力；
- 历史控制命令；
- ego 与腕部图像提供的弱初始化或漂移修正。

状态估计采用分层表示，不要求在进入接触控制前先把完整物体 SE(3) 位姿估计到毫米级：

- 接触前只维护弱视觉粗初始化、当前手腕/手指状态及其不确定性，用于检索相近技能；
- 接触后重点估计横向修正方向、轴线倾斜方向、插入进度、接触模式、手—物滑移和抓持稳定性；
- 完整工件—孔、工件—右掌和托盘—左掌位姿保留为训练辅助变量和外部评测指标，不作为在线控制的硬性精度门槛；
- 每个连续量和离散模式都必须输出置信度或有效性，供检索和鲁棒优化使用。

接触后可使用滑动窗口优化、MHE 或更轻量的鲁棒滤波。选择依据是能否改善检索、修正方向和插入判断，而不是算法名称。候选因子包括：

- 手指接触点位于 CAD/RAM 模板表面；
- 接触法向与模板表面法向一致；
- 粘着接触时满足近似无滑移约束；
- 多指运动与物体刚体运动兼容；
- 接触力近似满足物体力/力矩平衡；
- 相邻时刻运动满足动力学和平滑先验；
- 弱视觉观测作为低权重或鲁棒因子。

滤波只能降低已有观测噪声，不能补回缺失信息；接触发生后新增的力、运动兼容性和几何约束才是状态可观测性来源。若轻量滤波已满足控制决策，不为了复杂而强制使用完整因子图。

### F. 在线：接触技能检索

使用信念状态而非物体真值查询成功技能库：

- 接触前的弱视觉粗初始化和当前双手构型；
- 接触后的横向/倾斜修正方向和插入进度；
- 当前接触模式；
- 滑移与抓持稳定性；
- 状态估计协方差。

检索返回：

- 相似成功状态；
- 后续接触模式序列；
- 名义腕部、手指和接触力轨迹；
- 可变形的关键点；
- 允许运动方向和受约束方向；

检索轨迹不能直接回放，也不能作为 π0.5 残差。

### G. 在线：约束技能优化

第一版不做完整 44D 接触 MPC。优化变量限定为：

- 少量左右腕部关键点；
- 少量手指协同参数；
- 接触力或阻抗目标；
- 技能时间缩放参数。

使用 QP/SQP 对检索技能进行局部变形。目标函数包括：

- 接近检索到的成功接触演化；
- 保持双手抓持稳定；
- 减少估计的横向和轴线误差；
- 控制接触冲击和滑移；
- 保持动作与接触力平滑；
- 增加插入进度。

约束包括：

- 双臂和手指关节范围；
- 机器人运动学；
- 多指抓持兼容性；
- 摩擦锥和接触力范围；
- 孔壁非穿透和插入方向；
- 根据状态协方差收紧的鲁棒安全边界。

检索先验提供好的模式、方向和初值，优化器只解决当前信念下的局部适配问题，避免从零搜索完整高维接触轨迹。

### H. 在线：柔顺执行与安全停止

优化结果由阻抗或混合力/位控制器执行：

1. 执行短技能片段；
2. 读取新的触觉和本体状态；
3. 更新因子图信念；
4. 判断接触是否符合检索技能；
5. 继续、重新变形或安全停止。

若检测到异常力、持续无进展、明显滑移或掉落风险，小脑停止输出精密动作并交还 π0.5 或触发环境重置。这里的交还是安全退出边界，不属于小脑的恢复策略；本项目不学习、检索、优化或执行失败恢复技能。

## 图像的作用

图像不能被忽略，但不承担毫米级精密位姿真值：

- π0.5 使用图像完成语义、抓取和粗对准；
- RAM/DINO 模板特征可用于对象识别和粗位姿初始化；
- ego 与腕部图像作为因子图中的弱约束；
- 遮挡或图像抖动时，精密阶段依靠触觉、本体和接触模板继续估计；
- 图像可以检测掉落或严重全局漂移。

## 仿真即真实测试协议

为了验证真实部署可行性，正式仿真评估必须建立传感器防火墙。

控制器允许访问：

- 相机图像；
- 机器人本体状态；
- 手指触觉、接触力或可映射到真实硬件的电机力矩；
- 历史动作。

控制器禁止访问：

- MuJoCo 物体 `xpos/xquat/xmat`；
- 工件—孔真实误差；
- privileged primitive provider；
- 真实接触模式标签；
- 评测器计算的成功状态。

真值只能在控制器外部用于训练教师和计算评测指标。

必须随机化：

- 摩擦系数；
- 质量和惯量；
- 接触刚度与阻尼；
- 关节和触觉噪声；
- 控制延迟；
- 初始抓取和粗对准状态。

## 当前项目进展

> V2 更新（2026-08-22）：当前主线已切换到 `METHODOLOGY_V2.md`。下方 P3/P4 检索路线保留为 V1 历史记录，不再决定当前开发顺序。V2 已完成无 Oracle 四状态控制核心、连续三相机 pooled-CLIP 失败基线、82 条 episode 的实例分割＋四关键点空间监督，以及读取 RGB 的轻量空间模型。冻结 test 表明外部相机关键点已经稳定，腕相机仍有严重遮挡和长尾，因此下一步采用 ego 主估计、腕相机可靠度门控，不做三相机等权融合；模型尚未通过 P1，也未接入在线控制器。MuJoCo 投影、分割和真值可见性仍只允许离线训练与评测。

> V2 后续诊断：ego 二维关键点本身有效，但 ego 空间特征直接回归绝对五维状态在 validation 和横向修正方向上失败；即使加入本体与历史动作，test 横向方向同半平面正确率也只有 56.6%。该回归器已明确禁止接入控制。当前下一步改为 RGB 图像误差闭环＋安全小扰动动作—像素雅可比辨识，深度交给 guarded descent 和接触判断，而不是继续调绝对状态回归。

### 已完成：观测与评估基础

- 统一装配 primitive 表示；
- MuJoCo 双手接触和装配观测适配器；
- 手—物相对姿态、平移滑移和旋转晃动跟踪；
- 只读阶段监控器；
- OpenPI 每步 `retrieval_cerebellum_trace.jsonl` 输出；
- 抓取、搬运、对齐和插入失败阶段统计接口。

### 已完成：P1 传感器防火墙

- 新增不可变 `CerebellumSensorObservation`，只包含 46D 本体、双臂关节驱动力、八指外力代理、双腕六维力、图像和历史动作；
- 新增 `SimCerebellumSensorAdapter` 作为可信仿真边界，只读取机器人和触觉可映射通道；
- 原只读 monitor 已明确拆分为 `PrivilegedCerebellumEvaluator`，不参与在线控制；
- OpenPI 监控 episode 同时保存不含物体真值的 `retrieval_cerebellum_sensor_trace.jsonl`；
- 新增访问陷阱测试，读取 `xpos/xquat/xmat` 会立即失败；
- 真实 bimanual MuJoCo 烟雾测试确认可读取三路图像、双臂关节驱动力、八指外力和双腕六维力；
- 传感观测对象不包含物体位姿、工件—孔误差、primitive 或 `raw_env`。

### 已完成：P2A 仿真传感—教师对齐数据

2026-08-21 已完成 P2 数据格式、构建代码、82 条有效片段生成和全量审计。

本阶段为什么需要再次回放：

- 已有 `force_labels_20260812_current_replay` 包含八指外力和双腕六维力，但不包含 P1 接口要求的 14D 双臂关节驱动力；
- 46D 本体、44D 动作、时间戳和视频索引直接复用原 LeRobot 数据，不重新渲染或复制图像；
- 80 条确定性 episode 的工件—孔、手—物相对位姿和接触标签复用已有 `retrieval_cerebellum_geometry`；
- episode 74 和 93 存在跨次回放接触动力学差异，已改为在同一个 MuJoCo step 同时采集传感与 privileged 教师，避免错配；
- 后续从头构建时默认使用 `--teacher-source same-replay`，不再把不同回放的接触传感与教师标签拼接。

已实现：

- `sensor_replay.py`：从记录初始状态准确重放动作，通过 `SimCerebellumSensorAdapter` 采集纯传感观测；
- `estimation_data.py`：定义 `sensor_*` 与 `teacher_*` 显式分区的 parquet schema；
- `load_sensor_history(...)`：在线/推理侧只读取 `sensor_*` 列，不能返回教师真值；
- `build_estimation_dataset.py`：支持断点续跑、episode 筛选、完整索引校验、同回放教师采集和 geometry 片段自动恢复；
- `audit_estimation_dataset.py`：统计完整性、通道范围、零值比例、split 和旧 force replay 一致性；
- 图像不重复写入 parquet，通过 `episode_index + frame_index` 引用原三路 LeRobot 视频；
- 原审计 `segments.jsonl` 缺失时，从 82 个 geometry shard 的帧边界恢复有效片段，避免依赖临时输出目录。

全量数据与审计结论：

- 完成 82 条 episode、19,137 帧，split 保持 train 72、validation 6、test 4；
- 全局 `index` 唯一，全部 46D 本体、44D 历史动作、14D 双臂关节驱动力、24D 八指外力和 12D 双腕六维力均为有限值；
- 八指外力零值比例为 55.92%，说明当前成功片段中存在大量无指尖外力帧，后续窗口模型不能假设每根手指持续接触；
- 双臂关节驱动力范围为 `[-87.0, 70.16]`，双腕六维力整体范围为 `[-23.99, 19.60]`，仅作为当前仿真量纲审计，不直接作为真实硬件安全阈值；
- 与 2026-08-12 的 current force replay 比较，80/82 条逐元素完全一致；episode 74 和 93 出现跨回放接触差异，平均绝对差仍分别仅约 `8.29e-4` 和 `2.48e-3`，但已使用同回放教师重新导出，不能忽略为普通数值误差；
- 教师标签中 `peg_ok` 为 18,058 帧，`tray_ok` 为 19,101 帧，`insert_ok` 为 83 帧；当前数据主要覆盖抓持、搬运、对准和首次孔口接触；
- `dataset_timestamp_s` 用于数据与视频对齐，`sensor_timestamp_s` 只是恢复初始状态后的局部仿真时钟，不能跨 episode 直接比较；
- 当前数据仍是仿真传感代理，真实部署前必须标定关节力矩、腕部 F/T 和真实触觉之间的单位、坐标系、带宽、噪声与延迟。

本阶段产物目录：

```text
/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly/
  retrieval_cerebellum_estimation/
```

禁止后续 Agent 重复的工作：

- 不再重新生成 82 条 geometry sidecar；
- 不再用旧 force parquet 冒充完整 P2 输入，因为其缺少 14D 双臂关节驱动力；
- 不复制三路视频像素到新 parquet；
- 不允许在线估计器直接读取任何 `teacher_*` 列；
- 不允许对 episode 74 和 93 使用不同回放的传感与教师标签；当前 P2 shard 已同步修复；
- 全量构建中断时读取 `retrieval_cerebellum_estimation/checkpoint.json` 并使用默认 `--resume` 继续，不从头回放。

### 已完成：P2B-A 可实现传感契约与仿真压力模型

仓库当前只能确认仿真机器人为双 Panda/Franka 类 7DoF 机械臂、双 Allegro Hand、双腕 MuJoCo F/T 和三路相机。仓库中没有真实机器人驱动、真实腕部 F/T 型号、指尖触觉型号、taxel 布局、设备采样率或硬件时间同步配置，因此不能把任何数值 profile 宣称为真实硬件标定结果。

已实现：

- `real_sensor_model.py` 定义显式 `SensorModelConfig`，包含采样率、延迟、噪声、偏置漂移、量化、饱和、丢帧和随机种子；
- `RealisticCerebellumObservation` 不再暴露 24D 世界系指尖外力，而只暴露每指力幅值、接触位和有效性掩码；
- 双腕六维力从仿真世界系转换到各自腕部传感器局部坐标系；
- 丢帧使用 hold-last 数值配合显式 validity mask，估计器不得把保持值误认为新测量；
- `require_hardware_verified()` 会拒绝未标定 profile，防止用仿真假设宣称真实部署就绪；
- `sim_stress_v1.json` 提供一套仅用于鲁棒性消融的未验证压力测试参数；
- `audit_sensor_model.py` 可将 profile 运行在全部 P2A shard 上并生成审计结果。

`sim_stress_v1_unverified` 全量审计：

- 输入 82 条、19,137 帧；1 帧/episode 延迟对应 30Hz 下约 33.3ms，产生 82 个窗口预热帧，输出 19,055 帧；
- 降级后通道为 46D 本体、14D 关节驱动力、8D 指尖力幅值、8D 接触位和 12D 腕部局部六维力；
- 指尖接触位总体激活率为 41.05%；
- 本体、关节驱动力、指尖触觉和腕部 F/T 的有效率分别为 99.91%、99.56%、99.01% 和 99.48%；
- 所有降级后数值均有限；profile 审计明确输出 `deployment_ready: false`；
- 产物为 `retrieval_cerebellum_estimation/sensor_model_sim_stress_v1_unverified_audit.json`。

当前科学结论：

- 当前 P1 的 24D `cfrc_ext` 只能作为仿真教师/代理，不能作为真实估计器的直接输入假设；
- 在未确定触觉硬件前，P3 可使用每指力幅值、接触位、腕部局部 F/T、关节驱动力和本体历史开发仿真基线；
- 精确“接触点位于 CAD 表面”和“接触法向一致”因子尚缺真实可观测接触位置与法向，不能直接按项目愿景中的理想触觉接口实现；
- 若真实硬件只有电机电流或单点接触开关，P3 必须降级为指尖运动学位置＋接触位/力幅值因子；若有阵列触觉，才可恢复 taxel 接触中心和局部法向因子。

禁止后续 Agent 误用：

- `sim_stress_v1_unverified` 不是硬件规格，也不是论文最终噪声参数；
- 不得训练或评测一个读取世界系 24D `cfrc_ext` 的“真实部署模型”；
- 不得把丢帧后的 hold-last 数值当成有效新测量，必须使用 validity mask；
- 不得在缺少接触位置/法向观测时实现依赖精确 taxel 几何的 MHE 因子。

### 已完成：P3A 纯传感可观测性与随机游走 MHE 基线

本阶段不是最终状态估计器，而是回答一个先决问题：仅使用当前真实可实现的本体、动作历史、关节驱动力、每指力幅值/接触位和腕部局部 F/T，是否已经足以估计三个 SE(3) 隐状态。

信念状态为 18D：

- 工件相对孔：位置 3D＋旋转向量 3D；
- 工件相对右掌：位置 3D＋旋转向量 3D；
- 托盘相对左掌：位置 3D＋旋转向量 3D。

已实现：

- `belief_estimation.py`：143D 单帧纯传感特征、因果历史堆叠、标准化多输出 ridge 观测模型、有限滑窗随机游走 MHE、误差与协方差校准指标；
- `train_belief_baseline.py`：固定使用 train 72、validation 6、test 4 的 episode split，教师真值只参与训练目标和外部评测；
- 默认输入为 3 帧历史，共 429D，不使用图像、不读取世界系指尖力、不在推理时读取 `teacher_*`；
- 输出观测模型、各 split 结果和 MHE 可行性判断。

实验结果：

- stress profile 单帧观测模型在 test 上的三类位姿平均误差为约 `4.28cm / 0.363rad`；
- 默认 8 帧随机游走 MHE 恶化为约 `5.13cm / 0.367rad`，协方差也明显过度自信；
- 将过程标准差放宽到 `5cm / 0.5rad` 后，MHE 退化为接近单帧模型，test 约 `4.28cm / 0.362rad`，仍无实质收益；
- 去掉全部噪声、延迟、量化和丢帧的 `sim_ideal_common_v1_unverified` 上界，test 仍约 `4.25cm / 0.366rad`；
- 三个状态均未达到暂定 `5mm / 5deg` 控制目标。

当前结论：

- 当前主要瓶颈不是 stress profile 的噪声，而是信息缺失和跨 episode 抓持姿态泛化；
- 抓持后仅靠腕部/关节运动和聚合力幅值，无法唯一恢复未知手—物相对位姿；
- 没有接触位置、接触法向或弱视觉初始化时，随机游走 MHE 只能平滑错误观测，不能创造缺失信息；
- 当前 ridge 模型只是可观测性下界，不能证明所有非线性观测模型都失败，但无噪声 common-channel 上界仍为厘米级，已经足以否定“直接调 MHE 参数即可进入精密控制”的路线。

冻结与下一步：

- 当前随机游走 MHE 保留为失败 baseline，不继续调窗口、过程噪声或 ridge 正则；
- 下一步 P3B 必须增加真实可实现的新信息：优先计算编码器＋机器人模型得到的指尖运动学位置，并加入腕部/ego 弱视觉初始化；
- 只有新增信息使 test 上手—物位姿误差显著下降后，才实现 CAD 表面、接触法向、无滑移和刚体兼容因子；
- 若加入指尖运动学和弱视觉后仍无法接近控制精度，应触发项目中的状态不可观测停止条件。

实验产物：

```text
outputs/retrieval_cerebellum/belief_baseline/
outputs/retrieval_cerebellum/belief_baseline_ideal/
outputs/retrieval_cerebellum/belief_baseline_loose_mhe/
```

### 已完成：P3B-A Allegro 指尖运动学接触代理

已实现 `finger_kinematics.py`：

- 直接加载装配 MJCF，不启动渲染环境；
- 只读取 state46 中的双手 32D Allegro 关节编码器；
- 使用 MuJoCo 正向运动学计算左右手各四根指尖相对各自掌心的 3D 位置，共 24D；
- 输出顺序统一为左右手各 `食指、中指、无名指、拇指`；
- 不读取工件、托盘、孔、接触点或任何 MuJoCo 物体真值；
- 该特征可在真实机器人上由同一 URDF/MJCF 和关节编码器计算。

固定 split 消融结果：

- 原 stress 单帧观测模型 test aggregate：`4.277cm / 0.363rad`；
- 加入 24D 指尖掌心坐标后：`4.214cm / 0.376rad`；
- 工件—右掌位置从 `5.083cm` 小幅改善到 `4.910cm`；
- 托盘—左掌位置从 `2.553cm` 改善到 `2.252cm`；
- 工件—孔位置反而从 `5.196cm` 变差到 `5.481cm`，旋转从 `0.389rad` 变差到 `0.466rad`。

结论：

- 指尖运动学是后续接触因子的必要几何输入，但它是手指关节的确定性变换，不包含新的对象观测；
- 它能改善部分手—物位置回归，却不能解决未知抓取时的物体初始位姿和旋转歧义；
- 不继续通过增加更多关节多项式、历史长度或 ridge 正则来冒充可观测性提升；
- P3B 下一项必须接入已有 ego/腕部视频的弱对象初始化。

产物：

```text
outputs/retrieval_cerebellum/belief_baseline_kinematics/
```

### 已完成：P3B-B 三相机弱视觉初始化 baseline

本阶段只验证“已有图像是否提供新的 episode 初始状态信息”，不把视觉作为精密闭环真值来源。

已实现：

- `visual_initialization.py`：根据 LeRobot episode metadata 将片段首帧映射到 packed MP4 的准确时间戳，读取 ego、左腕和右腕三相机 RGB；
- 使用本机离线 `openai/clip-vit-base-patch16` 权重提取每相机 768D 全局特征，不联网下载；
- 三相机共 2304D 原始特征只在 train 72 条 episode 上拟合 PCA，validation/test 不参与均值或主成分计算；
- `build_visual_initialization.py`：为全部 82 条有效片段生成首帧视觉缓存，不重放 MuJoCo、不复制视频、不读取 `teacher_*`；
- `train_belief_baseline.py --visual-cache ...`：将固定低维视觉先验追加到因果传感历史之后，避免在每个历史帧重复扩维；
- 新增缓存 split 一致性、PCA 和 CLIP 预处理测试。

固定 split 诊断消融：

- P3B-A 指尖运动学 baseline：test 单帧 aggregate `4.214cm / 0.376rad`；
- 32D PCA、ridge `alpha=10` 发生明显 episode 过拟合，退化到 `5.399cm / 0.498rad`；
- 4D PCA、`alpha=100` 改善到 `3.832cm / 0.361rad`；
- 当前最佳诊断配置为 4D PCA、`alpha=1000`：`3.612cm / 0.328rad`，相对 P3B-A 位置改善约 `14.3%`、旋转改善约 `12.6%`；
- 随机游走 MHE 仍把最佳单帧结果恶化到 `4.601cm / 0.336rad`，继续冻结为失败对照。

当前结论：

- 三相机首帧确实提供了纯本体/聚合触觉之外的新信息，P3B-B 成功跨过“新增观测必须改善 test”的最低门槛；
- 全局 CLIP 特征主要编码场景和粗姿态，样本仅 72 条时高维特征会记忆 episode，必须保持极低维和强正则；
- 当前约 `3.6cm / 0.33rad` 的结果不能直接做视觉伺服，但足以作为接触前的粗检索初始化；不再把 `5mm / 5deg` 完整姿态误差作为进入下一阶段的硬门槛；
- 本轮使用固定小 test split 做诊断消融，结果不是无偏最终模型选择；后续建立更多扰动 episode 后必须重新冻结独立 test；
- 当前停止继续堆叠全局视觉或追求完整姿态回归；已有粗信念直接用于成功技能检索与轨迹变形原型。

产物：

```text
outputs/retrieval_cerebellum/visual_initialization/
outputs/retrieval_cerebellum/belief_baseline_kin_visual_pca4_a1000/
```

### 已冻结：工程 baseline

- PBVS/固定 hybrid insert 只作为传统对照；
- asymmetric grasp assist 只作为失败 baseline；
- 手工距离门控、冷却帧和闭合量不属于创新主线；
- 不继续增加阈值或状态机补丁。

### 已完成：成功示范数据

2026-08-21 数据审计结果：

- 总示范：100 条；
- 有效“双方抓起后到首次插孔接触”片段：82 条；
- 有效帧：19,137；
- 未观察到插孔接触：17 条；
- 未确认双方抓起：1 条。

完整 geometry sidecar 已生成：

```text
/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly/
  retrieval_cerebellum_geometry/
```

对象中心记忆数据已生成：

```text
/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly/
  retrieval_cerebellum_learning/
```

episode 划分：

- train：72；
- validation：6；
- test：4。

当前片段截止到首次插孔接触，适合抓持、搬运、对准、接触初始化和成功技能检索原型。

### 已完成：检索图候选 baseline

成功接触流形图已实现并完成真实数据构建：

- 训练 episode：72；
- 图节点：4,188；
- 终点节点：73；
- 可达节点：4,188；
- 时序边：4,116；
- 跨示范检索边：15,956。

该图只保留为检索、可达性和初始化基线，不是最终控制器，也不意味着主线必须使用图优化。

### 已完成：核心检索—变形 shadow 原型

2026-08-21 已打通第一版创新主线，不再等待姿态估计达到毫米级：

`纯传感信念 → train 成功技能检索 → 当前双手状态对齐 → 数据驱动步长投影 → 32 步短动作建议`

已实现：

- `skill_prototype.py`：将当前 18D 粗信念转换为已有 14D RAM 技能查询；
- 仅使用 72 条 train 成功轨迹作为技能库，validation/test 不进入 gallery；
- 检索后不直接回放，而是把左右腕部和手指轨迹对齐到当前本体状态；
- 平移、旋转和手指逐步变化上限从 train 成功轨迹的运动分布自动估计，不按当前孔径手工设置毫米/角度阈值；
- `run_skill_prototype.py`：在 6 条 validation 和 4 条 test episode 上运行完整离线 shadow 链路并保存计划。

原型结果：

- 10/10 held-out episode 均检索到 train 成功技能并输出 32 步变形计划；
- 直接回放检索轨迹时，首步双腕最大位置跳变平均约 `26.8cm`；
- 经当前状态对齐和数据驱动投影后，平均降为约 `1.18cm`；
- 当前只证明核心数据流和“检索后必须变形”成立，尚未在 MuJoCo 中执行这些动作，因此不宣称插入成功率提升。

产物：

```text
outputs/retrieval_cerebellum/skill_prototype/
```

### 已实现：RC-HB-SQP 数值核心

2026-08-21 已实现 `belief_space_sqp.py` 的第一版对象级连续子问题：

- 输入五维工件—孔信念、联合双侧附着协方差和 Top-K 检索技能候选；
- 同时优化左右腕部局部关键点修正，不假设任一侧抓持固定；
- 双侧附着过程协方差通过联合映射传播到装配状态机会约束；
- 显式检查孔壁径向包络、插入深度单调性、终端深度和双侧抓持扳手裕度；
- 对每个检索接触模式运行模式固定的 SLSQP 子问题，并选择代价最低的可行候选。

最小数值烟雾已验证：检索距离更近但右侧抓持超载的候选被判不可行，求解器选择双侧抓持与孔壁约束均可行的候选并达到目标深度。该结果只验证优化问题和候选选择逻辑，尚未接入真实技能库、在线信念和 MuJoCo 动作执行，因此不代表插入成功率。

### 已实现：真实双腕装配雅可比与 44D 动作映射

2026-08-21 已移除 RC-HB-SQP shadow 中的 `I/-I` 状态映射占位：

- 新增 `assembly_kinematics.py`，显式维护右掌—工件、左掌—孔和工件尖端偏置组成的 SE(3) 双侧附着链；
- 从当前 44D 本体状态、右手—工件附着和工件—孔相对状态重建局部装配运动学；
- 对左右腕部分别计算真实 `5×6` 世界系 twist 雅可比，包含腕部旋转对工件尖端横向位置和深度的耦合；
- SQP 左右腕控制变量由每侧 5D 占位量改为每侧 6D twist，检索技能的名义控制直接来自成功示范双腕 SE(3) 增量；
- 优化控制量可积分为 44D 绝对双腕目标，同时保留检索成功技能的双手手指目标；
- 每条验证 episode 保存 8 步 `action44`、左右雅可比、优化控制和一步非线性/线性状态对照。

已在原 10 条 held-out episode 上完成 oracle shadow 复验：

- episode 级 Top-K 成功率：`10/10`，其中 validation 6 条、test 4 条；
- 40 个候选子问题中 20 个正常收敛、23 个满足数值可行容差，Top-K 为每条 episode 保留至少一个成功候选；
- 拼接双腕雅可比条件数范围为 `1.281–1.439`，中位数约 `1.342`；
- 完整非线性 SE(3) 一步状态与雅可比线性预测的 L2 误差中位数约 `5.25e-6`，最大约 `3.16e-5`；
- 生成计划的最大单步腕部平移约 `4.0mm`，最大单步旋转约 `0.020rad`。

产物：

```text
outputs/retrieval_cerebellum/belief_space_sqp_real_jacobian_validation10/
```

该结果仍使用交接前最后一帧 privileged 装配信念，仅验证“真实局部运动学＋检索条件化机会约束优化＋44D 目标生成”的上界。腕力响应雅可比仍为零，显式接触力平衡、摩擦锥和 MuJoCo 实际执行尚未完成。

### 验证状态

- `retrieval_cerebellum` 测试：68 passed；
- Python 编译通过；
- `git diff --check` 通过；
- 82 条 geometry 回放和 82 条 P2 传感回放完成；
- 19,137 帧对象中心记忆与传感—教师对齐数据均已写盘；
- P2 全量审计写入 `retrieval_cerebellum_estimation/audit.json`。

## 代码与数据入口

核心模块：

```text
retrieval_cerebellum/
  primitives.py              装配 primitive 表示
  observer.py                privileged 观测与滑移跟踪
  geometry_labels.py         离线真值标签
  assembly_kinematics.py     双侧附着 SE(3) 状态与真实 5x6 腕部雅可比
  geometry_store.py          geometry parquet
  demo_segments.py           抓后示范切分
  learning_data.py           对象中心记忆表示
  contact_manifold.py        检索图候选 baseline
  sensor_observation.py      真实可部署的纯传感观测
  sim_sensor_adapter.py      仿真传感边界，不读取物体真值
  sensor_replay.py           准确重放并采集纯传感通道
  estimation_data.py         P2 传感—教师对齐格式与纯传感窗口读取
  real_sensor_model.py       可实现传感契约与噪声/延迟/丢帧模型
  belief_estimation.py       P3A 观测模型、滑窗 MHE 和评估指标
  finger_kinematics.py       编码器驱动的 Allegro 指尖掌心坐标
  visual_initialization.py   P3B-B 三相机首帧、CLIP 与 train-only PCA
  skill_prototype.py         成功技能检索与数据驱动轨迹投影原型
  belief_space_sqp.py        双侧附着不确定性下的机会约束 SQP 核心
  monitor.py                 privileged 真值评测器
```

脚本：

```bash
python -m retrieval_cerebellum.scripts.audit_post_grasp_demos
python -m retrieval_cerebellum.scripts.label_geometry
python -m retrieval_cerebellum.scripts.build_learning_dataset
python -m retrieval_cerebellum.scripts.build_contact_manifold
MUJOCO_GL=egl python -m retrieval_cerebellum.scripts.build_estimation_dataset
python -m retrieval_cerebellum.scripts.audit_estimation_dataset
python -m retrieval_cerebellum.scripts.audit_sensor_model
python -m retrieval_cerebellum.scripts.train_belief_baseline
python -m retrieval_cerebellum.scripts.train_belief_baseline --include-fingertip-kinematics
python -m retrieval_cerebellum.scripts.build_visual_initialization --pca-dim 4
python -m retrieval_cerebellum.scripts.train_belief_baseline --include-fingertip-kinematics --visual-cache outputs/retrieval_cerebellum/visual_initialization/clip_vit_b16_pca4.npz --ridge-alpha 1000
python -m retrieval_cerebellum.scripts.run_skill_prototype
```

主要产物：

```text
outputs/retrieval_cerebellum/post_grasp_demo_audit/
outputs/retrieval_cerebellum/contact_manifold/
/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly/retrieval_cerebellum_geometry/
/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly/retrieval_cerebellum_learning/
/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly/retrieval_cerebellum_estimation/
outputs/retrieval_cerebellum/belief_baseline/
outputs/retrieval_cerebellum/belief_baseline_kinematics/
outputs/retrieval_cerebellum/visual_initialization/
outputs/retrieval_cerebellum/belief_baseline_kin_visual_pca4_a1000/
outputs/retrieval_cerebellum/skill_prototype/
```

## 下一步实现顺序

当前原型优先顺序为：

`P4 核心 shadow 原型 → P5 MuJoCo 短片段执行 → P6 接触反馈重规划与评估`

**当前执行点是 P5。** P4 已证明传感信念可以驱动成功技能检索，并且检索轨迹能够通过真实双侧 SE(3) 雅可比在当前双手状态周围受限优化并生成 44D 目标。下一步直接在 MuJoCo 中执行极短片段，检查动作有效性、机器人运动学可行性和安全性；不继续优化完整视觉姿态，也不建设失败恢复系统。

### P1：传感器防火墙（已完成）

- `CerebellumSensorObservation`、仿真适配器和纯传感 trace 已接入；
- 在线数据结构不包含 MuJoCo 物体真值；
- privileged evaluator 与传感路径已分离；
- 后续在线估计器只能接收该传感观测类型。

### P2A：仿真传感—教师对齐数据（已完成）

- 已完成 82 条、19,137 帧完整传感—教师对齐数据；
- 已完成纯传感窗口读取器、全量通道审计和跨回放一致性检查；
- 在线估计器只能通过 `load_sensor_history(...)` 读取 `sensor_*` 列；
- episode 74 和 93 已使用同回放教师修复接触动力学不确定导致的潜在错配。

### P2B-A：可实现传感契约与仿真压力模型（已完成）

- 已将世界系指尖三维力降级为每指力幅值和接触位；
- 已将腕部 F/T 统一到腕部传感器局部坐标系；
- 已实现延迟、噪声、漂移、量化、饱和、丢帧和 validity mask；
- 已完成 82 条、19,137 帧未验证 stress profile 审计；
- profile 被强制标记为 `hardware_verified: false` 和 `deployment_ready: false`。

### P2B-B：目标硬件标定（等待硬件信息）

- 明确真实机器人可提供的关节力矩、腕部 F/T 和指尖触觉型号与采样率；
- 定义各通道坐标系、单位、偏置、饱和、带宽和时间同步；
- 确定触觉是否提供接触位、力幅值、剪切力、接触中心和局部法向；
- 用静态加载、空载漂移、冲击和同步实验替换 `sim_stress_v1` 假设参数；
- 该项是进入真实部署前的门槛，不阻塞仿真原型。

### P3A：纯传感观测与随机游走 MHE（已完成，失败 baseline）

- 已完成 18D 隐状态的纯传感历史 ridge 观测模型；
- 已完成 8 帧有限滑窗随机游走 MHE；
- stress 和无噪声 common-channel 上界均为厘米级/约 0.36rad；
- 随机游走 MHE 不改善 test 误差，冻结为失败对照。

### P3B-A：指尖运动学接触代理（已完成）

- 从 Allegro 编码器和机器人模型计算八指指尖在腕部/世界坐标系的位置；
- 将指尖位置、接触位和力幅值组成可实现的接触位置代理；
- 固定 test split 仅得到小幅位置改善且旋转变差，不能解决对象初始位姿不可观测问题。

### P3B-B：弱视觉初始化（已完成）

- 已从 ego/双腕视频片段首帧提取三相机 CLIP 特征，并仅用 train episode 拟合低维 PCA；
- 4D PCA＋强 ridge 正则将 test 单帧 aggregate 改善到 `3.612cm / 0.328rad`；
- 高维全局特征明显过拟合，当前只保留为接触前的粗初始化和技能检索特征。

### P3B-C：对象中心视觉跟踪（非当前主线）

- 不再将毫米级完整视觉位姿作为后续阶段的前置门槛；
- 只有当前粗信念导致技能检索持续失败时，才增加 CAD/稠密特征视觉跟踪；
- 若启用，视觉模块只能读取 RGB、相机标定和离线模板，不能读取 MuJoCo 对象真值。

### P4：核心检索—变形 shadow 原型（已完成）

- 当前 P3 信念直接查询 72 条 train 成功技能；
- 检索技能按当前左右腕部和手指状态进行坐标对齐；
- 使用成功数据分布学习的步长上限做投影式约束；
- 已在 10 条 held-out episode 输出 32 步 shadow 计划；
- 已将 RC-HB-SQP 双腕控制升级为真实 `5×6` 局部装配雅可比，并在 10 条 held-out episode 输出 8 步 44D 优化目标；
- 不直接回放、不读取在线真值、不生成恢复技能。

### P5：MuJoCo 短片段执行（下一步）

- 从 π0.5 到达孔口前的真实回放状态启动；
- 每次只执行检索—变形计划的少量动作，不一次执行完整轨迹；
- 执行前使用 MuJoCo `mj_jacSite` 检查双臂关节速度、奇异性、关节限位和末端目标可达性；
- 监控抓持、接触力、插入进度和异常终止条件；
- 首先验证动作不会造成大跳变、掉落或明显远离孔口；
- 与直接回放、平均成功轨迹和无检索投影进行对照。

### P6：接触反馈重规划与闭环评估

- 执行短片段后重新读取传感信念并重新检索或变形；
- 若异常力、无进展或滑移超过安全条件，则立即停止并交还 π0.5；
- 闭环只在成功插入技能的局部有效域内重规划，不把失败状态扩展成恢复任务；
- 不生成恢复数据，不建立恢复技能库，也不实现学习式失败恢复；
- 与纯 π0.5、PBVS baseline、无检索优化和 privileged 上界比较。

## 主要评估指标

### 状态估计

- 横向和倾斜修正方向正确率；
- 插入进度误差；
- 滑移和接触模式识别准确率；
- 抓持稳定性和掉落风险识别；
- 遮挡、噪声和动力学扰动下的稳定性；
- 估计协方差校准误差；
- 完整工件—孔和手—物姿态误差作为辅助诊断，不作为唯一通过标准。

### 检索与优化

- 当前信念检索到成功技能的比例；
- 检索初始化相对随机/最近邻 baseline 的优化收益；
- QP/SQP 求解成功率和耗时；
- 技能变形后的抓持保持率；
- 短片段执行后的孔口接近和插入进度。

### 端到端

- 双手抓持后到达孔口比例；
- 完整插孔成功率；
- 物体掉落率；
- 未见抓取和粗对准状态下的成功率；
- 对摩擦、质量、刚度、噪声和延迟的鲁棒性；
- 相比无检索、无信念估计和 privileged 上界的差距。

## 关键消融

- 无 RAM 接触模板；
- 无技能检索，只使用平均轨迹；
- privileged 真值状态；
- 弱视觉＋本体，不使用触觉；
- 触觉＋本体，不使用弱视觉；
- 单帧估计、轻量滤波与约束 MHE；
- 直接回放检索轨迹与约束轨迹变形；
- 固定最近邻与信念条件检索；
- 不使用状态协方差的确定性优化。

## 风险与停止条件

主要风险：

- 当前仿真接触信号与真实硬件触觉不一致；
- 抓持滑移时手—物姿态不可观测；
- 固定圆柱几何存在旋转对称歧义；
- 检索到的成功轨迹经变形后仍可能不适合当前状态；
- 约束投影过强会失去示范进度，过弱会造成动作跳变；
- 82 条数据只覆盖首次接触前，完整插入知识不足。

停止或改线条件：

- 隐藏 MuJoCo 真值后，传感信念检索不能比纯本体最近邻更好地选择成功技能；
- 检索先验不能提升局部优化成功率；
- 完整系统收益仅来自 privileged 信息或手工阈值。

## 非目标

- 不把 DexJoCo、OpenPI 和 RAM 简单拼接作为创新；
- 不重新训练或修改 π0.5；
- 不给 π0.5 动作添加残差；
- 不依赖真实部署不可获得的物体位姿真值；
- 不把 PBVS、固定门控或手工插入状态机作为主线；
- 不直接回放检索轨迹；
- 不为了复杂而堆叠卡尔曼、粒子滤波、图优化、MPC或生成模型；
- 只有在对应子问题和消融结果支持时才采用具体算法。

## 后续 Agent 交接原则

后续工作必须首先阅读本文件，并遵守：

1. π0.5 固定不变；
2. 正式控制不能读取 MuJoCo 物体真值；
3. 当前粗信念已经足够驱动核心原型；优先完成检索、变形、短片段执行和闭环重规划；
4. 现有图只是一项候选 baseline；
5. 本项目不建设恢复数据集、恢复技能库或学习式失败恢复系统；安全停止和交还 π0.5 不是研究创新；
6. 新模块必须说明解决的具体子问题和对应消融；
7. 若仅增加阈值、冷却帧或硬编码状态机，应停止并重新评估创新价值。

## V2 当前证伪结论（2026-08-22）

held-out episode 17/23 的 RGB 小扰动实验表明，四维投影雅可比分别只有秩 `3/4` 和 `2/4`。episode 23 的有界命令虽然降低 RGB 投影误差，却将真实横向误差从 `3.5067 mm` 恶化到 `3.9136 mm`。因此纯单目图像误差伺服正式停止，不再调增益、探测幅度或网络。

失败实现统一记录在 `EXPERIMENT_LOG_V2.md`，不再反复改写核心方法论。下一主线只允许二选一：带 CAD/尺度/深度证据的度量视觉，或视觉限定范围后的 guarded descent＋力控局部搜索。

## V2 接触控制原型进展（2026-08-22）

已完成固定分配、单臂和双手动态约束释放的无 Oracle 连续短闭环。episode 17 的困难扰动中只有双手动态分配成功；episode 23 中三者均成功，但双手版本少 1 步且峰值腕力残差降低约 43%。这与“双侧抓持闭链的模式门控与约束释放”创新主线一致。

当前不能宣称接触响应搜索成功，因为有效 rollout 都没有进入微探测状态。下一实验只构造和冻结 rim-contact 初态，验证动态双手分配相对固定分配的插入率与内力收益；不能稳定复现 rim contact 或没有跨状态收益时立即停止。

## V2 自然 Handoff 普查（2026-08-22）

后续纠正了上述人工 rim-contact 计划，直接在 9 个自然 held-out handoff 上评测。固定直线 guarded descent 只有 `2/9`；失败主要是丢失 π0.5 后续时变双腕意图后产生的平移和倾斜漂移。使用记录的双腕动作增量作为 π0.5 handoff action chunk 代理后达到 `8/9`，唯一失败是意图序列提前耗尽。

因此当前创新原型不再定义为独立视觉找孔器或螺旋搜索器，而是：π0.5 在控制权交接时输出短时双腕意图，独立小脑使用本体、腕力、指尖信号和后续视觉完成重定时、约束释放、柔顺执行、安全退出和成功判断。记录 action chunk 只是离线可行性上界，下一步必须替换为真实 π0.5 在线输出后重新验收。

已生成 episode 31 的 post-handoff 左右诊断视频：`outputs/retrieval_cerebellum/natural_handoff_baseline/episode_000031_fixed_vs_intent.mp4`。它从已经接近孔口的冻结 handoff 开始，不包含搜孔过程、在线视觉闭环或真实 π0.5 rollout；右侧使用的是同 episode 记录的后续动作代理。因此该视频不能称为找孔原型，只能用于展示固定轴压缩时变双腕意图后的失败。生成脚本为 `retrieval_cerebellum/scripts/render_natural_handoff_comparison.py`。

## V2 真实 Action Chunk 接口（2026-08-22）

在线 OpenPI 客户端现可接收策略响应中的显式 `handoff=true`，冻结该次真实双腕 action chunk，并暂停 π0.5 重规划。自然 handoff runner 也新增 `online_pi05_chunk` 模式：教师只用于恢复自然实验初态，初态后立即向真实 π0.5 服务请求新 chunk，不再读取记录后续动作。独立小脑执行器仅用本体与力传感对 chunk 进行重定时、单步限幅和硬力退让。

本机没有运行中的 π0.5 服务，因此不能报告自然 episode 在线成功率。下一步先用标准服务跑冻结自然 handoff 的真实 chunk 对照，确认能否接近记录代理 `8/9`；随后再补齐上游显式交接信号做完整 rollout。不能为了得到结果改用记录后续动作或真值自动触发。

上述真实 chunk 对照已经完成：9 个自然 held-out handoff 达到 `8/9`，与记录动作代理一致。失败 episode 17 没有触发硬力退出，而是在 100 个受限控制周期后仍未完成并出现倾斜恶化。由此可将主线从“记录动作可回放”提升为“真实 π0.5 在线短时双腕意图可被独立小脑安全执行”；但自动 handoff、可部署成功判断和接触模式修正仍未完成，因此不能宣布完整方法论或 P2 已实现。
