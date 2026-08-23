# DexContactRAM V2 实验日志

最后更新：2026-08-23

本文件只记录可证伪实现假设、实验结果和路线决策。稳定问题定义、在线输入边界、控制状态和验收门槛保留在 `METHODOLOGY_V2.md`，不再随单次模型实验反复改写。

## E1：Pooled CLIP 直接回归五维状态

- 结果：孔口局部窗口横向 P90 `31.7 mm`、倾斜 P90 `0.159 rad`、深度 P90 `42.1 mm`。
- 结论：失败。全局语义 embedding 丢失精密空间结构，禁止接入控制器。

## E2：三相机共享网络等权融合

- 结果：ego 关键点有效，但腕相机受遮挡和样本不平衡影响，整体关键点 P90 为 `16.61` 个输出像素。
- 结论：失败。三相机不得等权融合；ego 为主视角，腕相机必须可靠度门控。

## E3：Ego RGB＋本体直接回归绝对五维状态

- 结果：local test 为 `4.36 mm / 0.137 rad / 18.8 mm`，但 validation 深度 P90 `35.5 mm`；test 横向修正同半平面正确率 `56.6%`，45 度内仅 `32.1%`。
- 结论：失败。禁止用跨 episode 黑盒回归器生成世界系纠偏方向。

## E4：单目 RGB 图像误差＋在线像素雅可比

实现：

- `image_space_servo.py`：RGB-only ego 关键点推理、图像误差、中心差分雅可比和阻尼最小二乘控制；
- `run_image_space_servo_probe.py`：只向控制器提供 ego RGB 与已执行小扰动，MuJoCo 几何只在控制器外部评测；
- `test_image_space_servo.py`：覆盖图像误差、雅可比辨识、降误差命令和秩亏拒绝。

冻结 test episode 结果：

- episode 17、frame 690：四维位置＋轴向雅可比秩为 `3/4`，完整单目姿态伺服不可观；二维孔口中心子系统条件数 `38.65`。一次 `[-2, +2] mm` 有界命令令 RGB 总误差下降约 `2.6%`，外部横向误差仅从 `3.3799 mm` 到 `3.3794 mm`；
- episode 23、frame 729：四维雅可比秩为 `2/4`，二维孔口中心子系统条件数 `2.71`。同方向命令令 RGB 总误差下降约 `0.54%`，但外部横向误差从 `3.5067 mm` 恶化到 `3.9136 mm`，倾斜也从 `0.0786 rad` 恶化到 `0.0831 rad`。

结论：**失败并停止。** 单目图像中工件端点与孔口中心更重合，并不保证三维横向误差更小；两者深度不同会产生透视视差，局部像素雅可比只能优化投影误差，不能作为安全物理纠偏目标。不得继续调阻尼、增益、网络或探测幅度来掩盖该几何不可观测性。

后续只保留两条理论上成立的选择：

1. 使用已标定相机、CAD 尺度/轮廓和可部署深度证据恢复度量几何，再做物理空间纠偏；
2. 视觉只限定孔口局部搜索区域，由 guarded descent、腕力/触觉和小范围接触搜索完成最终找孔。

在新的度量深度证据或接触搜索闭环原型出现前，不再恢复“纯单目图像误差伺服”支线。

## E5：双手约束释放与接触响应搜索原型

实现：

- `contact_response_search.py`：固定螺旋、单臂顺序微探测和双臂动态运动分配三种策略；
- `run_contact_response_search.py`：从冻结回放状态连续执行，只向控制器提供本体、历史动作、指尖力和腕力；
- MuJoCo 几何只由独立 evaluator 记录成功、横向误差、倾斜和接近高度。

快速对照结果：

- episode 17、frame 711、8 mm 已知动作扰动：固定 `0.8` 双手分配和单臂策略均在 120 步内漂离并失败；双手动态分配在 58 步成功，最终横向误差 `1.63 mm`、倾斜 `0.0037 rad`；
- episode 23、frame 750、8 mm 扰动：三种策略均成功；双手动态分配用 6 步完成，另外两种为 7 步，峰值腕力残差由约 `0.816 N` 降到 `0.467 N`；
- episode 23、frame 729 暴露出历史动作无法可靠提供带符号的孔轴方向，错误方向会令所有策略向孔外移动。approach axis 必须作为 π0.5 handoff 的显式可部署意图，或由通过门控的度量视觉提供，不能从任意短历史盲猜。

结论分为两部分：

1. **双手闭链约束释放值得继续一轮。** 当前结果支持根据双侧抓持稳定性和腕力动态分配相对运动，而不是固定 `0.8/0.2` 或只移动工件手；但样本仍不足，尚未达到 P2。
2. **接触响应辨识尚未被验证。** 所有有效 rollout 的腕力残差都低于当前接触门槛，执行阶段始终停留在 `descent`，顺序微探测和固定螺旋实际均未触发。不得把成功归因于接触搜索。

下一最短实验只验证一个问题：在可重复的真实 rim-contact 初态上，双手动态分配是否比固定分配降低闭链内力并提高插入率。若不能构造不依赖 Oracle 的 rim-contact handoff，或 10 个冻结状态中无稳定收益，则停止该支线。

上述 E5 人工扰动结果不能用于证明自然任务需要 rim-contact 搜索，也不能作为项目主线证据。后续自然 handoff 普查已经取代该计划；接触搜索模块仅保留为未验证候选 baseline。

## E6：自然 handoff 必要性普查

协议：

- 使用 9 个冻结 held-out 自然 handoff，不增加横向扰动、不制造卡边；
- 控制器不读取物体真值；MuJoCo evaluator 只负责外部成功判定和失败诊断；
- 首先运行固定名义轴、固定 `0.8/0.2` 双手分配、无搜索的 guarded descent；
- 随后保留同一 episode 记录的后续双腕动作增量，作为“π0.5 在 handoff 时提供短时 action chunk”的离线代理。

结果：

- 原始指尖接触计数门将 9 个自然 handoff 全部误判为抓持失稳，得到伪 `0/9`。外部几何表明抓持实际仍成立，因此该门控当前不可用；
- 移除错误抓持门后，固定直线 guarded descent 为 `2/9`；7 个失败主要表现为横向与倾斜逐步漂移，而不是稳定 rim contact；
- 保留时变双腕动作意图后达到 `8/9`。唯一失败 episode 17 是记录 action chunk 在 54 步后耗尽，未到达插入状态；
- episode 23 的峰值腕力残差达到 `11.46 N`，说明 action chunk 仍需要小脑的力安全门和柔顺执行，不能直接裸回放。

结论：**项目真实瓶颈不是“是否需要螺旋搜孔”，而是 handoff 后不能丢失 π0.5 的时变双腕接近意图。** 固定一根孔轴下探会将自然成功轨迹压扁并导致 `7/9` 失败；保留短时双腕意图后可恢复到 `8/9` 的离线代理上界。

这仍不是 P2 通过：action chunk 来自同 episode 记录，而不是实际 π0.5 在线输出；成功终止也仍由外部 evaluator 判定。下一原型必须接入真实 π0.5 action chunk，并由小脑仅使用可部署传感完成重定时、双手约束释放、力安全和成功/退出判断。

已生成 post-handoff 诊断视频：`outputs/retrieval_cerebellum/natural_handoff_baseline/episode_000031_fixed_vs_intent.mp4`。该视频从已经位于孔口附近的冻结 handoff 开始，**不包含视觉找孔、粗接近或在线 π0.5 rollout**。左侧为固定轴 guarded descent，右侧为同 episode 记录的成功示范后续动作代理。它只能说明保留时变动作意图的重要性，不能作为无 Oracle 找孔原型或正式创新系统证据。视频叠加的几何数值只用于离线评测展示，不进入控制器。

## E7：真实 π0.5 action chunk 在线接管接口

已将 OpenPI 在线推理返回的完整 `actions` chunk 接入评测，不再从同 episode 记录中取后续动作。完整端到端评测协议要求策略响应显式包含 `handoff=true`；客户端收到后冻结该次剩余 chunk、清空 π0.5 动作缓冲，并在 chunk 执行期间停止继续请求 π0.5。接管触发不读取 MuJoCo 物体位姿、教师 handoff 行或外部成功标签。

同时新增自然 handoff 冻结评测模式 `online_pi05_chunk`：教师 handoff 行只用于确定外部实验初态；恢复到该自然状态后，runner 立即向真实 π0.5 服务请求一个全新 chunk，随后控制器只读取真实 chunk 与可部署传感。该模式直接检验“记录动作代理换成真实 π0.5 输出”这一问题，但不检验端到端自动 handoff 选择，因此结果仍不能单独算 P2。

独立执行器只读取本体状态、指尖力、双腕力和冻结 chunk。它把 chunk 的双腕时变增量重定时到当前状态，限制单步平移和旋转；腕力超过软阈值时降低相位推进速度，达到硬阈值时沿 chunk 总体运动反方向执行一次有界退让。限幅通过缩小相位步长实现，不会在限幅后跳过未执行区段。

E7 接口实现当时只完成代码闭环和单元验证：新增 4 个执行器测试均通过，但尚未运行 π0.5 policy server，因此当时不能报告真实在线成功率。该缺口随后由 E8 的 `online_pi05_chunk` 自然 handoff 验证补齐；端到端显式 handoff 信号仍未验证。不能退回记录后续动作、教师控制或真值触发。

## E8：真实 π0.5 chunk 的自然 handoff 验证

已使用正式 `bimanual_assembly` checkpoint 启动临时 OpenPI server，并在与 E6 相同的 9 个自然 held-out handoff 上运行 `online_pi05_chunk`。每个 episode 只使用记录动作恢复到外部实验初态；handoff 之后的 30 步意图全部由当前状态的 π0.5 在线推理生成，不读取该 episode 的记录后续动作。测试结束后 server 进程和端口均已关闭。

结果为 `8/9`，与记录 action chunk 代理的 `8/9` 一致：episode 23、31、42、78、81、83、88、97 成功；episode 17 在 100 个小脑控制周期内未完成。成功 episode 峰值腕力残差范围约 `1.52–7.55 N`，均未达到 `20 N` 硬力阈值。episode 17 的横向误差由 `3.55 mm` 到 `3.36 mm`，但倾斜由 `0.0109 rad` 恶化到 `0.1053 rad`，接近高度只由 `60.2 mm` 降到 `36.4 mm`，属于真实 chunk 在当前单步限幅下未能于执行 horizon 内完成，而不是硬力退出。

结论：**“handoff 后必须保留 π0.5 时变双腕意图”这一核心假设得到真实在线 chunk 支持，不再只是记录轨迹代理现象。** 但这仍不是完整 P2：handoff 行由外部冻结实验指定，成功仍由外部 evaluator 终止；所有 episode 的错误指尖抓持门仍为关闭状态；峰值腕力未跨过软阈值，因此本轮没有验证力触发重定时或接触修正。下一门控只剩端到端显式 handoff、可部署成功/退出判断，以及 episode 17 的有限 horizon 执行失败归因。

## E9：非 Oracle 动作—视觉可靠度模型

为验证“PBVS 动作能否主动改善下一帧视觉估计”，新增双线性可靠度动力学模型。训练和运行输入严格限制为：ego RGB 冻结特征、当前感知可靠度、`sensor_previous_action44` 转换得到的双腕局部 twist。训练 pair loader 不读取任何 teacher 几何列，运行模块也不导入 simulator、privileged provider 或外部 evaluator。

第一次直接预测下一帧可靠度时，held-out test MAE 为 `0.01829`，差于“下一帧可靠度等于当前帧”的 persistence 基线 `0.01320`。改为只学习相对 persistence 的可靠度增量，并扩大 ridge 正则范围后，validation 选择 `alpha=10000`；test MAE 改善到 `0.01437`，但仍差于 persistence `0.01320`。

结论：**当前成功示范数据不足以辨识动作对可见性的因果影响，模型拒绝接入主动控制。** 训练产物明确写入 `approved_for_active_control=false`，运行时 `load_approved` 会拒绝加载，不能通过继续调权重或只报告训练集指标绕过门控。

为获得可辨识数据，新增 sensor-only 主动视角采样器。候选动作只包含横向平移和 roll/pitch 微扰，轴向平移与绕孔轴旋转均为零；腕力、腕矩或双手稳定指尖数量不满足门槛时不输出动作；手指命令保持不变。该采样器目前只完成单元验证，尚未运行新的自然交互采集，因此不能声称主动视觉已经有效。下一实验必须采集正负对称视角微扰及回到基准位姿的数据，再重新检验是否在独立 episode 上稳定优于 persistence。

随后完成 episode 17、frame 690 的无头仿真采集 smoke。采集器从同一回放基准分别执行两个 \(\pm1\,\mathrm{mm}\) 横向 probe，每个 probe 只执行一步；两条 transition 均通过传感器安全门，感知可靠度变化分别为 `-0.00281` 和 `-0.00305`。输出 NPZ 只包含前后 RGB 特征、前后可靠度、双腕控制和前后腕力，不包含几何标签、成功标签或接触真值。

这两条 transition 已成功接回训练器，证明“安全采集→transition 文件→动作可靠度训练”的工程闭环可运行；但加入两条数据后的 test MAE 仍为 `0.01434`，差于 persistence `0.01315`，因此产物继续保持 `approved_for_active_control=false`。两条 smoke 数据不能用于得出动作方向结论，正式辨识仍需要跨多个自然状态的正负对称 probe。

## E10：P1 硬门触发与主路线切换

冻结 ego 视觉状态模型在 held-out local test 上得到：横向 P90 `21.06 mm`、倾斜 P90 `0.2077 rad`、深度 P90 `30.55 mm`；预注册 P1 门槛分别为 `1.2 mm`、`0.03 rad`、`3 mm`。三项全部失败。按照“任一 P1 指标失败即停止”的新硬门，IEKF-PBVS、主动协方差优化和以完整相对姿态为前提的多指信念空间控制正式停止作为项目主线。

新增 `methodology_gate.py` 从冻结 summary 自动输出路线决定，接口不提供 override。当前路线切换为 `intent_preserving_sensor_compliance`：保留真实 π0.5 时变双腕 action chunk，只使用本体、腕力、多指力和动作历史进行相位重定时、双手运动分配、力安全、退让和可部署成功/退出判断。此前 IEKF-PBVS 推导保留为条件性候选档案，不再用于声称系统可行，也不再获得主线实验预算。

## E11：显式 handoff 的无 Oracle runtime 接线

已将 `OnlineIntentChunkExecutor` 封装为可审计的 `OnlineIntentChunkRuntime`，并接入 OpenPI 在线评测循环。runtime 独占维护上一条实际下发命令，视觉/本体采集器必须把该命令作为 `previous_action44` 回传；若调用方错误地用当前状态替代上一命令，runtime 会拒绝执行，避免跟踪门长期退化为恒等于一。相位推进现在联合记录腕力、抓持保持率、跟踪误差和左右手动态运动分配。

intent-only 模式不再实例化 `PrivilegedCerebellumEvaluator`。控制审计只声明 `state46`、双臂关节力矩、多指力、双腕力和上一动作五类输入。handoff 后策略 action buffer 被清空；chunk 完成、硬力退让、抓持失稳或跟踪误差停止后，runtime 先记录传感器终止原因，再显式记录 `policy_replan_requested`，不会把 `chunk_complete` 冒充插入成功。

新增 runtime 行为测试覆盖：handoff—完成—重规划顺序、错误上一命令拒绝、硬力安全退出不产生 success 事件、过短 chunk 不计为有效 handoff。当前 `retrieval_cerebellum/tests` 共 `169 passed`，客户端与 runtime 编译检查通过。

这仍不是 A0/A1/A2 实验通过：本轮没有启动真实 π0.5 policy server，也没有自然 `env.reset()` 端到端 rollout。环境 `is_done/is_success` 仍只作为 benchmark 外层终止与评分信号，不能写入 cerebellum runtime 的成功事件。下一实验必须使用策略显式返回的 `handoff=true`，保存 `retrieval_cerebellum_intent_chunk.json` 和纯传感 trace，并核验其中 `privileged_evaluator_enabled=false` 后，才能评估重定时是否降低超力退出或提高有限 horizon 完成率。

## E12：端到端显式 handoff smoke 失败与能力预检硬门

2026-08-23 使用正式 `bimanual_assembly` checkpoint、GPU 0 和临时端口 `8011` 运行了 1 个从自然 `env.reset()` 开始的 intent-only episode。运行脚本通过独立进程组和 `trap` 管理 OpenPI server；episode 完成后以及后续 fail-fast 复验后，端口 `8011` 均已关闭，未残留 `serve_policy.py` 进程，也未触碰 GPU 3 上正在运行的训练任务。

真实 rollout 运行满 `1500` 帧（约 `29.98 s`）后失败。`retrieval_cerebellum_intent_chunk.json` 明确记录 `policy_handoff_observed=false`、`privileged_evaluator_enabled=false`、`events=[]`；说明正式 π0.5 server 从未输出 `handoff=true`，sensor-only runtime 因此没有获得控制权。当前 `DualArmOutputs` 只返回 `actions`，server metadata 也没有声明显式 handoff 能力，不能通过客户端手工注入信号绕过该失败。

纯传感 trace 的审计结果写入 `outputs/retrieval_cerebellum/explicit_handoff_smoke_20260823/sensor_audit.json`：峰值腕力 `113.72 N`，`8 N` 软门以上 `738` 帧，`20 N` 硬门以上 `671` 帧，首次超过硬门发生在约 `7.94 s`。这些超力发生在 π0.5 尚未交接、runtime 未接管期间，证明继续重复裸 rollout 不安全且没有研究价值。

因此新增不可绕过的 server capability 预检：intent 模式在环境 rollout 前读取 metadata，只有严格布尔值 `capabilities.explicit_handoff=true` 才允许继续；字段缺失、`false`、整数 `1` 或其他 truthy 值均拒绝。使用当前正式 server 的复验在环境动作执行前立即报错，临时服务随后自动关闭。下一步不再运行端到端 intent smoke，直到训练或实现真正的 π0.5 handoff 输出，并由 server 正式声明该 capability。

## E13：诚实标记的 synthetic handoff 视频验证

为避免在尚无正式 handoff head 时直接投入训练，新增 `retrieval_cerebellum_synthetic_handoff_frame` 调试模式。该模式不会修改 OpenPI server 输出；客户端在指定帧后把每个真实 π0.5 action chunk 标记为 `handoff_source=synthetic_test` 并交给 sensor-only runtime。审计同时记录 `policy_handoff_observed=false` 与 `synthetic_handoff_observed=true`，因此不能用于通过正式 A0 门。

自然 `env.reset()` seed 0 在 frame 362 首次 synthetic 接管。与同 seed 裸 π0.5 对照相比，接管后的腕力峰值从 `113.72 N` 降至 `12.96 N`，`20 N` 硬门超限从 `671` 帧降为 `0`，峰值下降约 `88.6%`。完整并排视频为 `outputs/retrieval_cerebellum/synthetic_handoff_frame360_20260823/raw_vs_synthetic_full.mp4`，交接附近十秒剪辑为 `outputs/retrieval_cerebellum/synthetic_handoff_frame360_20260823/raw_vs_synthetic_handoff_clip.mp4`。该 episode 仍失败；时间线检查表明 peg 在 synthetic 接管前已经掉落，因此这轮只验证限力与接管链路，不验证插入能力。

随后在冻结的自然 held-out handoff episode 31、frame 437 上进行低成本后半段验证。外部记录动作只用于恢复“peg 与孔座已抓持、接近孔口”的测试初态；恢复后立即向正式 π0.5 server 请求新的 30 步 action chunk，runtime 只读取实时 chunk、本体、多指力和腕力。系统在 9 步内完成插入，峰值残差腕力 `1.45 N`，最终横向误差 `1.68 mm`、轴线误差 `0.0102 rad`。视频为 `outputs/retrieval_cerebellum/frozen_handoff_ep31_synthetic_20260823/videos/episode_000031_online_pi05.mp4`，最后一帧明确标记 `inserted`。

结论：不训练 handoff head 也能先证明后半段值得保留——sensor runtime 显著降低超力，且在有效 handoff 状态上可利用实时 π0.5 chunk 完成插入。但这仍不是自然 `env.reset()` 端到端成功：seed 0 的失败发生在 handoff 前抓取阶段，episode 31 的成功使用了外部冻结初态和 privileged evaluator 终止。下一步是否训练 handoff head，应取决于更多冻结 handoff 上的安全/完成率收益，而不是把 synthetic 信号伪装成正式输出。

## E14：粗对准接管边界修正与接触补偿消融

2026-08-23 修正了项目边界：小脑算法从“π0.5 已完成双手抓取、搬运和粗对准”开始。冻结 held-out handoff 是局部精细插入方法的正式测试初态，不再要求局部控制器从自然 `env.reset()` 解决前置长程操作。自然全流程和显式 policy handoff 被降为后续系统集成门，不能反向否定局部视触觉方法。毫米级完整视觉姿态也不再是局部方法前置条件；视觉只需提供粗区域或可靠度，最终补偿允许来自腕力、指尖力和有界动作响应。

代码审计发现旧 synthetic handoff gate 可在当前接触手指数为 `0` 时仍返回 ready，现已要求交接瞬间双方仍满足当前多指接触和总指尖力门。完整测试由此前的 `179 passed, 1 failed` 修复为 `186 passed`。

随后使用正式 `bimanual_assembly` checkpoint，在同一服务、同一 9 个冻结粗对准状态上执行实时 π0.5 chunk 消融。首次复验只有 `2/9`，7 个 episode 均被 `grasp_unstable_stop` 提前终止。原因是当前仿真指尖 `cfrc_ext` 通道尚未证明能覆盖实际灵巧手抓持接触，却被直接用于 retention 硬门。关闭该未经验证的控制门、保留其审计值后，无接触补偿基线连续两轮恢复为 `8/9`，确认旧 `8/9` 结果可重复，`2/9` 是错误传感门造成的假失败。抓持 retention 现在默认只记录，不参与减速、左右手分配或停止；只有完成独立传感有效性标定后才允许显式启用。

接触补偿实现从实时 π0.5 chunk 的相对双腕位移提取接近轴，在腕力超过接触门后只添加有界切向微探测，不覆盖 chunk 的轴向和时变双腕意图。平移补偿消融保持 `8/9`，平均峰值腕力由 `3.19 N` 降至 `2.72 N`，平均最终横向误差由 `2.11 mm` 降至 `1.84 mm`，但平均执行步数由 `36.44` 增至 `39.67`，且 episode 17 仍失败。进一步加入腕矩驱动的相对旋转柔顺后仍为 `8/9`，episode 17 最终倾斜从约 `0.1121 rad` 恶化到 `0.1204 rad`，没有形成救回证据。

因此接触补偿不作为默认主结果启用。当前代码将其保留为显式 `--retrieval-cerebellum-contact-response` 实验开关，并新增接触释放滞回；默认接触门提高到 `4 N`，未验证的腕矩旋转增益默认为零。结果位于 `outputs/retrieval_cerebellum/contact_response_ablation9_20260823/comparison.json`。本轮结束后 π0.5 服务已关闭，端口 `8011` 无监听。下一步应构造真实、可重复的 rim-contact 粗对准子集，单独辨识接触响应方向；不能继续在所有低力正常插入帧上永久探测，也不能把当前等成功率结果写成补偿提高了完成率。
