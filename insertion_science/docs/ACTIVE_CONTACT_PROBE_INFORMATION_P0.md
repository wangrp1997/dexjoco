# Active Contact Probe Information P0

- 日期：2026-08-16
- 状态：执行
- 对应问题：P1「决策状态是否充分」。
- 禁止：训练策略、调用 HybridInsert/skill_replay、调 controller gains、看结果后改 roots/幅度/门槛。

## 科学问题

静态 root state 难以预测 matched action 的异质物理后果。一个固定、极短、净位移近零的
微探针序列，是否能产生足够的 wrench/contact temporal signature，在 episode-held-out 上
区分局部错位方向？

近期 active contact-manifold 工作表明，短 primitive interaction 的接触序列可用于状态识别；
本 P0 不迁移其控制算法，只检验本地模拟器中是否存在该信息通道。

## 与旧路线的区别

- 不是 compliance：controller gains 固定不变。
- 不是 contact-affordance：不做跨 geometry 表示，也不预测可行 twist。
- 不是 handoff/HybridInsert：只在 frozen pre-insert snapshot 上运行 10 个以内 action steps。
- 不是策略：探针与分类器均冻结，分类器只作为信息探针。

## 协议

1. 6 个 frozen pre-insert roots：3 discovery、3 episode-held-out。
2. 每 root 从完全相同 snapshot 构造 `world ±X/±Y` 四个 4 mm 级错位。
3. 每个错位执行相同 8-step、净位移近零的 ±X/±Y probe sequence。
4. 观测仅用右腕 6D wrench + peg-hand contact class counts；不输入 privileged tip/lat/pose。
5. 每 root 的 branch feature 减去同 root 零错位 baseline，消除 episode identity。
6. 冻结 ridge one-hot classifier；discovery 训练，held-out 测试。
7. 比较 static feature 与完整 temporal sequence，并做 200 次 train-label shuffle。
8. Discovery 开发阶段发现错位可能触发 episode 提前终止；在任何 held-out 运行前冻结处理：
   终止后使用 terminal sensor padding，并为每个 step 加入 done bit。终止属于探针响应，不丢弃 branch。

## 生死门

- sequence held-out accuracy ≥ 0.58；
- 每个 held-out root accuracy ≥ 0.50；
- sequence 相对 static accuracy 提升 ≥ 0.17；
- label-shuffle mean accuracy ≤ 0.35。

全过：主动接触历史含静态状态之外的信息，允许进入 probe robustness P0。  
任一不过：停止 active-probing information 方向，不靠更复杂模型抢救。
