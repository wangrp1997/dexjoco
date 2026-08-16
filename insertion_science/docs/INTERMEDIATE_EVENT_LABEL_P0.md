# Intermediate Physical Event Label P0

- 日期：2026-08-16
- 状态：执行
- 对应问题：P3「监督目标是否与真实任务因果一致」。
- 禁止：训练、重新运行 handoff、修改旧数据、看结果后调阈值。

## 问题

binary `insert_ok` 之外，现有 matched continuation 中是否存在一个**非退化、跨 split、
可外部复验**的中间物理事件，可作为后续因果时序审计的候选标签？

本 P0 只检验统计支持，不宣称事件已经具有因果性；通过后仍必须验证事件发生时间早于
terminal success，且干预该事件能够改变后续结果。

## 数据

1. Primary：`handoff_perturb_recoverability_p0` 的非 identity rows，冻结
   `discovery/held_out` split。
2. External：`handoff_datagen_redesign_p0` 的 in/out rows，仅作复验。
3. 全部为 `insertion_science` 现有只读输出；不启动任何旧项目 runner。

## 预注册事件

- `approach_100mm`：rollout 最小 tip distance ≤ 100 mm。
- `approach_60mm`：最小 tip distance ≤ 60 mm。
- `deep_20mm`：最小 tip distance ≤ 20 mm。
- `retained`：未发生 `peg_lost_abort`。
- `retained_and_approach_60mm`：同时满足 retained 与 approach 60 mm。

## 生死门

至少一个事件同时满足：

1. discovery 与 held_out prevalence 均在 `[0.15, 0.85]`；
2. 每个 split 的 event/non-event 组均至少 4 条；
3. 两个 split 的 `P(insert|event)-P(insert|not event) >= 0.25`；
4. pooled primary 同时存在 false positive 和 false negative，避免复制 terminal label；
5. external 的 insert lift ≥ 0.20。

通过：只允许进入 timestamp/causal-order P0，不允许训练。  
失败：停止“用稠密中间事件替代 terminal label”方向。
