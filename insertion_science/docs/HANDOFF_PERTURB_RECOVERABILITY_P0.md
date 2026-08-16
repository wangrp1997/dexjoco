# Demo Handoff Perturbation Recoverability P0

- 日期：2026-08-15
- 状态：执行
- 禁止：PrivHI/废弃策略、训练、扩采、纯 demo 包围盒审计

## 问题

成功 demo 的 handoff 周围，是否存在**连续可恢复邻域**？
（原 demo 后续动作不变，只微扰 handoff 状态。）

## 回放口径（预注册）

- 动作：`raw_flat_to_dict`（与 `InsertHandoffEnv.demo_insert_transitions` 一致；不用 policy46 OSC）。
- 终止：仅 `insert_ok` / peg_lost / demo 耗尽（不用 FullEpisode geom-seat ~4.5cm 提前 success）。
- 入选：identity continuation 能复现 `insert_ok` 的 episode；筛出失败者只记数据质量，不进扰动主表。

## 协议

1. 回放 demo 至 `peg_lift_end`，捕获 snapshot。
2. **Baseline**：零扰动 + 原 demo 后续绝对动作 → 必须能复现 `insert_ok`（否则先审 replay）。
3. 预注册微扰：o2h / tip-lat / tip-along / axis / finger-grasp，多档幅度。
4. 每条分支：restore → 扰动 → 同一后续 demo 动作 → 记录 `insert_ok`、tip_min、peg_lost。
5. Discovery / held-out episode 预冻结；禁止事后改幅度。

## 判定

| 分支 | 条件 | 决策 |
|------|------|------|
| 1 | 非零扰动下仍有成功，且成功率随幅度大致平滑下降 | 可恢复邻域存在；可研究如何覆盖/学习 |
| 2 | 仅近零扰动成功 | 成功是窄孤岛；重做数据生成 |
| 3 | 原始 continuation 不能稳定复现成功 | 先审 replay/环境口径；暂停算法研究 |
