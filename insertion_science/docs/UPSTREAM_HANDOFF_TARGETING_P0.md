# Upstream Handoff Targeting P0（handoff 方向最终生死门）

- 日期：2026-08-16
- 状态：执行
- 禁止：训练、PrivHI、demo 邻域扰动当上游、删 along_far 的假 P0.1

## 问题

真实上游（skill_replay：检索→δ* grasp→lift→privileged approach）产生的 handoff，
能否命中已测可恢复盆地？命中后端到端能否 `insert_ok`？

若不能：停止整个 handoff 方向（不是再改 datagen 扰动）。

## 上游定义（本 P0）

- 调用 `run_skill_replay`：**禁止** `--force-demo` / `--restore-demo-layout`。
- 在 hybrid `handoff` 确认后采集 tip/lat/along/axis。
- 相对 recoverability 成功 root 换算等效 scale，用 coverage 保守边界判定 in/out。
- 对照：同次 run 的 insert success。

说明：接近 socket 仍为 privileged servo（非学习运输）；本门测的是「真实抓抬 + 现有接近管线」能否进盆地并完成插入。

## 预注册生死门

| 指标 | 门槛 |
|------|------|
| 尝试 seeds | 冻结列表，≥12 |
| 到达 handoff 的比例 | ≥ `min_handoff_rate` |
| handoff 中盆地命中率 | ≥ `min_basin_hit_rate` |
| 盆地命中且 insert 成功 | ≥ `min_basin_insert_rate` |

任一不过 → `stop_handoff_direction`。  
全过 → `handoff_upstream_viable_continue_research`（仍不自动开训）。

## 判定

- `pass_upstream_targeting`：生死门全过。
- `fail_stop_handoff_direction`：上游无法把真实 handoff 稳定送进可恢复盆地并兑现插入。
