# Handoff Datagen Redesign P0

- 日期：2026-08-16
- 状态：执行
- 禁止：训练策略、抢救 PrivHI、扩采 teleop、把旧失败硬塞进训练集

## 问题

能否在**不训练**的前提下，从已有成功 demo 生成一批约束在可恢复盆地内的 handoff，
并让「盆地内可恢复 / 盆地外难恢复」成立？

## 协议

1. 根：identity continuation 已验证 `insert_ok` 的成功 episode（复用 recoverability 根 + 可选扩筛）。
2. **盆地内采样**：仅 tip_lat / tip_along / 小 o2h；幅度 ≤ 保守边界×安全系数；**不做 axis**（边界≈0）。
3. 每条：snapshot → 扰动 → 原 demo 后续 `raw_flat` → 记 `insert_ok`。
4. **盆地外对照**：同根、超界幅度（含 axis），预期成功率显著更低。
5. 导出通过样本特征与元数据（不写回旧 sidecar 主库）。
6. 复验 coverage：旧归档失败相对**新接受根**的 outside_frac（预期仍高；本门不要求失败搬家）。

## 生死门（预注册）

| 条件 | 通过 |
|------|------|
| 盆地内 `insert_ok` 率 | ≥ `in_basin_min_rate` |
| 盆地外 `insert_ok` 率 | ≤ `out_basin_max_rate` |
| 接受样本数 | ≥ `min_accepted` |
| 内外差 | ≥ `min_rate_gap` |

通过 → 可进入「用约束 handoff 集做后续研究」；失败 → 盆地/回放口径仍不可用，暂停。

## 判定

- `pass_datagen_redesign`：生死门全过。
- `fail_basin_not_generative`：盆地内也产不出稳定可恢复 handoff。
- `fail_no_contrast`：内外无区分，边界不可操作。
