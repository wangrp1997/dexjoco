# Handoff Recoverability Boundary / Coverage P0

- 日期：2026-08-16
- 状态：执行
- 禁止：训练策略、扩采、挂 PrivHI 主线、纯 tip/lat 包围盒当结论

## 问题

1. 各扰动类型×方向×幅度的**可恢复边界**在哪（各向异性）？
2. 现有真实失败 handoff 多数落在边界**内还是外**？

## 数据

- 边界：`outputs/handoff_perturb_recoverability_p0/results.json`（held-out 优先）。
- 失败 handoff A：归档 insert 失败 traj 首帧（只读状态样本，不复活 PrivHI）。
- 失败 handoff B：demo identity continuation 失败 episode 的 `peg_lift_end` 特征（sim 回放）。

## 协议

1. 按 `pert_name × scale` 统计 held-out `insert_ok` 率；`rate >= inside_rate_min` 视为该 cell 可恢复。
2. 每方向取最大可恢复 scale 为边界；axis 等脆弱方向单独报告。
3. 每个失败相对最近成功 root，换算等效 `s_lat / s_along / s_axis`。
4. 任一已测轴等效 scale 超过该方向边界 → **边界外**；否则 **边界内**。
5. 边界内再查：是否存在同根/近邻成功纠正证据（同 ep 成功 traj 或 recoverability 成功 root）。

## 判定

| 分支 | 条件 | 决策 |
|------|------|------|
| 1 | 失败多数在边界外 | 重做 handoff 数据生成 |
| 2 | 失败多数在边界内，但缺少成功纠正配对 | 定向生成同根成功/失败配对 |
| 3 | 边界内已有足够可区分配对 | 才允许最小监督策略 P0 |

不训练；本 P0 只出边界表 + 覆盖判定。
