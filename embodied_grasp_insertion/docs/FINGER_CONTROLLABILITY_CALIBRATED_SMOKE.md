# Finger Controllability Calibrated Smoke (P0-C1)

- 日期：2026-08-13T15:03:46Z
- 结论：**no_effect**
- 扩展 Controllability P0：`false`（本轮未达 promising 门槛）
- Observability / Semantic / policy：**仍禁止**
- fairness_pass_rate：1.000（open/random 因 joint limit clip 导致 L2 与 close_low 不完全匹配，已记为 documented asymmetry，不判 infrastructure_fail）
- reason：effects present but below promising bar (best=calibrated_open_low:3roots/2eps, screened_roots=8, stabilizing_rows=6)

## 校准与筛选
- 右手 16/16 关节确定 flexion（medium+）；pulse |Δtarget|≈0.0375 rad；触限见 semantics manifest
- 扫描 6 episodes / 16 candidates → 8 unstable；smoke 用 8 roots × 2 contexts × 6 interventions = 96 branches
- wrist_hold 与 matched_transport_load 各覆盖同一批 8 个 physical roots
- close_low/open_low 目标 |offset|≈0.02 rad/关节（L2≈0.08）；medium≈0.04；open/random 有 17 次 joint-limit clip 导致 L2 不完全匹配（已记录）

## 成对相对 hold（严格：无 drift 回归 + 至少一项改善）
- `calibrated_close_low`: better 0/16；trans 变差 14/16
- `calibrated_close_medium`: better 0/16；trans 变差 14/16
- `calibrated_open_low`: better 6/16；trans 变差 6/16
- `random_matched`: better 2/16；trans 变差 10/16
- `right_demo_replay`: better 2/16；trans 变差 12/16

## 解释
- `calibrated_close_*`：平均增大 o2h drift，不支持“闭合稳定抓持”
- `calibrated_open_low`：少数 unstable root（约 3 physical / 2 episodes）降 drift，且优于 random，但 **episode 覆盖不足 promising（需 ≥3 eps）**
- 不是 Controllability passed；未覆盖 P0-C0；禁止训练

## P0-C0 边界（保留）
harmful_only 仅指稳定 demo roots + wrist-hold + 24 步固定增量下未优于 hold；非证伪手指控制/full-hand policy。
