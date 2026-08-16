# Handoff Support-Region Audit（已废止）

> **2026-08-15**：纯 demo percentile-box 不批准。  
> 改由 **Demo Handoff Perturbation Recoverability P0**（见 `HANDOFF_PERTURB_RECOVERABILITY_P0.md`）。  
> 下文仅历史记录；勿再挂 PrivHI。

- 日期：2026-08-15
- 状态：废止
- 前置：Candidate B 停止，不抢救

## 问题

成功插入所需的 grasp → transport → handoff 状态，在现有数据中是否形成
**足够连续、可学习的支持区域**？

## 禁止

不训练策略、不扩采 episode、不修改旧项目、不重跑 Candidate A/B。

## 数据（只读）

1. Sidecar 演示（成功插入轨迹）：handoff 几何与抓持/运输漂移。
2. PrivHI expand15 + holdout 评测：已知 `insert_ok`，回放至 `peg_lift_end` 取 handoff 状态。

## 分析轴

- 抓持稳定性与手内位姿（o2h / contact retention）
- transport 累计漂移
- handoff 时 tip / lat / along / axis
- 失败 handoff 是否落在成功支持区域内（可恢复性代理）
- 成功区域是否连续（近邻密度），而非孤立点

## 判定

1. **支持可确认**：成功 handoff 形成非空连续区域，且覆盖相当比例的失败 handoff
   → 覆盖不是主瓶颈；应审策略/接口而非先扩采。
2. **覆盖缺口**：失败系统性落在成功 hull 外 → 现有数据未覆盖可恢复 handoff；考虑重做数据生成。
3. **无法确认**：成功样本过少/区域不连续/指标不足 → 暂停插孔研究主线，不宣称环境坏。
