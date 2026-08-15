# Embodied Grasp-Insertion — Project Freeze

- 日期：2026-08-15
- 状态：`project_frozen_low_roi`

## 成立的事实

1. **H2 可控性存在**（P0-C2-S1b）：finger 命令传到关节；matched branches 在 frozen/held-out 上有可重复物理分叉。
2. **当前观测/导出不足以支撑可靠策略**：B0/B1 被动 command+FT 预测弱；Stage-2 在 S1b JSON 上 privilege+action 未过门。
3. Stage-2 应记为 **无结论（导出不完整）**，不宜写成“任务/标签本质无效”——S1b 未导出 root o2h、qdot、wrist FT。

## 冻结内容

- 不再补测 Stage-2；不训练策略；不扩展触觉/视觉、pilot、基础设施。
- 不自动开启后续实验。
- 保留全部代码、报告与 `outputs/p0_c2_stage1_v1/`、`outputs/p0_c2_s1b_v1/`。

## 禁止的错误叙述

- 不得声称仿真插孔本身不可实现。
- 不得用 Stage-2 tree B 单独证伪 H2。
- 不得把“缺字段的 oracle 失败”写成完整 sensing-gap 已证实（tree C 未成立）。
