# Geometry Env Settle Smoke v1b (P0-S0.1b)

> **更正**：`support_settle` 仍可信 8/8；`insertion_frame_clearance` 为 **provisional/invalid**（居中 tip 撞 `socket_base`、8mm 穿透容差、mild 未进通过条件、仍跑动力学）。已被 **P0-S0.1c** 取代。正式 arena 以 v1c 为准。

- 日期：2026-08-14T02:00:26Z
- 原结论标签：**pass**（clearance 部分事后作废）
- reason：all 8 families support+clearance ok; mismatch_neg=3
- 纠正：P0-S0.1 仅为 numerical execution pass，**不能**称为物理 settle pass
- 本轮拆分：`support_settle` + `insertion_frame_clearance`；要求 8/8
- 训练 / 全量采集 / Semantic P0：**仍禁止**
- 允许改正式 arena：False（以 v1c 为准）

## Families
- `round_4mm`: family=True support=True clearance=True above=True enter=True block=True max_lin=0.196 peg_sock0=0
- `round_8mm`: family=True support=True clearance=True above=True enter=True block=True max_lin=0.196 peg_sock0=0
- `round_12mm`: family=True support=True clearance=True above=True enter=True block=True max_lin=0.196 peg_sock0=0
- `round_16mm`: family=True support=True clearance=True above=True enter=True block=True max_lin=0.196 peg_sock0=0
- `rectangular_4mm`: family=True support=True clearance=True above=True enter=True block=True max_lin=0.196 peg_sock0=0
- `rectangular_8mm`: family=True support=True clearance=True above=True enter=True block=True max_lin=0.196 peg_sock0=0
- `rectangular_12mm`: family=True support=True clearance=True above=True enter=True block=True max_lin=0.196 peg_sock0=0
- `rectangular_16mm`: family=True support=True clearance=True above=True enter=True block=True max_lin=0.196 peg_sock0=0

## Mismatch
- `round_16mm` vs `round_4mm`: neg_ok=True
- `rectangular_16mm` vs `rectangular_4mm`: neg_ok=True
- `round_8mm` vs `rectangular_8mm`: neg_ok=True

未改正式 arena。

