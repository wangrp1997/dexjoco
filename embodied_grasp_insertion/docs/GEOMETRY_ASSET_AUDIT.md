# Geometry Asset Audit (P0-S0)

- 日期：2026-08-14T02:48:25Z
- families：8（round+rect × 4/8/12/16）

## 单位口径（关键）

- raw mesh / yaml：米制，round_8 peg Ø≈0.007986
- 官方 XML mesh scale：4.5
- 官方 8mm collision radius：0.01785 m
- 由 mesh×scale 测得 round_8 radius：0.017968 m
- |官方 collision − 测得|：0.00011849999999999708

Asset filename '8mm' refers to IndustReal nominal size in RAW meters (~0.008). MuJoCo arena visual/collision are ~4.5x larger. Never equate name-mm to collision meters.

## Family 摘要

| family | section | yaml peg | mesh xy span | visual Ø/W after×4.5 | clearance (raw) |
|---|---|---|---|---|---|
| round_4mm | round | Ø0.003988 | 0.003988 | Ø0.0179 | 0.000112 |
| round_8mm | round | Ø0.007986 | 0.007986 | Ø0.0359 | 0.000114 |
| round_12mm | round | Ø0.011983 | 0.011983 | Ø0.0539 | 0.000217 |
| round_16mm | round | Ø0.015983 | 0.015983 | Ø0.0719 | 0.000517 |
| rectangular_4mm | rectangular | 0.003970×0.003970 | 0.003970 | 0.0179×0.0179 | 0.000140 |
| rectangular_8mm | rectangular | 0.007964×0.006910 | 0.007964 | 0.0358×0.0311 | 0.000180 |
| rectangular_12mm | rectangular | 0.011957×0.007910 | 0.011957 | 0.0538×0.0356 | 0.000221 |
| rectangular_16mm | rectangular | 0.015957×0.009910 | 0.015957 | 0.0718×0.0446 | 0.000261 |

## 结论

- 不得把文件名 mm 当成 MuJoCo collision 直径。
- 不得把 8mm XML collision 数字复制到其他 family。
- 临时 smoke collision 必须来自 measured mesh×scale（+ yaml hole）。
- XML compile 成功 ≠ Semantic P0 通过。
