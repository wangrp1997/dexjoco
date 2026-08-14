# Geometry Env Settle Smoke v1c (P0-S0.1c)

- 日期：2026-08-14T02:48:16Z
- 结论：**pass**
- reason：all 8 families support+clearance ok; mismatch_neg=3/3
- 纠正：v1b clearance 为 provisional；本轮纯几何 + tip/base 校准 + wall 分统
- 训练 / 全量采集 / Semantic P0：**仍禁止**
- 允许改正式 arena：True

## Families
- `round_4mm`: family=True support=True clearance=True above=True enter=True mild=True block=True tol=5.00e-05 enter_wall=0 enter_base=0 enter_min=None
- `round_8mm`: family=True support=True clearance=True above=True enter=True mild=True block=True tol=5.00e-05 enter_wall=0 enter_base=0 enter_min=None
- `round_12mm`: family=True support=True clearance=True above=True enter=True mild=True block=True tol=5.00e-05 enter_wall=0 enter_base=0 enter_min=None
- `round_16mm`: family=True support=True clearance=True above=True enter=True mild=True block=True tol=5.00e-05 enter_wall=0 enter_base=0 enter_min=None
- `rectangular_4mm`: family=True support=True clearance=True above=True enter=True mild=True block=True tol=5.00e-05 enter_wall=0 enter_base=0 enter_min=None
- `rectangular_8mm`: family=True support=True clearance=True above=True enter=True mild=True block=True tol=5.00e-05 enter_wall=0 enter_base=0 enter_min=None
- `rectangular_12mm`: family=True support=True clearance=True above=True enter=True mild=True block=True tol=5.00e-05 enter_wall=0 enter_base=0 enter_min=None
- `rectangular_16mm`: family=True support=True clearance=True above=True enter=True mild=True block=True tol=5.00e-05 enter_wall=0 enter_base=0 enter_min=None

## Mismatch (same depth as matched enter)
- `round_16mm` vs `round_4mm`: neg_ok=True wall=True depth=0.0
- `rectangular_16mm` vs `rectangular_4mm`: neg_ok=True wall=True depth=0.0
- `rectangular_8mm` vs `round_8mm`: neg_ok=True wall=True depth=0.0

未改正式 arena。

