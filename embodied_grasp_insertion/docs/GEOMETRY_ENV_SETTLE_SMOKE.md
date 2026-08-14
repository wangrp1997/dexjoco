> **已作废作为物理 settle pass**：仅 numerical execution。见 `GEOMETRY_ENV_SETTLE_SMOKE_V1B.md`。

# Geometry Env Settle Smoke (P0-S0.1)

- 日期：2026-08-14T01:52:51Z
- 结论：**pass**
- reason：8/8 families settle-stable; trials 16/16
- settle_steps：200
- 每 family 2 次 reset（居中悬停 / 小侧偏悬停）后短物理 settle
- 训练 / 全量采集 / Semantic P0：**仍禁止**
- 允许改正式 arena：True

## Families
- `round_4mm`: family_passed=True trials=2/2
- `round_8mm`: family_passed=True trials=2/2
- `round_12mm`: family_passed=True trials=2/2
- `round_16mm`: family_passed=True trials=2/2
- `rectangular_4mm`: family_passed=True trials=2/2
- `rectangular_8mm`: family_passed=True trials=2/2
- `rectangular_12mm`: family_passed=True trials=2/2
- `rectangular_16mm`: family_passed=True trials=2/2

仅用临时 XML；未改正式 arena；未采集 demo。

