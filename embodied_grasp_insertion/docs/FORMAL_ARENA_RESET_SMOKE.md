# Formal Arena Reset Smoke (P0-S0.2)

- 日期：2026-08-14T05:01:56Z
- 结论：**pass**
- reason：all 8 families formal arena reset/settle ok
- 默认 `round_8mm` arena 文件未改写；其他族生成旁路 XML
- 通过条件已含：末尾 50 步低速（lin<0.005、ang<0.05）+ settle 后深穿透检查
- 训练 / 全量采集 / Semantic P0：**仍禁止**

## Families
- `round_4mm`: passed=True resets_ok=1/1 lookup=True err=None
- `round_8mm`: passed=True resets_ok=1/1 lookup=True err=None
- `round_12mm`: passed=True resets_ok=1/1 lookup=True err=None
- `round_16mm`: passed=True resets_ok=1/1 lookup=True err=None
- `rectangular_4mm`: passed=True resets_ok=1/1 lookup=True err=None
- `rectangular_8mm`: passed=True resets_ok=1/1 lookup=True err=None
- `rectangular_12mm`: passed=True resets_ok=1/1 lookup=True err=None
- `rectangular_16mm`: passed=True resets_ok=1/1 lookup=True err=None

未开始采集或训练。

