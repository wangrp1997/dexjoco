# Geometry Compile Smoke (P0-S0)

- 日期：2026-08-14T01:47:17Z
- 结论：**pass**
- reason：8 families compile+lookup; insertable=8; mismatch_neg=3; sizes sourced from mesh×scale/yaml
- compile+lookup：8
- insertable families：8
- mismatch negatives：3
- 训练 / 全量采集 / Semantic P0：**仍禁止**

## Compile
- `round_4mm`: compile=True lookup=True init_pen=False size_match=True
- `round_8mm`: compile=True lookup=True init_pen=False size_match=True
- `round_12mm`: compile=True lookup=True init_pen=False size_match=True
- `round_16mm`: compile=True lookup=True init_pen=False size_match=True
- `rectangular_4mm`: compile=True lookup=True init_pen=False size_match=True
- `rectangular_8mm`: compile=True lookup=True init_pen=False size_match=True
- `rectangular_12mm`: compile=True lookup=True init_pen=False size_match=True
- `rectangular_16mm`: compile=True lookup=True init_pen=False size_match=True

## Insertion
- `round_4mm`: insertable=True strict=3 soft=6
- `round_8mm`: insertable=True strict=3 soft=6
- `round_12mm`: insertable=True strict=3 soft=6
- `round_16mm`: insertable=True strict=3 soft=6
- `rectangular_4mm`: insertable=True strict=3 soft=6
- `rectangular_8mm`: insertable=True strict=3 soft=6
- `rectangular_12mm`: insertable=True strict=3 soft=6
- `rectangular_16mm`: insertable=True strict=3 soft=6

## Mismatch
- `round_16mm` vs `round_4mm`: fail_proxy=True pen=True
- `rectangular_16mm` vs `rectangular_4mm`: fail_proxy=True pen=True
- `round_8mm` vs `rectangular_8mm`: fail_proxy=True pen=True

临时 XML 仅在 `outputs/geometry_xml_tmp/`，未改正式 arena。
即使 pass，下一步只能做每 family 1–2 个 reset/settle 环境 smoke。
