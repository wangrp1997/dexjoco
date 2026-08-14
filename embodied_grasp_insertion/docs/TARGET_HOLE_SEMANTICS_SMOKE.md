# Target Hole Semantics Smoke (P0-S0.3)

- 日期：2026-08-14T05:10:31Z
- 结论：**pass**（单目标孔 metadata/plumbing；非策略知孔）
- reason：all 4 families target-hole semantics ok; cross-family neg ok
- 不声称策略已知目标孔；不采集 / 不训练

## Families
- `round_8mm`: passed=True info=True wrong_disc=True labeler=True d_true=0.430 d_wrong=0.550
- `round_16mm`: passed=True info=True wrong_disc=True labeler=True d_true=0.407 d_wrong=0.527
- `rectangular_8mm`: passed=True info=True wrong_disc=True labeler=True d_true=0.402 d_wrong=0.522
- `rectangular_16mm`: passed=True info=True wrong_disc=True labeler=True d_true=0.472 d_wrong=0.592

## Cross-family negative
- passed=True round_8mm vs round_16mm

下一步：抓取稳定性门已完成（S0.4 pass）；仍禁采集/训练。

