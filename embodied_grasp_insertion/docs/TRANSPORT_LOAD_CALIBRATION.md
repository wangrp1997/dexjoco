# Transport Load Calibration (P0-C1.1)

- 日期：2026-08-13
- **verdict：`load_calibration_fail` / `screening_fail`**
- selection_ok：`false`
- selected_scale / selected_delta44：`null`（禁止使用 fallback）
- P0-C1 历史结论仍为 `no_effect`；不扩 Controllability；不训 policy

## 本轮结果（constant mild_transport）

| scale | unstable | stable | median Δdrift |
|---|---|---|---|
| 1.0 | 0 | 14 | ~0.03 mm |
| 2.0 | 0 | 14 | ~0.05 mm |
| 3.0 | 1 | 13 | ~0.10 mm |
| 4.5 | 2 | 12 | ~0.14 mm |
| 6.0 | 2 | 12 | ~0.29 mm |

未达门槛：unstable≥4、stable≥3、median Δdrift>0.5mm、且需稳定/不稳定分化。

## 物理判断

同向连续 wrist translation 主要是平稳搬运：手与 peg 一起动，o2h 相对漂移几乎不增。
不是手指控制失败，是**激励波形不对**。

## 下一轮

**已取消** waveform load 继续调参。
正式停止 P0-C1.1；转向 `docs/SEMANTIC_DATA_DESIGN.md`（多物体/多孔几何只读设计）。
