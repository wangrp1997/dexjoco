# Finger Actuation Semantics (P0-C1)

- 日期：2026-08-13T14:19:38Z
- episode：0
- pulse_norm=0.25 → target Δ≈0.0375 rad
- 判定：**pass**（16/16，medium+=16）

注：Allegro tip body 常与 distal 原点重合；校准使用 tip 局部 +z 12mm 虚拟点。

| idx | joint | flexion | confidence | reason |
|---|---|---|---|---|
| 0 | `ffj0_right` | negative | medium |  |
| 1 | `ffj1_right` | positive | medium |  |
| 2 | `ffj2_right` | positive | medium |  |
| 3 | `ffj3_right` | positive | medium | used_tip_peg_because_tip_palm_ambiguous |
| 4 | `mfj0_right` | positive | medium | used_tip_peg_because_tip_palm_ambiguous |
| 5 | `mfj1_right` | positive | medium |  |
| 6 | `mfj2_right` | positive | medium |  |
| 7 | `mfj3_right` | positive | medium | used_tip_peg_because_tip_palm_ambiguous |
| 8 | `rfj0_right` | positive | medium |  |
| 9 | `rfj1_right` | positive | medium |  |
| 10 | `rfj2_right` | positive | medium |  |
| 11 | `rfj3_right` | positive | medium | used_tip_peg_because_tip_palm_ambiguous |
| 12 | `thj0_right` | positive | high |  |
| 13 | `thj1_right` | positive | medium |  |
| 14 | `thj2_right` | positive | high |  |
| 15 | `thj3_right` | positive | high |  |

左手仅 schema；peg 干预只用右手。

