# Grasp Stability Multi-Family Physical Smoke (P0-S0.4c hardened)

- 日期：2026-08-14T07:34:12Z
- 结论：**pass**
- reason：family_physical_ok=4/4
- 准确命名：多族 oracle 建立接触后的物理抓取配方 smoke（非学会抓取）
- round_8mm：demo transport root（S0.4b，同态 snapshot + 横移门）
- 其他族：establish 可 snap；settle 后抓 MjData；open/closed 同态恢复
- 硬门槛：四族均 `snap_call_count_after_establish == 0` + transport 横移门
- transport：记录并要求手/peg 横向位移（含 round_8mm demo 路径）
- 三项回归现均 4/4：matched snapshot / zero-snap / lateral
- 不采集 / 不训练
- `claims_physical_grasp_stability=True`

## Families
- `round_8mm`: passed=True src=demo_transport hold=True lift=True transport=True hand_lat=0.0951 peg_lat=0.0977 neg=True closed_beats=True snap_after=0 matched=True root_c=4
- `round_16mm`: passed=True src=oracle_establish_formal hold=True lift=True transport=True hand_lat=0.0803 peg_lat=0.0796 neg=True closed_beats=True snap_after=0 matched=True root_c=12
- `rectangular_8mm`: passed=True src=oracle_establish_formal hold=True lift=True transport=True hand_lat=0.0804 peg_lat=0.0799 neg=True closed_beats=True snap_after=0 matched=True root_c=17
- `rectangular_16mm`: passed=True src=oracle_establish_formal hold=True lift=True transport=True hand_lat=0.0803 peg_lat=0.0797 neg=True closed_beats=True snap_after=0 matched=True root_c=19

三项回归 4/4 通过后，可讨论极小规模、可撤销的 micro-demo pilot；仍禁止常规采集/训练。
