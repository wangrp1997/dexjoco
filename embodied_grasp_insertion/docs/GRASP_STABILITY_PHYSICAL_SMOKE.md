# Grasp Stability Physical Smoke (P0-S0.4b)

- 日期：2026-08-14T05:42:42Z
- 结论：**pass**
- reason：physical_roots_ok=3/3
- 范围：demo transport root 单次 restore 后的**纯动力学** hold/lift/transport
- 负对照：同 root open-hand，且闭手必须显著优于开手
- 无逐步 snap/weld；不采集 / 不训练
- `claims_physical_grasp_stability=True`
- 不声称：学会的抓取策略、正式多族脚本物理抓取

## Roots
- ep0 f302: passed=True hold=True lift=True transport=True neg=True closed_beats_open=True root_c=4
- ep2 f281: passed=True hold=True lift=True transport=True neg=True closed_beats_open=True root_c=4
- ep4 f250: passed=True hold=True lift=True transport=True neg=True closed_beats_open=True root_c=4

相对 S0.4 instrumentation：本门才是物理抓取稳定性门。

