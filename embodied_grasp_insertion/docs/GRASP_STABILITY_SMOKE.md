# Grasp Stability Smoke (P0-S0.4)

- 日期：2026-08-14T05:21:47Z
- 结论：**instrumentation / 阶段编排 smoke = pass**；**物理抓取稳定性门未通过**
- reason：4 代表族在 oracle palm-snap 夹具下 lift/hold/transport+开手指标编排 ok
- 正对照：每步 `snap_peg_to_palm`（非手指摩擦抓取）
- 负对照：关 snap 后自由落体，证明掉落指标敏感，**不**证明开手相对闭手抓取失败
- 阈值按 family 特征长度缩放；不采集 / 不训练
- `claims_stable_grasp_policy=false`
- `claims_physical_grasp_stability=false`
- 下一门：`P0-S0.4b` 纯动力学物理抓取门

## Families
- `round_8mm`: passed=True lift=True hold=True transport=True neg=True L=0.0179685
- `round_16mm`: passed=True lift=True hold=True transport=True neg=True L=0.03596175
- `rectangular_8mm`: passed=True lift=True hold=True transport=True neg=True L=0.017919
- `rectangular_16mm`: passed=True lift=True hold=True transport=True neg=True L=0.03590325

## 准确边界
- 证明：指标与阶段编排在夹具正对照下可跑通，开手下能检出掉落。
- 不证明：真实接触抓取、闭手优于开手、摩擦稳定、策略抓取。
- 正对照 `contacts=0`、`max_speed≈0` 来自每步重设 free-joint，属预期。

不声称抓取策略已稳定；不关闭“抓取稳定性门”。

