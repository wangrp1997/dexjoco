# Finger Controllability Matched Smoke (P0-C0)

- 日期（UTC）：见 `outputs/finger_controllability_smoke_v1/summary.json`
- 结论：**harmful_only**（**范围受限**，见下）
- 允许扩展 Controllability P0：**否**
- Observability / Semantic / policy：**仍禁止**

## 结论边界（重要）

`harmful_only` **仅表示**：

> 在稳定 demo roots、wrist-hold、连续 24 步固定增量干预下，测试过的手指干预均未优于 hold。

**不得写成**：

- 手指控制无法稳定抓持；
- Controllability 假设已证伪；
- full-hand policy 没有价值。

### 已知局限

1. hold baseline 为 7/7 peg retained，存在天花板；
2. mild_close 每步累计：`0.12 × finger_scale(0.15) × 24 ≈ 0.432 rad/关节`，剂量过大；
3. 所有关节统一正号，**未验证**为物理闭合方向；
4. `demo_finger_replay` 同时改变左右手手指；
5. contact retention 相对**第一动作后** contact，而非 snapshot root contact；
6. 报告中的 `peg_loss` 只是 terminal `peg_ok=false`，不等同真实掉落；
7. 当前只有 wrist-hold，没有 transport load。

保留：`verdict=harmful_only`，`allow_extended_controllability_p0=false`。

## Snapshot

- Determinism：`outputs/snapshot_restore_smoke_ep0.json`，2 roots × 8 steps，**逐步 bit-exact 通过**。
- 保存：`MjData` deepcopy + `mjSTATE_INTEGRATION` + FullEpisodeEnv Python 累计量。

## 规模

- episodes：3（0, 2, 4）
- roots：7；branches：35；fairness：1.0
- wrist：`hold`
- interventions：hold / demo_replay / mild_close / mild_open / random

## 主要观察（在上述局限内）

- 手指动作进入控制通路，并对抓持指标有因果影响。
- 在该剂量与稳定 root 上，测试干预相对 hold 未显示稳定化收益。
- tip distance 不是主指标；slip_* 仅为 proxy。

## 后续

见 **P0-C1**：校准手指动作语义 + 不稳定 root + 低剂量 target-offset pulse。
