# Observability Minimal Ridge Diagnostic (P0-Obs-B0)

- 日期：2026-08-14T18:10:07Z
- overall_verdict：**diagnostic_signal**
- research_decision：**continue_candidate_sensing_signal**
- pack：`/mnt/hdd/dexjoco/datasets/embodied_grasp_insertion/observability_eval_v1`
- samples：200；target = primary H=8 窗末帧 o2h
- 模型：Ridge only；alpha 网格 [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]；val 选 alpha，test 评一次
- episode 等权；bootstrap n=1000
- claims_observability_p0_pass=False
- allow_policy_training=False
- 稳定优于 train-mean：['A_H1', 'A_H8', 'B_H1']
- FT 有效声称：False

## Val / Test（episode-equal）

| condition | val | test | alpha_t / alpha_r |
|---|---|---|---|
| train_mean | tMAE=0.0890m tRMSE=0.0890m rMAE=30.89deg | tMAE=0.0607m tRMSE=0.0609m rMAE=19.50deg | None / None |
| A_H1 | tMAE=0.0586m tRMSE=0.0588m rMAE=22.60deg | tMAE=0.0347m tRMSE=0.0349m rMAE=16.18deg | 100.0 / 100.0 |
| A_H8 | tMAE=0.0590m tRMSE=0.0595m rMAE=25.52deg | tMAE=0.0375m tRMSE=0.0378m rMAE=19.50deg | 100.0 / 100.0 |
| B_H1 | tMAE=0.0562m tRMSE=0.0572m rMAE=22.01deg | tMAE=0.0387m tRMSE=0.0393m rMAE=16.97deg | 10.0 / 100.0 |
| B_H8 | tMAE=0.0576m tRMSE=0.0584m rMAE=23.99deg | tMAE=0.0390m tRMSE=0.0395m rMAE=21.02deg | 100.0 / 100.0 |
| B_H8_shuffled_FT | tMAE=0.0587m tRMSE=0.0595m rMAE=24.40deg | tMAE=0.0396m tRMSE=0.0401m rMAE=21.84deg | 100.0 / 100.0 |
| privileged_o2h_ceiling | tMAE=0.0001m tRMSE=0.0001m rMAE=0.03deg | tMAE=0.0001m tRMSE=0.0001m rMAE=0.02deg | 0.001 / 0.001 |

## Bootstrap CI (test, 95%)

- **train_mean**: tMAE [0.0485, 0.0722]；rMAE [16.39, 22.39] deg
- **A_H1**: tMAE [0.0292, 0.0401]；rMAE [11.55, 20.16] deg
- **A_H8**: tMAE [0.0326, 0.0428]；rMAE [13.10, 24.79] deg
- **B_H1**: tMAE [0.0320, 0.0453]；rMAE [12.42, 21.05] deg
- **B_H8**: tMAE [0.0320, 0.0459]；rMAE [15.77, 26.15] deg
- **B_H8_shuffled_FT**: tMAE [0.0320, 0.0466]；rMAE [16.36, 26.91] deg
- **privileged_o2h_ceiling**: tMAE [0.0000, 0.0001]；rMAE [0.02, 0.02] deg

## Guards

- {"WRITE_IMPLEMENTATION_ENABLED": false, "evaluation_only": true, "allow_policy_training": false, "claims_observability_p0_pass": false, "no_policy_training": true, "no_new_collection": true, "no_pilot_write": true, "no_reopen_c0_c1": true}

## Next

- 本轮结束；不自动进入下一模型。
- **研究判断（本轮）**：
  - **可继续（弱）**：A_H1 / A_H8 / B_H1 在 val+test 上同时优于 train-mean → 部署输入对 o2h 有可检出线性信号。
  - **不可声称 FT 有效**：B_H8 未稳定优于 A_H8 与 shuffled-FT（test 旋转甚至差于 train-mean）。
  - **H8 不优于 H1**：更长历史未带来线性增益。
  - **标签可学**：privileged-o2h ceiling ≈ 0 → 瓶颈是 sensing/表征，不是标签坏。
  - **仍禁止**：Obs P0 通过声明、控制策略训练、换复杂网络（本轮已回答问题）。
  - 若继续，需另授权下一实验设计；若停，接受当前 A/B 仅弱可观测、不支撑策略训。
