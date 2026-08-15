# P0-C2 Stage-2 Action-Conditioned H4 (P0-C2-S2)

- 日期：2026-08-15T04:50:00Z
- overall_verdict：**stage2_oracle_cannot_predict**
- decision_tree：**B**
- research_decision：**stop_task_or_label_invalid**
- enter_stage3：False；allow_policy_training：False
- 数据：仅 S1b matched-branch JSON（n_samples=36）
- 目标：动作相对 hold 的有符号后果（不要求跨 root 同向）
- 模型：Ridge only；episode/root held-out；配对 bootstrap

## 数据边界（重要）

- `{"wrist_ft": "unavailable_in_s1b_export", "command_history_pre_root": "unavailable_in_s1b_export", "qdot_at_root": "unavailable_in_s1b_export", "privilege": "root_contact_total_by_class_plus_phase_only__no_o2h_in_export", "future_action": "finger_slice_of_future_actions44_padded", "target": "signed_intervention_minus_hold_metrics"}`
- wrist FT / root qdot / 预 root command history：**S1b 导出中不存在**，本轮标为 unavailable，不作假阴性。

## Split

- train eps [4, 6, 8] roots ['ep004_f0484_pre_insert', 'ep006_f0256_early_grasp', 'ep006_f0271_transport', 'ep008_f0229_early_grasp']
- val eps [14, 18] roots ['ep014_f0307_early_grasp', 'ep014_f0517_pre_insert', 'ep018_f0243_early_grasp', 'ep018_f0415_pre_insert']
- test（S1b held-out）roots ['ep003_f0431_pre_insert', 'ep009_f0407_pre_insert', 'ep011_f0258_early_grasp', 'ep013_f0344_early_grasp']

## Primary target `d_trans_drift_max_m`

- oracle_ok=False；deploy_qpos_ok=False
- checks：{"oracle_beats_mean_val": false, "oracle_beats_mean_test": false, "oracle_beats_action_val": false, "oracle_beats_action_test": false, "oracle_beats_mismatch_val": false, "oracle_beats_mismatch_test": false, "qpos_beats_mean_test": false, "qpos_beats_action_test": false, "qpos_beats_mismatch_test": false}

| condition | val MAE | test MAE | alpha |
|---|---|---|---|
| train_mean | 0.00311 | 0.00333 | None |
| phase_mean | 0.00338 | 0.00331 | None |
| action_only | 0.00338 | 0.00369 | 100.0 |
| command_future_action | 0.00338 | 0.00369 | 100.0 |
| qpos_plus_action | 0.00650 | 0.00348 | 1000.0 |
| privilege_contact_only | 0.00311 | 0.00339 | 1000.0 |
| privilege_contact_plus_action | 0.00727 | 0.00350 | 1000.0 |
| mismatched_action | 0.01659 | 0.00362 | 100.0 |

## All targets (decision branches)

- **d_trans_drift_max_m**: tree=B_oracle_cannot_predict；oracle_ok=False；deploy_qpos_ok=False
- **d_rot_drift_max_rad**: tree=B_oracle_cannot_predict；oracle_ok=False；deploy_qpos_ok=False
- **d_contact_retention**: tree=B_oracle_cannot_predict；oracle_ok=False；deploy_qpos_ok=False
- **d_terminal_peg_ok**: tree=B_oracle_cannot_predict；oracle_ok=False；deploy_qpos_ok=False

## Stop rules applied

- B：Oracle+action 预测不了 → 任务/标签无效，停止
- C：Oracle 能、部署不能 → sensing gap
- D：部署能区分动作后果 → H4 初步证据，仍不训策略
- **不进入 Stage-3**；完成后等人决策

## Retain

- `outputs/p0_c2_s1b_v1/` 与 `outputs/p0_c2_stage1_v1/` 继续保留
