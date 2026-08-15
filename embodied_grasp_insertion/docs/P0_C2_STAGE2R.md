# P0-C2 Stage-2R Privilege-Complete (P0-C2-S2R)

- 日期：2026-08-15T05:18:04Z
- overall_verdict：**stage2r_privilege_cannot_predict**
- decision_tree：**B**
- research_decision：**stop_project_formal**
- summary：完整 privileged+action 仍不能预测物理分叉 → 正式停止当前项目。
- n_samples=36；roots exported=12
- 复用 S1b roots/outcomes；MjData 补导出 o2h pose/vel、finger q/qdot、wrist state/FT
- Ridge only；禁止新采集/复杂网络/策略/触觉视觉/pilot
- enter_stage3=False；allow_policy_training=False

## Split

- {"train_episodes": [4, 6, 8], "val_episodes": [14, 18], "train_root_ids": ["ep004_f0484_pre_insert", "ep006_f0256_early_grasp", "ep006_f0271_transport", "ep008_f0229_early_grasp"], "val_root_ids": ["ep014_f0307_early_grasp", "ep014_f0517_pre_insert", "ep018_f0243_early_grasp", "ep018_f0415_pre_insert"], "test_root_ids": ["ep003_f0431_pre_insert", "ep009_f0407_pre_insert", "ep011_f0258_early_grasp", "ep013_f0344_early_grasp"]}

## Per-target

- **d_trans_drift_max_m**: tree=B oracle_ok=False q_ok=False qft_ok=False
| condition | val MAE | test MAE |
|---|---|---|
| train_mean | 0.00311 | 0.00333 |
| phase_mean | 0.00338 | 0.00331 |
| action_only | 0.00334 | 0.00337 |
| privilege_plus_action | 0.00343 | 0.00353 |
| qpos_qdot_plus_action | 0.00332 | 0.00334 |
| qpos_qdot_ft_plus_action | 0.00336 | 0.00337 |
| mismatched_action | 0.00328 | 0.00345 |

- **d_rot_drift_max_rad**: tree=B oracle_ok=False q_ok=False qft_ok=False
| condition | val MAE | test MAE |
|---|---|---|
| train_mean | 0.07179 | 0.01573 |
| phase_mean | 0.07315 | 0.02353 |
| action_only | 0.05847 | 0.03562 |
| privilege_plus_action | 0.07465 | 0.01706 |
| qpos_qdot_plus_action | 0.07291 | 0.01603 |
| qpos_qdot_ft_plus_action | 0.07358 | 0.01647 |
| mismatched_action | 0.07310 | 0.01674 |

- **d_contact_retention**: tree=B oracle_ok=False q_ok=False qft_ok=False
| condition | val MAE | test MAE |
|---|---|---|
| train_mean | 0.06719 | 0.00433 |
| phase_mean | 0.06719 | 0.00433 |
| action_only | 0.06719 | 0.00433 |
| privilege_plus_action | 0.06719 | 0.00433 |
| qpos_qdot_plus_action | 0.06719 | 0.00433 |
| qpos_qdot_ft_plus_action | 0.06719 | 0.00433 |
| mismatched_action | 0.06719 | 0.00433 |

- **d_terminal_peg_ok**: tree=B oracle_ok=False q_ok=False qft_ok=False
| condition | val MAE | test MAE |
|---|---|---|
| train_mean | 0.16667 | 0.00000 |
| phase_mean | 0.16667 | 0.00000 |
| action_only | 0.16667 | 0.00000 |
| privilege_plus_action | 0.16667 | 0.00000 |
| qpos_qdot_plus_action | 0.16667 | 0.00000 |
| qpos_qdot_ft_plus_action | 0.16667 | 0.00000 |
| mismatched_action | 0.16667 | 0.00000 |

## Stop

- 本轮结束；等待人工决策；不自动扩展。
