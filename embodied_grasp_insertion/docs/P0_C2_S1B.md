# P0-C2-S1b Actuation/Fork Audit (P0-C2-S1b)

- 日期：2026-08-15T04:12:06Z
- overall_verdict：**h2_controllability_exists_heterogeneous**
- research_decision：**enter_stage2_action_conditioned_eligible**
- enter_stage2（仅资格，不自动开跑）：True
- Stage-1 A 判定：**已撤回**；本轮验证执行与存在性分叉
- independent dose projection：是（calibrated/random 不再 common-scale 互拖）
- frozen actuation_moved=True (frac=1.00)
- frozen existence_any=True roots=['ep006_f0256_early_grasp', 'ep006_f0271_transport', 'ep008_f0229_early_grasp', 'ep014_f0307_early_grasp', 'ep014_f0517_pre_insert', 'ep018_f0243_early_grasp', 'ep018_f0415_pre_insert']
- held-out existence_any=True roots=['ep003_f0431_pre_insert', 'ep009_f0407_pre_insert', 'ep011_f0258_early_grasp', 'ep013_f0344_early_grasp']
- directional_any_metric(frozen)=True （方向性≠存在性）

## Summary

关节动了，并出现可重复物理分叉（含 held-out）：H2 可控性存在；方向同号另计，可进入 action-conditioned Stage-2。

## Notes

- 存在性：同 root matched branches 超过重放容差的物理差异。
- 方向性：跨 root 同号；只说明是否已有通用干预。
- `outputs/p0_c2_stage1_v1/` 仍保留，勿删。
