# P0-C2 Stage-1 Controllability (P0-C2-S1)

- 日期：2026-08-15T02:28:24Z
- overall_verdict：**h2_failed_no_finger_causal_effect**
- decision_tree：**A**
- research_decision：**stop_h2_controllability_route**
- enter_stage2：False
- criteria：`c2_root_v1`（冻结；hold-screen 后再跑干预）
- wrist：`demo_wrist`（全分支相同 demo wrist；未重调 transport load）
- selected roots：8 / screened 27
- fairness_pass_rate：1.0
- tip distance 仅诊断，不作通过理由
- claims_controllability_p0_pass=false；allow_policy_training=false

## 必须回答

1. finger action 是否制造真实因果分叉？ **否**
2. privileged+future action 预测？（Stage-2；未进入则 N/A）
3. true proprio/command/FT 增量？（Stage-2；未进入则 N/A）
4. 项目动作：**stop_h2_controllability_route**

## Selected roots

- ep6 f256 early_grasp reasons=['hold_trans_drift'] hold_drift=0.035154878674684496
- ep4 f484 pre_insert reasons=['hold_trans_drift'] hold_drift=0.02115548005469454
- ep18 f415 pre_insert reasons=['hold_trans_drift', 'hold_reduced_retention'] hold_drift=0.019653297995270422
- ep8 f229 early_grasp reasons=['hold_trans_drift'] hold_drift=0.0189619214887451
- ep6 f271 transport reasons=['hold_trans_drift'] hold_drift=0.017032507810864666
- ep18 f243 early_grasp reasons=['hold_trans_drift'] hold_drift=0.0154763347571578
- ep14 f307 early_grasp reasons=['hold_trans_drift', 'hold_reduced_retention'] hold_drift=0.010916474376791812
- ep14 f517 pre_insert reasons=['hold_trans_drift', 'hold_reduced_retention'] hold_drift=0.010250502555462036

## Paired effects vs hold_finger

### demo_finger_replay (fork=False)
- trans_drift_max_m: mean_diff=0.0021296298227199547 CI[-0.0013365232706225843,0.007336125365067772] sig=False
- rot_drift_max_rad: mean_diff=0.08369767984196358 CI[-0.004900680231912267,0.20614885730235383] sig=False
- contact_retention_vs_root_mean: mean_diff=-0.03828125000000002 CI[-0.09845703125000006,0.0078125] sig=False
- object_dropped_proxy: mean_diff=0.0 CI[0.0,0.0] sig=False
- terminal_peg_ok: mean_diff=-0.25 CI[-0.625,0.0] sig=False

### calibrated_finger_intervention (fork=False)
- trans_drift_max_m: mean_diff=-5.775291070149501e-05 CI[-0.004782859824202723,0.0046932478007755025] sig=False
- rot_drift_max_rad: mean_diff=0.006772866234027445 CI[-0.028183865821798416,0.040768054687979546] sig=False
- contact_retention_vs_root_mean: mean_diff=-0.0009765625000000139 CI[-0.0146484375,0.010766601562499976] sig=False
- object_dropped_proxy: mean_diff=0.0 CI[0.0,0.0] sig=False
- terminal_peg_ok: mean_diff=0.0 CI[0.0,0.0] sig=False

### random_finger_control (fork=False)
- trans_drift_max_m: mean_diff=0.0010979262549513532 CI[-0.004377886739128414,0.006820045940215829] sig=False
- rot_drift_max_rad: mean_diff=0.005628012114485459 CI[-0.022781322032624427,0.038727795286355206] sig=False
- contact_retention_vs_root_mean: mean_diff=-0.03151041666666668 CI[-0.09401041666666668,0.0078125] sig=False
- object_dropped_proxy: mean_diff=0.0 CI[0.0,0.0] sig=False
- terminal_peg_ok: mean_diff=0.0 CI[0.0,0.0] sig=False

## Conclusion

当前 simulator/control formulation 下 H2 失败：finger intervention 在跨 root 配对分析中未能稳定改变主物理指标。

停止当前策略路线；不得用 sensing/网络容量抢救；仅可另行决定是否修 actuator/contact/control interface。
