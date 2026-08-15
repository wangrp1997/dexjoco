# Observability Future-Drift Falsification (P0-Obs-B1)

- 日期：2026-08-15T01:38:55Z
- overall_verdict：**ab_sensing_falsified**（**窄配置**；见文末边界修正）
- research_decision：**stop_passive_act44_ft_linear_forecast**（非“停项/只能触觉”）
- pack：`/mnt/hdd/dexjoco/datasets/embodied_grasp_insertion/observability_eval_v1`；samples=200；单几何；test ep=15
- 目标：从 t=7 预测未来 o2h 漂移 Δ∈[1, 8]
- 正式判定：配对 episode bootstrap CI（平移+旋转均显著，且 val+test）
- oracle：仅目标帧之前的 privileged history（含 t，不含 t+Δ）
- any_deploy_real_signal=False；any_ft_helps=False
- claims_observability_p0_pass=false；allow_policy_training=false
- **边界**：本结果只关闭无未来动作条件的 `act44_command+wrist_ft` 被动线性预测；action-conditioned observability 未判定。

## B0 修正（本轮前提）

- B0「稳定优于」仅点估计，偏强；旋转配对 CI 跨 0。
- B0 ceiling 含目标帧，近零属必然，不能证明可学。
- act44 含 wrist pose，同时刻 o2h 回归可能有运动学捷径。

## Δ=1

- verdict：`ab_sensing_falsified`；real_signal=无；ft_helps=False

| condition | val | test |
|---|---|---|
| train_mean | tMAE=0.00060m rMAE=0.223deg | tMAE=0.00061m rMAE=0.225deg |
| phase_mean | tMAE=0.00060m rMAE=0.222deg | tMAE=0.00059m rMAE=0.219deg |
| current_command | tMAE=0.00071m rMAE=0.245deg | tMAE=0.00061m rMAE=0.226deg |
| time_index | tMAE=0.00060m rMAE=0.223deg | tMAE=0.00059m rMAE=0.220deg |
| A_H1 | tMAE=0.00074m rMAE=0.261deg | tMAE=0.00064m rMAE=0.246deg |
| A_H8 | tMAE=0.00116m rMAE=0.382deg | tMAE=0.00074m rMAE=0.310deg |
| B_H1 | tMAE=0.00073m rMAE=0.265deg | tMAE=0.00069m rMAE=0.251deg |
| B_H8 | tMAE=0.00117m rMAE=0.432deg | tMAE=0.00080m rMAE=0.328deg |
| B_H8_shuffled_FT | tMAE=0.00115m rMAE=0.424deg | tMAE=0.00081m rMAE=0.313deg |
| privileged_o2h_causal_ceiling | tMAE=0.00040m rMAE=0.140deg | tMAE=0.00040m rMAE=0.156deg |

- A_H8−A_H1 test 配对：trans mean=0.00011 CI[0.00004,0.00018]；rot mean=0.064 CI[0.029,0.102]

- B_H8_vs_A_H8_test: trans CI[-0.00004,0.00016] sig=False；rot CI[-0.021,0.058] sig=False
- B_H8_vs_B_H8_shuffled_FT_test: trans CI[-0.00008,0.00008] sig=False；rot CI[-0.011,0.045] sig=False

## Δ=8

- verdict：`ab_sensing_falsified`；real_signal=无；ft_helps=False

| condition | val | test |
|---|---|---|
| train_mean | tMAE=0.00411m rMAE=1.484deg | tMAE=0.00414m rMAE=1.501deg |
| phase_mean | tMAE=0.00429m rMAE=1.569deg | tMAE=0.00436m rMAE=1.560deg |
| current_command | tMAE=0.00443m rMAE=1.494deg | tMAE=0.00408m rMAE=1.499deg |
| time_index | tMAE=0.00412m rMAE=1.505deg | tMAE=0.00420m rMAE=1.506deg |
| A_H1 | tMAE=0.00441m rMAE=1.753deg | tMAE=0.00421m rMAE=1.700deg |
| A_H8 | tMAE=0.00600m rMAE=2.559deg | tMAE=0.00632m rMAE=2.613deg |
| B_H1 | tMAE=0.00492m rMAE=1.871deg | tMAE=0.00442m rMAE=1.882deg |
| B_H8 | tMAE=0.00716m rMAE=3.155deg | tMAE=0.00587m rMAE=2.657deg |
| B_H8_shuffled_FT | tMAE=0.00721m rMAE=3.280deg | tMAE=0.00621m rMAE=2.504deg |
| privileged_o2h_causal_ceiling | tMAE=0.00373m rMAE=1.312deg | tMAE=0.00308m rMAE=1.054deg |

- A_H8−A_H1 test 配对：trans mean=0.00211 CI[0.00075,0.00427]；rot mean=0.913 CI[0.362,1.708]

- B_H8_vs_A_H8_test: trans CI[-0.00237,0.00109] sig=False；rot CI[-0.313,0.392] sig=False
- B_H8_vs_B_H8_shuffled_FT_test: trans CI[-0.00078,-0.00000] sig=True；rot CI[-0.021,0.370] sig=False

## 研究判断规则

- 若 A/B 未能在未来漂移上、以配对 CI 击败全部代理基线 → **停止 A/B sensing 路线**。
- 若 FT 仍不优于 shuffled-FT → 腕力时序信息未证实。
- 不自动换复杂网络；不训策略；不宣称 Obs P0。

## 最终研究判断

- **结束的是窄配置，不是广义 sensing 全盘证伪**：
  > 在当前单几何稳定 demo 数据上，使用截至 `t` 的 `act44 command history + wrist FT`、且**不包含未来动作条件**的线性 future-drift forecast，正式结束。
- **不得继续声称**：广义 proprioception 已反证；所有 A/B sensing 已反证；下一步只能触觉/视觉/停项。
- **正式表述**：`act44_command + wrist_ft` 的当前被动线性预测配置结束；**广义 action-conditioned observability 尚未判定**。
- 原因边界：B1 无未来动作输入；`act44`≠独立 `q/qdot`；稳定 demo 低激励/floor；缺未来动作的 privileged ceiling 不能单证 sensing 失败。
- 不重跑 B1、不改写下方原始指标表；主线转入 **P0-C2 Controllability**（见 `docs/PROGRESS.md`）。

## Guards

- {"WRITE_IMPLEMENTATION_ENABLED": false, "evaluation_only": true, "allow_policy_training": false, "claims_observability_p0_pass": false, "no_policy_training": true, "no_new_collection": true, "no_pilot_write": true, "no_complex_networks": true}
