# Demo Handoff Perturbation Recoverability P0 — Result

- UTC: `2026-08-15T12:29:55Z`
- Protocol: `DemoHandoffPerturbRecoverabilityP0`
- Rows: `224`
- Verdict: `branch1_continuous_recoverable_neighborhood`
- Decision: `neighborhood_exists_study_coverage`
- Reason: held_out rate@0.5=0.611, monotone_ok=True

## Rates

- Baseline (identity) insert_ok: `1.000`
- Held-out non-id by scale: `{"0.5": 0.6111111111111112, "1.0": 0.3611111111111111, "2.0": 0.2777777777777778}`
- Discovery non-id by scale: `{"0.5": 0.5555555555555556, "1.0": 0.5, "2.0": 0.4166666666666667}`
- Monotone ok: `True`

## Gates (pre-registered)

- min_baseline_insert_ok_rate: `0.75`
- neighborhood_min_rate_at_0_5: `0.25`
- island_max_rate_at_0_5: `0.05`

## Note

不依赖 PrivHI；不做策略训练；纯 demo continuation + 预注册微扰。
