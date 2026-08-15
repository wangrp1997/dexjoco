# Compliance Utility / Oracle P0 Result

- 完成：2026-08-15T08:42:09Z
- 判定：`abandon_compliance`
- 分支：1
- 允许 wrapper：False
- 只改默认增益：False
- 摘要：Oracle 在 held-out 上不优于 baseline；放弃 compliance 方案。

## Oracle vs baseline

```json
{
  "mean_tip": 0.015169186430482602,
  "mean_baseline_tip": 0.014862585879367134,
  "mean_insert": 0.0,
  "mean_baseline_insert": 0.0,
  "n_beats_tip": 0,
  "n_beats_insert": 0,
  "n_jam_improved": 0
}
```

## Best fixed

```json
{
  "gain_name": "axial_hard_lat_soft",
  "mean_insert_ok": 0.0,
  "mean_tip_progress_m": 0.014923426383009367,
  "mean_jam_proxy": 0.0,
  "mean_retention": 0.9976190476190476,
  "retention_ok_vs_baseline": true,
  "score": [
    0.0,
    0.014923426383009367,
    -0.0,
    0.9976190476190476
  ]
}
```

## Oracle picks (held-out)

```json
[
  {
    "episode_index": 9,
    "frame": 407,
    "action": "hold",
    "gain_name": "axial_hard_lat_soft",
    "metrics": {
      "insert_ok_end": false,
      "tip_progress_m": 0.01456914293443766,
      "lat_progress_m": -0.0009143775658506429,
      "jam_proxy": false,
      "contact_retention_vs_root_mean": 1.0,
      "object_dropped_proxy": false,
      "wrist_ft_mean_n": 5.614231173490067,
      "contact_force_mean_n": 38.519289784901865,
      "nonfinite_obs": false
    },
    "baseline_metrics": {
      "insert_ok_end": false,
      "tip_progress_m": 0.013610763590814555,
      "lat_progress_m": -0.0008633156030829728,
      "jam_proxy": false,
      "contact_retention_vs_root_mean": 0.96875,
      "object_dropped_proxy": false,
      "wrist_ft_mean_n": 4.456820309668714,
      "contact_force_mean_n": 41.84649562107669,
      "nonfinite_obs": false
    },
    "reason": "max_utility_among_retention_ok",
    "beats_baseline_tip": false,
    "beats_baseline_insert": false,
    "jam_improved": false
  },
  {
    "episode_index": 9,
    "frame": 407,
    "action": "demo_matched",
    "gain_name": "soft_iso",
    "metrics": {
      "insert_ok_end": false,
      "tip_progress_m": 0.015667682633916345,
      "lat_progress_m": -0.0012575210478556523,
      "jam_proxy": false,
      "contact_retention_vs_root_mean": 1.0,
      "object_dropped_proxy": false,
      "wrist_ft_mean_n": 5.104219712608606,
      "contact_force_mean_n": 37.68566748651386,
      "nonfinite_obs": false
    },
    "baseline_metrics": {
      "insert_ok_end": false,
      "tip_progress_m": 0.015570844666169356,
      "lat_progress_m": -0.0009429913056685463,
      "jam_proxy": false,
      "contact_retention_vs_root_mean": 1.0,
      "object_dropped_proxy": false,
      "wrist_ft_mean_n": 5.704573005554213,
      "contact_force_mean_n": 34.55447823142852,
      "nonfinite_obs": false
    },
    "reason": "max_utility_among_retention_ok",
    "beats_baseline_tip": false,
    "beats_baseline_insert": false,
    "jam_improved": false
  },
  {
    "episode_index": 3,
    "frame": 431,
    "action": "hold",
    "gain_name": "axial_hard_lat_med",
    "metrics": {
      "insert_ok_end": false,
      "tip_progress_m": 0.014843178858915257,
      "lat_progress_m": 0.0007642380800072048,
      "jam_proxy": false,
      "contact_retention_vs_root_mean": 0.9897959183673469,
      "object_dropped_proxy": false,
      "wrist_ft_mean_n": 6.417066194921536,
      "contact_force_mean_n": 32.43916908043294,
      "nonfinite_obs": false
    },
    "baseline_metrics": {
      "insert_ok_end": false,
      "tip_progress_m": 0.014671993965823478,
      "lat_progress_m": 0.0010597950591634034,
      "jam_proxy": false,
      "contact_retention_vs_root_mean": 0.989010989010989,
      "object_dropped_proxy": false,
      "wrist_ft_mean_n": 6.443205530518275,
      "contact_force_mean_n": 32.41699032449498,
      "nonfinite_obs": false
    },
    "reason": "max_utility_among_retention_ok",
    "beats_baseline_tip": false,
    "beats_baseline_insert": false,
    "jam_improved": false
  },
  {
    "episode_index": 3,
    "frame": 431,
    "action": "demo_matched",
    "gain_name": "baseline",
    "metrics": {
      "insert_ok_end": false,
      "tip_progress_m": 0.015596741294661147,
      "lat_progress_m": 0.00020380539778961163,
      "jam_proxy": false,
      "contact_retention_vs_root_mean": 0.987012987012987,
      "object_dropped_proxy": false,
      "wrist_ft_mean_n": 6.453762183198387,
      "contact_force_mean_n": 32.55287802348226,
      "nonfinite_obs": false
    },
    "baseline_metrics": {
      "insert_ok_end": false,
      "tip_progress_m": 0.015596741294661147,
      "lat_progress_m": 0.00020380539778961163,
      "jam_proxy": false,
      "contact_retention_vs_root_mean": 0.987012987012987,
      "object_dropped_proxy": false,
      "wrist_ft_mean_n": 6.453762183198387,
      "contact_force_mean_n": 32.55287802348226,
      "nonfinite_obs": false
    },
    "reason": "max_utility_among_retention_ok",
    "beats_baseline_tip": false,
    "beats_baseline_insert": false,
    "jam_improved": false
  }
]
```

## 决策

- **放弃** compliance 方案。
- 不实现 wrapper，不开训。
