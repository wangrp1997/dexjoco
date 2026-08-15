# Controller Compliance Causal P0 Result

- 完成时间：2026-08-15T08:28:12Z
- 判定：`pass_compliance_causal_effect`（**仅因果**）
- 因果通过：True
- 任务效用通过：False（尚未测试）
- 停止训练/wrapper：True（效用未过前禁止）
- 摘要：held-out 上刚度改变物理结果；**全部 `insert_ok=0`**；降刚度多数减少
  `tip_progress`。不得据此进入 wrapper 或训练。

## 结论降级（2026-08-15 修正）

- 已证明：刚度是真实因果变量。
- **未证明**：有任务收益；动态 compliance 优于固定全局刚度。
- 下一步：`Compliance Utility/Oracle P0`（见 `COMPLIANCE_UTILITY_ORACLE_P0.md`）。

## Restore

- ok: True

## Held-out scale reports

```json
[
  {
    "stiffness_scale": 0.5,
    "action": "hold",
    "n_heldout": 2,
    "n_existence": 2,
    "direction_consistent_keys": [
      "trans_drift_max_m",
      "rot_drift_max_rad",
      "tip_progress_m"
    ],
    "direction_ok": true,
    "all_effects_harmful_only": false,
    "pass": true
  },
  {
    "stiffness_scale": 0.5,
    "action": "demo_matched",
    "n_heldout": 2,
    "n_existence": 2,
    "direction_consistent_keys": [
      "contact_force_mean_n"
    ],
    "direction_ok": true,
    "all_effects_harmful_only": false,
    "pass": true
  },
  {
    "stiffness_scale": 0.25,
    "action": "hold",
    "n_heldout": 2,
    "n_existence": 2,
    "direction_consistent_keys": [
      "trans_drift_max_m",
      "rot_drift_max_rad",
      "wrist_ft_mean_n",
      "tip_progress_m"
    ],
    "direction_ok": true,
    "all_effects_harmful_only": false,
    "pass": true
  },
  {
    "stiffness_scale": 0.25,
    "action": "demo_matched",
    "n_heldout": 2,
    "n_existence": 2,
    "direction_consistent_keys": [
      "contact_force_mean_n",
      "tip_progress_m"
    ],
    "direction_ok": true,
    "all_effects_harmful_only": false,
    "pass": true
  }
]
```

## Row index (compact)

```json
[
  {
    "role": "discovery",
    "episode_index": 4,
    "frame": 484,
    "action": "hold",
    "stiffness_scale": 1.0,
    "existence": false,
    "hit_metrics": [],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.015348924104592288,
      "rot_drift_max_rad": 0.10583498109188677,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 5.796094199580552,
      "contact_force_mean_n": 44.35832250309566,
      "tip_progress_m": 0.007563006205999763,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.05239676773570449,
      "lat_progress_m": -0.002658952277492693,
      "wrist_ft_max_n": 6.640394070787611,
      "contact_force_max_n": 50.79148955886739,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "discovery",
    "episode_index": 4,
    "frame": 484,
    "action": "hold",
    "stiffness_scale": 0.5,
    "existence": true,
    "hit_metrics": [
      "trans_drift_max_m",
      "rot_drift_max_rad",
      "contact_force_mean_n"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.01074254697909526,
      "rot_drift_max_rad": 0.07458725925448027,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 6.215004913915423,
      "contact_force_mean_n": 42.35414607634641,
      "tip_progress_m": 0.007141915298743681,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.05281785864296057,
      "lat_progress_m": -0.002738065373316846,
      "wrist_ft_max_n": 6.751604219583955,
      "contact_force_max_n": 46.91260927829288,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "discovery",
    "episode_index": 4,
    "frame": 484,
    "action": "hold",
    "stiffness_scale": 0.25,
    "existence": true,
    "hit_metrics": [
      "trans_drift_max_m",
      "rot_drift_max_rad",
      "contact_force_mean_n",
      "tip_progress_m"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.007435511776654388,
      "rot_drift_max_rad": 0.05695035950372396,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 6.438569189107394,
      "contact_force_mean_n": 42.08709416276087,
      "tip_progress_m": 0.005648825276610897,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.05431094866509335,
      "lat_progress_m": -0.002783243925549703,
      "wrist_ft_max_n": 6.894296546534215,
      "contact_force_max_n": 44.27657091606014,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "discovery",
    "episode_index": 4,
    "frame": 484,
    "action": "demo_matched",
    "stiffness_scale": 1.0,
    "existence": false,
    "hit_metrics": [],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.02082382193281851,
      "rot_drift_max_rad": 0.1404514096011792,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 5.356167900317926,
      "contact_force_mean_n": 47.90081673186563,
      "tip_progress_m": 0.015215622787898314,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.044744151153805936,
      "lat_progress_m": -0.002883189578145898,
      "wrist_ft_max_n": 6.640261767670845,
      "contact_force_max_n": 89.7757350558941,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "discovery",
    "episode_index": 4,
    "frame": 484,
    "action": "demo_matched",
    "stiffness_scale": 0.5,
    "existence": true,
    "hit_metrics": [
      "trans_drift_max_m",
      "rot_drift_max_rad",
      "contact_force_mean_n",
      "tip_progress_m"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.017791429521996204,
      "rot_drift_max_rad": 0.11972157263029543,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 5.520586418157035,
      "contact_force_mean_n": 43.27795985039716,
      "tip_progress_m": 0.013959979946220627,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.04599979399548362,
      "lat_progress_m": -0.0026384497681438417,
      "wrist_ft_max_n": 6.607435670708237,
      "contact_force_max_n": 49.025731427590145,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "discovery",
    "episode_index": 4,
    "frame": 484,
    "action": "demo_matched",
    "stiffness_scale": 0.25,
    "existence": true,
    "hit_metrics": [
      "trans_drift_max_m",
      "rot_drift_max_rad",
      "contact_force_mean_n",
      "tip_progress_m"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.012225670375008685,
      "rot_drift_max_rad": 0.08046560950890767,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 6.096089467635776,
      "contact_force_mean_n": 44.969830391435785,
      "tip_progress_m": 0.011365001606669,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.04859477233503525,
      "lat_progress_m": -0.002879734210791267,
      "wrist_ft_max_n": 6.7117580688308065,
      "contact_force_max_n": 86.82024282149771,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "discovery",
    "episode_index": 6,
    "frame": 271,
    "action": "hold",
    "stiffness_scale": 1.0,
    "existence": false,
    "hit_metrics": [],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.017527573644835585,
      "rot_drift_max_rad": 0.09943557848109745,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 5.9560661872030165,
      "contact_force_mean_n": 33.80939067520919,
      "tip_progress_m": 0.07613318420540993,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.16041203701768178,
      "lat_progress_m": 0.0969727165343344,
      "wrist_ft_max_n": 6.1447660706844465,
      "contact_force_max_n": 33.907179465491474,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "discovery",
    "episode_index": 6,
    "frame": 271,
    "action": "hold",
    "stiffness_scale": 0.5,
    "existence": true,
    "hit_metrics": [
      "tip_progress_m"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.017456450837553223,
      "rot_drift_max_rad": 0.10079222689923516,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 6.128767437999397,
      "contact_force_mean_n": 33.880809895057986,
      "tip_progress_m": 0.06903895735232335,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.16750626387076836,
      "lat_progress_m": 0.07538711125676437,
      "wrist_ft_max_n": 6.304390586787615,
      "contact_force_max_n": 33.97883641401482,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "discovery",
    "episode_index": 6,
    "frame": 271,
    "action": "hold",
    "stiffness_scale": 0.25,
    "existence": true,
    "hit_metrics": [
      "tip_progress_m"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.01689211225851142,
      "rot_drift_max_rad": 0.09828533286559717,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 6.175011180689522,
      "contact_force_mean_n": 33.89958921480372,
      "tip_progress_m": 0.058683762452920785,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.17786145877017093,
      "lat_progress_m": 0.058159773761895434,
      "wrist_ft_max_n": 6.415972565643186,
      "contact_force_max_n": 34.06654002560925,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "discovery",
    "episode_index": 6,
    "frame": 271,
    "action": "demo_matched",
    "stiffness_scale": 1.0,
    "existence": false,
    "hit_metrics": [],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.019335661950821488,
      "rot_drift_max_rad": 0.10516052649250356,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 5.933774708770651,
      "contact_force_mean_n": 33.88400312794605,
      "tip_progress_m": 0.07923367976678139,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.15731154145631032,
      "lat_progress_m": 0.10923777232106925,
      "wrist_ft_max_n": 6.157126524958573,
      "contact_force_max_n": 33.98531188376145,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "discovery",
    "episode_index": 6,
    "frame": 271,
    "action": "demo_matched",
    "stiffness_scale": 0.5,
    "existence": true,
    "hit_metrics": [
      "tip_progress_m"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.01970968706151251,
      "rot_drift_max_rad": 0.10977017313090426,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 6.125227082988976,
      "contact_force_mean_n": 33.99179019035067,
      "tip_progress_m": 0.07393032383060394,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.16261489739248777,
      "lat_progress_m": 0.0843063589975391,
      "wrist_ft_max_n": 6.307983878260376,
      "contact_force_max_n": 34.09505750810566,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "discovery",
    "episode_index": 6,
    "frame": 271,
    "action": "demo_matched",
    "stiffness_scale": 0.25,
    "existence": true,
    "hit_metrics": [
      "tip_progress_m"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.019621810364702855,
      "rot_drift_max_rad": 0.1105687502446237,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 6.177014131152621,
      "contact_force_mean_n": 34.02856486093702,
      "tip_progress_m": 0.0635564549573448,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.1729887662657469,
      "lat_progress_m": 0.06469094969397829,
      "wrist_ft_max_n": 6.43488372474969,
      "contact_force_max_n": 34.16746170782,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "held_out",
    "episode_index": 9,
    "frame": 407,
    "action": "hold",
    "stiffness_scale": 1.0,
    "existence": false,
    "hit_metrics": [],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.01257262710628666,
      "rot_drift_max_rad": 0.07414474025864089,
      "contact_retention_vs_root_mean": 0.96875,
      "wrist_ft_mean_n": 4.456820309668714,
      "contact_force_mean_n": 41.84649562107669,
      "tip_progress_m": 0.013610763590814555,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.04534756623617394,
      "lat_progress_m": -0.0008633156030829728,
      "wrist_ft_max_n": 6.640262975985541,
      "contact_force_max_n": 53.54882025593754,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "held_out",
    "episode_index": 9,
    "frame": 407,
    "action": "hold",
    "stiffness_scale": 0.5,
    "existence": true,
    "hit_metrics": [
      "trans_drift_max_m",
      "rot_drift_max_rad",
      "contact_force_mean_n",
      "tip_progress_m"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.009296060585195467,
      "rot_drift_max_rad": 0.0528754356279878,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 5.280869583379355,
      "contact_force_mean_n": 38.292667577028936,
      "tip_progress_m": 0.010937589811912625,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.04802074001507587,
      "lat_progress_m": -0.0008626057971874187,
      "wrist_ft_max_n": 6.815599956626116,
      "contact_force_max_n": 45.833206945053064,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "held_out",
    "episode_index": 9,
    "frame": 407,
    "action": "hold",
    "stiffness_scale": 0.25,
    "existence": true,
    "hit_metrics": [
      "trans_drift_max_m",
      "rot_drift_max_rad",
      "wrist_ft_mean_n",
      "contact_force_mean_n",
      "tip_progress_m"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.0066700494483673545,
      "rot_drift_max_rad": 0.03639589212019999,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 6.043791441101391,
      "contact_force_mean_n": 35.557030222646944,
      "tip_progress_m": 0.010328733279864097,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.0486295965471244,
      "lat_progress_m": -0.0008616831625650988,
      "wrist_ft_max_n": 7.139632219933653,
      "contact_force_max_n": 44.629394137260945,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "held_out",
    "episode_index": 9,
    "frame": 407,
    "action": "demo_matched",
    "stiffness_scale": 1.0,
    "existence": false,
    "hit_metrics": [],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.012567052881022608,
      "rot_drift_max_rad": 0.062473291309143995,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 5.704573005554213,
      "contact_force_mean_n": 34.55447823142852,
      "tip_progress_m": 0.015570844666169356,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.04338748516081914,
      "lat_progress_m": -0.0009429913056685463,
      "wrist_ft_max_n": 6.616933582474032,
      "contact_force_max_n": 38.2437244342513,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "held_out",
    "episode_index": 9,
    "frame": 407,
    "action": "demo_matched",
    "stiffness_scale": 0.5,
    "existence": true,
    "hit_metrics": [
      "trans_drift_max_m",
      "rot_drift_max_rad",
      "contact_force_mean_n"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.019393979829636733,
      "rot_drift_max_rad": 0.09942984277510665,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 5.104219712608606,
      "contact_force_mean_n": 37.68566748651386,
      "tip_progress_m": 0.015667682633916345,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.04329064719307215,
      "lat_progress_m": -0.0012575210478556523,
      "wrist_ft_max_n": 6.788090798393752,
      "contact_force_max_n": 45.90523715949736,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "held_out",
    "episode_index": 9,
    "frame": 407,
    "action": "demo_matched",
    "stiffness_scale": 0.25,
    "existence": true,
    "hit_metrics": [
      "trans_drift_max_m",
      "rot_drift_max_rad",
      "contact_force_mean_n",
      "tip_progress_m"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.017117089635486835,
      "rot_drift_max_rad": 0.08533326103549993,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 5.78182338826388,
      "contact_force_mean_n": 37.3734699525026,
      "tip_progress_m": 0.01385278513057539,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.045105544696413105,
      "lat_progress_m": -0.0008593813249633958,
      "wrist_ft_max_n": 7.091428966595594,
      "contact_force_max_n": 68.29808791456475,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "held_out",
    "episode_index": 3,
    "frame": 431,
    "action": "hold",
    "stiffness_scale": 1.0,
    "existence": false,
    "hit_metrics": [],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.011488052069291797,
      "rot_drift_max_rad": 0.06562562570572143,
      "contact_retention_vs_root_mean": 0.989010989010989,
      "wrist_ft_mean_n": 6.443205530518275,
      "contact_force_mean_n": 32.41699032449498,
      "tip_progress_m": 0.014671993965823478,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.04499824996373053,
      "lat_progress_m": 0.0010597950591634034,
      "wrist_ft_max_n": 6.544980314979389,
      "contact_force_max_n": 33.091038731444286,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "held_out",
    "episode_index": 3,
    "frame": 431,
    "action": "hold",
    "stiffness_scale": 0.5,
    "existence": true,
    "hit_metrics": [
      "trans_drift_max_m",
      "tip_progress_m"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.009827471591989113,
      "rot_drift_max_rad": 0.0563684070659546,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 6.465848725792022,
      "contact_force_mean_n": 32.54452829224944,
      "tip_progress_m": 0.012916956871864511,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.0467532870576895,
      "lat_progress_m": 0.0010131732732007576,
      "wrist_ft_max_n": 6.61352315610852,
      "contact_force_max_n": 32.65384976018791,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "held_out",
    "episode_index": 3,
    "frame": 431,
    "action": "hold",
    "stiffness_scale": 0.25,
    "existence": true,
    "hit_metrics": [
      "trans_drift_max_m",
      "rot_drift_max_rad",
      "tip_progress_m"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.007257517642311296,
      "rot_drift_max_rad": 0.04580937136069985,
      "contact_retention_vs_root_mean": 0.9910714285714286,
      "wrist_ft_mean_n": 6.467113248068535,
      "contact_force_mean_n": 32.69345124118552,
      "tip_progress_m": 0.009672639666255145,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.04999760426329886,
      "lat_progress_m": 0.0010308996773416312,
      "wrist_ft_max_n": 6.792844577852504,
      "contact_force_max_n": 32.98804347802312,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "held_out",
    "episode_index": 3,
    "frame": 431,
    "action": "demo_matched",
    "stiffness_scale": 1.0,
    "existence": false,
    "hit_metrics": [],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.01273474617127762,
      "rot_drift_max_rad": 0.08568811050142619,
      "contact_retention_vs_root_mean": 0.987012987012987,
      "wrist_ft_mean_n": 6.453762183198387,
      "contact_force_mean_n": 32.55287802348226,
      "tip_progress_m": 0.015596741294661147,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.04407350263489286,
      "lat_progress_m": 0.00020380539778961163,
      "wrist_ft_max_n": 6.650946884873549,
      "contact_force_max_n": 32.80036992527456,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "held_out",
    "episode_index": 3,
    "frame": 431,
    "action": "demo_matched",
    "stiffness_scale": 0.5,
    "existence": true,
    "hit_metrics": [
      "trans_drift_max_m"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.011699902499442149,
      "rot_drift_max_rad": 0.08274506384933124,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 6.467130689340883,
      "contact_force_mean_n": 32.67490271018042,
      "tip_progress_m": 0.015166072839516119,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.04450417109003789,
      "lat_progress_m": 0.0001562735067244204,
      "wrist_ft_max_n": 6.5696014046540725,
      "contact_force_max_n": 32.8965724232523,
      "nonfinite_obs": 0.0
    }
  },
  {
    "role": "held_out",
    "episode_index": 3,
    "frame": 431,
    "action": "demo_matched",
    "stiffness_scale": 0.25,
    "existence": true,
    "hit_metrics": [
      "trans_drift_max_m",
      "tip_progress_m"
    ],
    "harmful_only": false,
    "metrics_mean": {
      "trans_drift_max_m": 0.009906648564292577,
      "rot_drift_max_rad": 0.08102732043204247,
      "contact_retention_vs_root_mean": 1.0,
      "wrist_ft_mean_n": 6.4476105705642155,
      "contact_force_mean_n": 32.82778653884959,
      "tip_progress_m": 0.013278759705269684,
      "jam_proxy": 0.0,
      "terminal_peg_ok": 1.0,
      "insert_ok_end": 0.0,
      "tip_end_m": 0.046391484224284324,
      "lat_progress_m": 0.00018231974471631923,
      "wrist_ft_max_n": 6.682604548030125,
      "contact_force_max_n": 33.320180555828266,
      "nonfinite_obs": 0.0
    }
  }
]
```

## 决策

- P0 通过：下一步可设计将 compliance/gains 纳入动作接口的方案与训练硬门。
- 仍禁止直接开训。
