# Active Contact Probe Information P0 — Result

- UTC: `2026-08-16T03:39:26Z`
- Protocol: `ActiveContactProbeInformationP0`
- Verdict: `fail_no_active_probe_information_gain`
- Decision: `stop_active_probe_information_direction`
- static accuracy: `0.250`
- sequence accuracy: `0.167`
- gain over static: `-0.083`
- per-root sequence: `{'3': 0.25, '9': 0.0, '13': 0.25}`
- shuffle mean: `0.242`
- checks: `{'sequence_accuracy_ok': False, 'per_root_ok': False, 'gain_over_static_ok': False, 'shuffle_ok': True}`

## Note

只运行 insertion_science matched snapshot 微探针；未调用 HybridInsert/skill_replay，未训练策略。
