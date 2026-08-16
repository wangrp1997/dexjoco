# Intermediate Physical Event Label P0 — Result

- UTC: `2026-08-16T03:24:19Z`
- Protocol: `IntermediateEventLabelP0`
- Verdict: `fail_no_stable_intermediate_event_label`
- Decision: `stop_dense_intermediate_label_direction`
- Reason: no preregistered event passed split, non-copy, and external gates
- Data: primary=`216` (discovery=108, held_out=108), external=`32`

## Events

- `approach_100mm` pass=`False`; discovery prev/lift=`0.796/0.616`; held_out=`0.796/0.523`; external lift=`0.893`; FP/FN=`74/0`
- `approach_60mm` pass=`False`; discovery prev/lift=`0.519/0.946`; held_out=`0.426/0.978`; external lift=`1.000`; FP/FN=`4/0`
- `deep_20mm` pass=`False`; discovery prev/lift=`0.500/0.981`; held_out=`0.417/1.000`; external lift=`1.000`; FP/FN=`1/0`
- `retained` pass=`False`; discovery prev/lift=`0.935/0.525`; held_out=`0.954/0.437`; external lift=`0.806`; FP/FN=`106/0`
- `retained_and_approach_60mm` pass=`False`; discovery prev/lift=`0.519/0.946`; held_out=`0.426/0.978`; external lift=`1.000`; FP/FN=`4/0`

## Note

本结果不复活 handoff 方向，不运行任何旧项目，只审计既有 insertion_science 输出。
通过仅允许检查事件时间顺序与因果干预，不允许策略训练。
