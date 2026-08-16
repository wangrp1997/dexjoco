# Insertion Science Progress

## 2026-08-15：Cross-Geometry Contact-Affordance P0

- 判定：`fail_stop_affordance_direction` → **停止该方向**。
- 6 族 × 位姿 × twist；family LOO + instance holdout + shuffle。
- `free` 仅 2/192，探针近退化；仍未通过预注册门槛。
- 报告：`docs/CROSS_GEOMETRY_AFFORDANCE_P0_RESULT.md`
- 模块：`affordance/`

## 2026-08-15：放弃动态 compliance

- Utility/Oracle → `abandon_compliance`。
- 不是放弃整个插孔研究。

## 2026-08-15：Demo Handoff Perturbation Recoverability P0

- 判定：`branch1_continuous_recoverable_neighborhood` → `neighborhood_exists_study_coverage`
- Baseline identity insert_ok：`1.0`（8/8）
- Held-out non-id：0.5→0.611，1.0→0.361，2.0→0.278（单调下降）
- 报告：`docs/HANDOFF_PERTURB_RECOVERABILITY_P0_RESULT.md`

## 2026-08-16：Handoff Recoverability Boundary / Coverage P0

- 判定：`branch1_fails_mostly_outside_boundary` → `redesign_handoff_data_generation`
- outside_frac：`0.929`（14 个失败里 13 个在边界外）
- 边界很脆：axis 最大可恢复 scale=`0`；tip_lat 保守 limit=`0.5`
- 报告：`docs/HANDOFF_RECOVERABILITY_BOUNDARY_COVERAGE_P0_RESULT.md`

## 2026-08-16：Handoff Datagen Redesign P0

- 判定：`fail_no_contrast` → `pause_boundary_not_operational`（预注册生死门未过）
- 盆地内 insert_ok：`0.917`（24 条，接受 22）→ 生成能力本身可用
- 盆地外 insert_ok：`0.375` > `0.35`：3/3 `along_p_far` 仍成功；`lat_px_far`/`axis_tx_far` 全失败
- 旧失败 coverage 仍 ~0.91 在外
- 报告：`docs/HANDOFF_DATAGEN_REDESIGN_P0_RESULT.md`
- `training_allowed=false`

## 2026-08-16：Upstream Handoff Targeting P0（最终生死门）

- 判定：`fail_stop_handoff_direction` → **停止整个 handoff 方向**
- 12 seeds 无 force-demo：`handoff_rate=0`（0/12）；多数 `tray_lift_hold_unstable`，seed0 `handoff_never`
- 未进入盆地判定阶段；端到端 insert 全失败
- 报告：`docs/UPSTREAM_HANDOFF_TARGETING_P0_RESULT.md`
- 不跑 along_far P0.1；不训练；不抢救 PrivHI

## 2026-08-16：Intermediate Physical Event Label P0

- 只读审计：primary 216 条（discovery/held-out 各 108），external 32 条
- 判定：`fail_no_stable_intermediate_event_label` → `stop_dense_intermediate_label_direction`
- 100/60/20mm 与 retained 事件均未通过预注册的 non-copy + split + external 全门
- 零 false-negative 表明这些阈值更像 success 必要条件；因缺少 timestamp，不能宣称因果中间标签
- 报告：`docs/INTERMEDIATE_EVENT_LABEL_P0_RESULT.md`

## 2026-08-16：Active Contact Probe Information P0

- 6 个 frozen pre-insert roots；3 discovery / 3 episode-held-out
- 4 类 ±X/±Y matched 错位；固定 8-step 微探针；仅 wrist wrench + contact counts
- static accuracy=`0.250`；sequence=`0.167`；gain=`-0.083`；shuffle mean=`0.242`
- held-out per-root=`0.25/0.00/0.25`，全部低于预注册门槛
- 判定：`fail_no_active_probe_information_gain` → `stop_active_probe_information_direction`
- 报告：`docs/ACTIVE_CONTACT_PROBE_INFORMATION_P0_RESULT.md`

## 下一步

1. **Insertion Science 当前轮正式暂停：没有剩余已准入候选。**
2. `training_allowed=false`、`collection_allowed=false`。
3. 不抢救 compliance / affordance / handoff / dense-event / active-probe 失败方向。
4. 仅当出现新的外部科学假设，且能证明不重复禁区并可低算力证伪时，才允许重开 shortlist。
