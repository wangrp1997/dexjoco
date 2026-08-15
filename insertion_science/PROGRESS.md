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

## 下一步

1. 审查阶段接口与数据支持（grasp→handoff→insert 误差传递、接触态覆盖）。
2. 不训练策略；不抢救 compliance / affordance。
3. `training_allowed=false`。
