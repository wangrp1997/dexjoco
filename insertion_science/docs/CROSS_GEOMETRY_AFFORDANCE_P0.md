# Cross-Geometry Contact-Affordance P0

- 日期：2026-08-15
- 状态：已执行；`fail_stop_affordance_direction`
- 目录：`insertion_science/affordance/`

## 科学问题

固定 tip/lat/axis 标量能否跨截面与尺寸表达接触约束；object–target
contact-affordance 关系是否在 held-out geometry 上更好预测微运动可行性。

## 协议（预注册）

- 复用正式多几何资产（`geometry_families.yaml` + formal XML）；不扩采 episode。
- 每个 family：若干接近位姿 × 预注册 twist；matched restore 后扰动 peg。
- 标签：`free` / `blocked` / `jam`（由 tip 进展、侧偏、接触力判定）。
- 表示：`tip_lat_axis`、`raw_relation`、`contact_affordance`；**禁止** geometry ID 特征。
- 评测：family leave-one-out；socket-instance held-out；shuffled object–target pairing。
- 探针：Logistic/Ridge；不训练控制策略。

## 通过线

1. `contact_affordance` 在**所有** family held-out fold 上优于 `tip_lat_axis`；
2. shuffled pairing 明显下降；
3. socket-instance held-out 上同样不弱于 tip baseline；
4. 若仅同 section/同尺寸有效 → 失败并停止该方向。
