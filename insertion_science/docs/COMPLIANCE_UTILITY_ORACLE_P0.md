# Compliance Utility / Oracle P0

- 日期：2026-08-15
- 状态：已执行；判定 `abandon_compliance`
- 前置：Causal P0 仅证明刚度改变物理，**任务效用未过**

## 目标

在相同 frozen roots 与相同腕/指动作下，检验合规刚度是否带来**任务收益**，
以及是否需要**状态依赖**的动态刚度（而非一个全局固定值）。

## 禁止

- 不扩采、不训练、不实现 learnable wrapper。
- 不按 episode 事后调参；配置全部预注册。
- 不修改旧项目源码（继续用 `insertion_science` OSC patch）。

## 预注册条件

复用 Causal P0 的 discovery / held-out roots、`hold` 与 `demo_matched`、horizon/repeats。

刚度配置（示例，执行前写入 config 并冻结）：

1. `baseline`：`s=1.0` 各向同性（现行）
2. `soft_iso`：`s=0.5` 各向同性
3. `softer_iso`：`s=0.25` 各向同性
4. 少量**轴向/横向**分解（预注册 2–3 个）：例如轴向硬/横向软、轴向软/横向硬
   （在 task frame 或 wrist frame 定义；禁止事后发明更多格子）

## 评估

对每个 root × action × gain_config 记录：

- `insert_ok`、`tip_progress`、`lat_progress`、`jam_proxy`
- `contact_retention`、`object_dropped_proxy`、腕部/接触载荷
- 数值稳定性

定义：

- **Oracle**：每个 root 在 retention 不下降（相对 baseline 的预注册门）的前提下，
  选使任务效用最好的配置（主：`insert_ok`；次：`tip_progress` 升 / `jam` 降）。
- **Best fixed**：在全部 held-out 上选一个全局固定配置（禁止 per-root 偷看后命名）。

## 判定树

1. **Oracle 在 held-out 上不优于 baseline**（效用无增益，或仅靠 retention 恶化换来）
   → **放弃 compliance 方案**。
2. **某一固定刚度全局优于 baseline，且 Oracle 相对该固定值无显著增益**
   → **只改 controller 默认值**；不做策略、不做 wrapper 训练。
3. **不同状态需要不同刚度，且 Oracle 明显优于 Best fixed**
   → 才进入 compliance wrapper + learnability P0。

## 与 Causal P0 的关系

| 层级 | 状态 |
|------|------|
| 因果效应 | 已通过 |
| 任务效用 / Oracle | **未通过（本 P0）** |
| Wrapper / 训练 | 禁止，直至本 P0 走到判定树第 3 支 |
