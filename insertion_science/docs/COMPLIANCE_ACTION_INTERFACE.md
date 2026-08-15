# Compliance Action Interface & Training Hard Gates

- 日期：2026-08-15（修正）
- 状态：**冻结**。Causal P0 ≠ 效用通过；**禁止**实现 wrapper / 开训。
- 门控：须先通过 `COMPLIANCE_UTILITY_ORACLE_P0.md` 判定树第 3 支。

## Causal P0 只证明了什么

刚度改变物理结果。全部分支 `insert_ok=0`；降刚度多数减少 `tip_progress`。
**没有**证明任务收益，也**没有**证明动态 compliance 优于固定全局刚度。

## 当前正确结论

> 因果效应通过，任务效用尚未通过。暂不放弃，但不进入 wrapper/训练。

## 动作接口 / 训练硬门

草案保留作参考，**在 Utility/Oracle P0 通过前一律不实施**。
若 Oracle 失败或仅固定刚度更优，删除本草案对应训练路径。
