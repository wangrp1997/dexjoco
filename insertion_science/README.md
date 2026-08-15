# Insertion Science

DexJoCo 插孔研究的新独立入口。

本目录不继承任何过去项目的模型假设，也不以“再训练一个策略”为默认动作。
研究顺序固定为：

1. 审计过去方法及其隐含假设；
2. 提炼尚未解决的科学问题；
3. 检索论文和官方实现；
4. 排除与旧方法同构或近似重复的候选；
5. 为剩余候选设计低算力、可证伪 P0；
6. P0 通过后才讨论实现或训练。

## 项目边界

- 不修改或解冻 `embodied_grasp_insertion/`。
- 不把结论写回 `reach_insert_rl`、`dex_voc_insert`、恢复策略项目或 BotYard。
- 不启动训练、数据扩采或 MuJoCo 大规模 branching，除非科学前提通过且用户明确批准。
- Controller Compliance P0 可通过本目录内 OSC monkeypatch 做小规模 matched rollout；
  仍禁止据此直接开训。
- 不把 residual、gate、servo、候选排名、skill retrieval、force-history、
  trajectory decoder 或换主干重新包装成新方法。

## 文档

- `docs/PRIOR_METHOD_AUDIT.md`：既有项目、基线和实验结果审计。
- `docs/SCIENTIFIC_PROBLEM_MAP.md`：跨项目科学问题、证据和未决问题。
- `docs/SEARCH_PROTOCOL.md`：检索、去重和候选准入规则。
- `docs/SEARCH_LOG.md`：本轮实际查询、论文、代码和排除理由。
- `docs/CANDIDATE_SHORTLIST.md`：检索完成后才允许写入的候选方案。
- `PROGRESS.md`：项目进度与决策。

