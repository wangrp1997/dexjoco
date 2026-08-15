# Search Protocol

## 检索前提

检索以 `SCIENTIFIC_PROBLEM_MAP.md` 的未决问题为查询来源，不以模型名称或流行方法为
起点。每个查询必须包含：物理机制、任务条件和需要验证的科学关系。

## 来源优先级

1. 原始论文和作者官方项目页；
2. 官方代码仓库及其方法实现；
3. 机器人会议正式论文页面；
4. 仅在缺少原始来源时使用综述定位关键词。

技术结论不以博客、二手摘要或搜索结果片段作为唯一依据。

## 去重检查

每个候选必须与 `PRIOR_METHOD_AUDIT.md` 对照以下轴：

- observation/history；
- action/control interface；
- data source and intervention；
- prediction or training target；
- policy class；
- online computation；
- stage decomposition；
- geometry/task semantics；
- evaluation split and success definition。

只换网络、损失名称、编码器、候选数量或执行频率，视为同族重复。

## 硬性排除

- residual、gate、servo、解析 action projection；
- task-frame nearest-neighbor skill retrieval；
- force-history/contact classifier 的直接重做；
- candidate outcome ranking、listwise selector、online branch MPC；
- future state/subgoal/trajectory 到 action 的直接 decoder；
- analytic funnel、DART 或噪声 recovery augmentation；
- 单纯换 Diffusion/Flow/VLA backbone；
- 只在 oracle handoff 或固定几何上验证却声称端到端泛化。

## 候选输出格式

每个候选必须记录：

- `scientific_problem`；
- `hypothesis`；
- `new_causal_variable_or_intervention`；
- `closest_prior_local_methods`；
- `why_not_equivalent`；
- `papers` 与 `official_code`；
- `minimal_p0`；
- `negative_controls`；
- `pass_threshold`；
- `stop_condition`；
- `estimated_compute`。

缺任一字段，不进入 shortlist。

