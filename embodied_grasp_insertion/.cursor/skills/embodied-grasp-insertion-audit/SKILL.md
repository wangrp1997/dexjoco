---
name: embodied-grasp-insertion-audit
description: Hard-gated audit/innovation loop for embodied_grasp_insertion. Enforce Observability, Controllability, and Semantic P0 gates before any policy; never return to 37D+wrist12 recovery; never treat MuJoCo privilege as deployment observation; never call fixed peg/socket tip error general insertion semantics. Use on wake, P0 audits, or when asked to continue grasp-insertion work.
---

# Embodied Grasp-Insertion Audit Skill

## 每次开始前（硬门）

1. 读取本项目 `outputs/state.json`；若 `busy=true`，先报告运行中任务，不另开长训。
2. 读取 `docs/PROGRESS.md` 与 `docs/MOTIVATION_AND_PLAN.md`。
3. 复习 `prior_failures_reviewed` / `why_not_repeat`，禁止重复已证伪族。
4. **任何 policy / critic / actor / Diffusion / Flow / Transformer 训练前**，必须三项硬门全部通过：
   - Observability P0
   - Controllability P0
   - Semantic P0
   - 正式验收口径：`docs/P0_ACCEPTANCE_CHECKLIST.md`（输入/标签/split/通过与停止）
5. 未过门时只允许：只读审计、matched intervention smoke、最小定向重放补标签。
6. 真实写盘仅在用户明确发送「授权真实写入 1 条」后；当前默认**不授权**（pilot=`insert_phase=skipped` 对算法增益小）。

## 绝对禁止

- 回到 `37D observation + wrist12` recovery actor / critic。
- 把 simulator privilege（peg7/tray7、tip/axis、flags、cfrc 指力）写成部署观测。
- 把固定 peg/socket 的相对 tip/lateral/axis 误差称为“通用插孔语义”。
- 用解析标签、gate、residual、servo、在线候选搜索或 MPPI 当部署策略。
- 把结论写回 `recovery_trajectory_policy` 或 `contact_cmdp_recovery_policy`；**所有新结论只写入** `embodied_grasp_insertion/`。
- 伪造无法从代码或数据确认的字段与统计。
- 未确认前扩采全量数据或开长训。

## P0-A / P0-C0 / P0-C1 / P0-C1.1 / P0-S0 之后的默认动作

- 硬门现状：
  - P0-C0 = scoped `harmful_only`
  - P0-C1 = `no_effect`
  - P0-C1.1 = `load_calibration_fail`（已停止）
  - P0-S0 = `pass`（临时多几何 plumbing；≠ Semantic P0）
- P0-S0.1 = `numerical_execution_pass / physical_settle_invalid`
- P0-S0.1b = `support_settle` 可信；clearance provisional（已废）
- P0-S0.1c = `pass`（纯几何 clearance）
- P0-S0.2 = `pass`（正式 arena 参数化 + 8 族 reset smoke）
- P0-S0.3 = `pass`（单目标孔 metadata/plumbing；`claims_policy_knows_hole=false`）
- P0-S0.3b = `pass`（同 family 双 socket instance/site/pose；仍非策略知孔）
- P0-S0.4 = `pass`（**instrumentation / 阶段编排** only；palm-snap；`claims_physical_grasp_stability=false`）
- P0-S0.4b = `pass`（demo transport root 纯动力学物理抓取；仅 round_8mm ep0/2/4）
- P0-S0.4c = `pass`（4 族物理抓取 root：round_8mm=demo，其余=oracle-once+动力学；`claims_stable_grasp_policy=false`）
- **禁止**：调 load/pulse；finger smoke；训练；全量采集。
- **当前阶段**：`c2_inconclusive_heterogeneous_forks_actuation_unverified`（**撤回 A**）；跑 C2-S1b 执行/分叉审计；不进 Stage-2；不训策略；不停项。
- 保留 `outputs/p0_c2_stage1_v1/`；禁止用跨 root 同号均值单独否定因果存在。
- 尺寸口径以 `docs/GEOMETRY_ASSET_AUDIT.md` 为准；禁止复制 8mm collision 到其他族时勿覆盖官方 8mm 文件。
- 更新 `outputs/state.json`（`busy=false`）与 `docs/PROGRESS.md`。

## 关键产物

- `docs/GEOMETRY_ASSET_AUDIT.md`
- `docs/GEOMETRY_COMPILE_SMOKE.md`
- `docs/GEOMETRY_ENV_SETTLE_SMOKE.md`（S0.1，物理 settle 作废）
- `docs/GEOMETRY_ENV_SETTLE_SMOKE_V1B.md`（clearance provisional）
- `docs/GEOMETRY_ENV_SETTLE_SMOKE_V1C.md`
- `docs/FORMAL_ARENA_RESET_SMOKE.md`
- `docs/TARGET_HOLE_SEMANTICS_SMOKE.md`
- `docs/TARGET_HOLE_MULTI_INSTANCE_SMOKE.md`
- `docs/GRASP_STABILITY_SMOKE.md`
- `docs/GRASP_STABILITY_PHYSICAL_SMOKE.md`
- `docs/GRASP_STABILITY_MULTIFAMILY_PHYSICAL_SMOKE.md`
- `docs/SEMANTIC_DATA_DESIGN.md`
- `docs/P0_ACCEPTANCE_CHECKLIST.md`
- `docs/P0_LABEL_DERIVABILITY_AUDIT.md`
- `docs/PRIVILEGED_LABEL_SCHEMA_V1.md`
- `docs/OBSERVABILITY_PRIVILEGED_LABEL_SMOKE.md`
- `docs/OBSERVABILITY_DATA_DESIGN.md`
- `docs/OBSERVABILITY_DATASET_FEASIBILITY.md`
- `data/manifests/observability_dataset_feasibility_v1.json`
- `data/manifests/label_derivability_audit_v1.json`
- `data/manifests/observability_privileged_label_smoke_v1.json`
- `configs/geometry_families.yaml`
- `data/manifests/geometry_*` / `formal_arena_reset_smoke_v1.json` / `target_hole_semantics_smoke_v1.json` / `target_hole_multi_instance_smoke_v1.json` / `grasp_stability_smoke_v1.json` / `grasp_stability_physical_smoke_v1.json` / `grasp_stability_multifamily_physical_smoke_v1.json`
- `dexjoco/sim/envs/assembly_geometry.py`
- （保留）P0-C0/C1/C1.1 全部旧产物

## 运行约定

- conda 环境：`dexjoco`
- `MUJOCO_GL=egl`；不抢占正在训练的 GPU
- 审计优先 `--metadata-only`；MuJoCo 最多单 episode smoke
- 不启动 tmux 长任务，除非用户明确要求且硬门允许

## 输出位置

- `outputs/state.json`
- `data/manifests/`
- `docs/SYSTEM_IDENTIFIABILITY_AUDIT.md`
- `docs/PROGRESS.md`
