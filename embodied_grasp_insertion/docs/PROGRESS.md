# Embodied Grasp-Insertion 进展

## 当前状态

- 日期：2026-08-14
- 阶段：**P0-L1 Observability Privileged Label Smoke = pass**
- 冻结特权标签契约；非训练数据集；写盘仍关；不训模型；不重开 C0/C1/C1.1
- 产物：`docs/PRIVILEGED_LABEL_SCHEMA_V1.md`、`docs/OBSERVABILITY_PRIVILEGED_LABEL_SMOKE.md`、`data/manifests/observability_privileged_label_smoke_v1.json`
- 下一步：可讨论 Observability **数据设计**条件；仍不直接训练

## 2026-08-14：P0-L1 Privileged Label Smoke

- episodes 0/2/4 × window 8；bit-exact / restore / timeline 全过
- 纳入：o2h pose、冻结有限差分 velocity、peg-hand contact、指力 active、outcome_raw、provenance
- 排除：slip truth、细 contact-mode、风险标签

## 2026-08-14：P0-L0 Label Derivability Audit

- o2h pose：**derivable**；其余多为 **partial**（slip/细 mode 无真值契约）
- restore 一致性：ep0 frame 287 all_ok
- 禁止：采集、训练、写盘、重开旧 Controllability

## 2026-08-14：P0 验收清单（不写盘）

- 文档：`docs/P0_ACCEPTANCE_CHECKLIST.md`
- 内容：Obs/Ctrl/Sem 的输入字段、标签、held-out、通过/停止条件与当前缺口
- 明确：S0.* plumbing ≠ Semantic；C0–C1.1 ≠ Controllability 通过；pilot ≠ 算法数据
- 交接修正：`next_action`→标签可派生性；pilot 测试计数→40

## 2026-08-14：mock 写入器 hardening

- manifest 失败：回滚已发布 traj + `incomplete` run 记录
- scaffold：`trajectories/`/`manifests/`/`.tmp`/banner 拒 symlink
- schema：`horizon_steps_used <= horizon_budget_max`；manifest 约束加强
- 单测：失败注入、symlink、`..` 路径攻击；仍仅 `/tmp` mock

## 2026-08-14：单条 trajectory 写入设计（mock）

- 设计：`docs/MICRO_DEMO_PILOT_WRITE_DESIGN.md`
- schema：`pilot/traj_schema.py`
- 原子写入：`pilot/atomic_write.py`（生产入口拒绝；`commit_trajectory_mock` 仅 `/tmp`）
- 单测：`tests/test_pilot_atomic_write.py`

## 2026-08-14：Micro-demo pilot dry-run 安全加固

- schema/gates/caps 强制消费；关 gate / 负 horizon / 未知字段 → 环境前 aborted
- symlink `lstat` + 禁训 resolved 检查；`/tmp` 报告 O_EXCL|O_NOFOLLOW
- 单测 22 通过；集成三门 + horizon_steps_used=80
- 产物：`pilot/config_schema.py`、`tests/test_pilot_dry_run_guards.py`

## 2026-08-14：P0-S0.4c hardened（采集前回归）

- 4/4 pass；命名：多族 oracle 建立接触后的物理抓取配方 smoke
- establish 可 snap；settle 后抓 MjData；open/closed 同态恢复（非再 establish）
- 硬门槛：`snap_call_count_after_establish == 0`（四族）
- transport 横移门：含 `round_8mm` demo 路径（hand≈9.5cm / peg≈9.8cm）与三 formal 族
- 产物：同 `docs/GRASP_STABILITY_MULTIFAMILY_PHYSICAL_SMOKE.md`（protocol=P0-S0.4c-hardened）

## 2026-08-14：P0-S0.4c 多族物理抓取

- `round/rectangular × 8/16mm` → **4/4 pass**（首轮；后由 hardened 收紧）
- `round_8mm`：demo transport root（S0.4b）
- 其他族：抬高 + oracle establish（demo 相对 in-hand）后纯动力学
- hold/lift/transport + 开手负对照；闭手优于开手
- 产物：`docs/GRASP_STABILITY_MULTIFAMILY_PHYSICAL_SMOKE.md`

## 2026-08-14：P0-S0.3b 同族多 socket instance

- `round_8mm` / `rectangular_8mm` 双 socket arena → **2/2 pass**
- 区分：`target_instance_id`、`socket_site`、在场 pose；`claim_matches_env` 同 family 仍可判错实例
- wrong claim 经 `semantic_target_features(claimed_target=...)` 读真实次 socket site
- 产物：`docs/TARGET_HOLE_MULTI_INSTANCE_SMOKE.md`

## 2026-08-14：P0-S0.4b 物理抓取门（用户确认）

- 3 demo transport roots（ep0/2/4）→ **3/3 pass**
- 证明：已有真实接触可在短时纯动力学 hold/lift/transport 保持；闭手优于开手
- 边界：仅 `round_8mm` demo roots；非学会策略；非多族脚本抓取
- 产物：`docs/GRASP_STABILITY_PHYSICAL_SMOKE.md`

## 2026-08-14：P0-S0.4 命名收紧

- 准确结论：抓取稳定性**指标与阶段编排** smoke = pass
- 非结论：物理抓取稳定性门（由 S0.4b 单独判定）
- 原因：正对照每步 `snap_peg_to_palm`；`contacts=0`；负对照关 snap 后自由落体
- `claims_physical_grasp_stability=false`（对本门）；`claims_stable_grasp_policy=false`

## 2026-08-14：P0-S0.4 抓取稳定性（instrumentation）

- 4 代表族：`round/rectangular × 8/16mm` → instrumentation **4/4 pass**
- 相位：lift / hold / transport + open-hand 负对照（夹具路径）
- 正对照：oracle 运动学 palm-snap（非物理/学习抓取）
- 产物：`docs/GRASP_STABILITY_SMOKE.md`、`data/manifests/grasp_stability_smoke_v1.json`

## 2026-08-14：P0-S0.3 目标孔语义

- 准确命名：**单目标孔语义 metadata/plumbing pass**（非机器人目标孔认知）
- `geometry/target_hole.py` + `scripts/run_target_hole_semantics_smoke.py`
- reset info：`geometry_family_id` / `target_instance_id` / socket·peg / tip_to_target
- `claim_matches_env` 比 family+instance+site；wrong-target 经 `claimed_target`/`claimed_socket_pose_xyz`
- labeler / `_insert_geometry` / `full_obs` / `grasp_metrics` / `full_episode_utils`：family-aware
- 产物：`docs/TARGET_HOLE_SEMANTICS_SMOKE.md`

## 2026-08-14：P0-S0.2 正式 arena

- 命名：`dexjoco/sim/envs/assembly_geometry.py`
- 生成器：`embodied_grasp_insertion/geometry/formal_xml_builder.py`
- env：`PandaBimanualAssemblyGymEnv(geometry_family=...)`，默认 `round_8mm`
- smoke：`scripts/run_formal_arena_reset_smoke.py` → 8/8 pass（含末尾低速门）
- 产物：`docs/FORMAL_ARENA_RESET_SMOKE.md`、`data/manifests/formal_arena_reset_smoke_v1.json`
- 未证明：全深度插入、抓稳、策略知孔

## 2026-08-14：P0-S0.1c

- 脚本：`scripts/run_geometry_env_settle_smoke_v1c.py`
- 产物：`data/manifests/geometry_env_settle_smoke_v1c.json`、`docs/GEOMETRY_ENV_SETTLE_SMOKE_V1C.md`
- 要点：tip=碰撞下端；base 顶低于 site；wall/bottom/base/floor 分统；peg+socket 冻结仅 `mj_forward`；pen tol≪间隙；mild 进通过条件；mismatch 同深度 3/3

## 2026-08-13：P0-C0 Finger Controllability Matched Smoke

### 命令

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/wangrenpeng/dexjoco \
  conda run -n dexjoco --no-capture-output \
  python scripts/smoke_full_snapshot_restore.py --episode-id 0 --horizon 8 --strict \
  --output outputs/snapshot_restore_smoke_ep0.json

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= PYTHONPATH=/home/wangrenpeng/dexjoco \
  conda run -n dexjoco --no-capture-output \
  python scripts/run_finger_controllability_smoke.py \
  --config configs/finger_controllability_smoke.yaml --skip-determinism
```

### 结果摘要

- snapshot restore：2 roots 逐步 bit-exact 通过
- 3 episodes / 7 roots / 35 branches；fairness 35/35
- wrist=`hold`；finger={hold, demo_replay, mild_close, mild_open, random}
- 手指动作有因果效应，但相对 hold 多为增大 drift / 掉杆（尤其 mild_open）
- 结论 **harmful_only**；不是 Controllability passed

### 产物

- `simulation/full_episode_snapshot.py`
- `physics/grasp_metrics.py`
- `docs/FINGER_CONTROLLABILITY_SMOKE.md`
- `data/manifests/finger_controllability_smoke_v1.json`
- `outputs/finger_controllability_smoke_v1/`

### 下一步

- 不进入扩展 Controllability P0，除非换干预设计（不稳定 root、幅度日程、wrist transport matched）并再现 stabilizing 证据
- 仍不开始 Observability / Semantic / policy

## 2026-08-13：P0-A System Identifiability Audit

### 审计命令

```bash
# smoke
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= conda run -n dexjoco --no-capture-output \
  python scripts/audit_system_identifiability.py --max-episodes 2 --metadata-only --strict

# full metadata + 单 episode MuJoCo smoke
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= conda run -n dexjoco --no-capture-output \
  python scripts/audit_system_identifiability.py --metadata-only --strict --mujoco-smoke-episode 0
```

### 数据路径

- sidecar：`/mnt/hdd/dexjoco/interaction_sidecar/bimanual_assembly`
- manifest：同上 `/manifest.json`（100 episodes）
- zarr 根：`/mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly`
- 输出：`data/manifests/system_identifiability_audit_v1.json`、`docs/SYSTEM_IDENTIFIABILITY_AUDIT.md`、`outputs/state.json`

### 主要统计

- 审计有效 episode：**100**；排除：**0**
- object asset：**1**（`industreal_round_peg_8mm`）
- socket/tray asset：**1**（`industreal_tray_insert_round_peg_8mm`）
- geometry family：**1**（同一几何的不同轨迹/位姿，不是多物体多孔）
- 85D/44D 与 `full_obs.py` / `full_env.py`：**一致**
- MuJoCo smoke ep0：obs=85、act=44、FingerForceLabeler 指力 shape 可用

### 无法确认 / 缺失字段

- 85D 内无逐指接触、无 object-in-hand 6D、无 slip
- sidecar 无逐步接触/滑移时序标签
- `AssemblyOutcome` 无 capture/rim/jam/backout
- `FullEpisodeEnv` 无 snapshot/restore API
- object / geometry held-out split：**不可行**
- 仓库另有 4/12/16mm 等 mesh，但未进入本 100 ep 覆盖（不得按 episode ID 猜测）

### P0-A 判定

- **部分通过（partial）**
- 原因：snapshot（insert 路径）可恢复手/物/接触物理态，episode split 可行，schema 一致；但几何多样性停止条件触发，部署抓持观测不足。

### 下一步

1. 为 `FullEpisodeEnv` 增加 snapshot/restore（本项目内包装，不改 reach 实验逻辑亦可先在本仓做适配层）后做 Controllability P0 matched smoke；
2. 不启动完整 Observability P0；
3. Semantic / Observability 硬门需要多几何 episode 覆盖后再开。

## 2026-08-13：新项目建立

### 触发原因

用户指出此前连续建立 recovery policy 项目，但系统的真正痛点可能不在 policy family：

- 灵巧手抓持物体可能不稳定；
- 手内滑移和接触变化会直接改变插入几何；
- 插孔失败可能源于抓持、对孔、capture、jam 或 backout，而非单一腕轨迹误差；
- 当前模型不知道手里是什么、孔是什么以及两者如何配合，只是在固定任务上模仿 demo 的大致动作。

该判断经代码和数据审计后成立。

### 当前接口证据

- `reach_insert_rl/env/obs.py` 的 insert observation 为 37D：相对几何、孔轴/peg 轴、双腕位姿、双腕 wrench；
- 37D observation 不含手指关节、各指接触、object identity/shape、object-in-hand pose 或显式 slip；
- `InsertHandoffEnv` 的 12D action 只控制双腕，代码明确冻结全部手指；
- offline-search NPZ 仅有 37D observation、12D wrist action、tip distance 与 lateral error；
- recovery 元信息没有物体/孔几何变化，不能支持 object/hole held-out 语义结论；
- 仓库已有 full-episode 85D observation 与 44D wrist+finger action 接口；P0-A 已审计其字段与局限。

### 复习的已证伪族

1. **未来状态轨迹条件动作解码**：Final 24 folds 相对 Direct 为负；本项目不再输入未来轨迹。
2. **Diffusion/Flow recovery**：P0 不训练生成策略。
3. **Set-Listwise / 候选选择**：不做在线候选评分。
4. **Online Physics Branch MPPI**：`5/15` 低于 PrivHI `6/15`。
5. **wrist-only Contact-CMDP**：已降级。
6. **gate / residual / servo**：禁止。

### 已完成

- [x] 新建独立项目目录 `embodied_grasp_insertion/`；
- [x] 创建 `docs/MOTIVATION_AND_PLAN.md`；
- [x] 创建 `docs/PROGRESS.md`；
- [x] 固化三项前置硬门与策略禁入规则；
- [x] P0-A 只读审计脚本、manifest、报告、`outputs/state.json`；
- [x] 项目技能 `.cursor/skills/embodied-grasp-insertion-audit/SKILL.md`。

### 未开始

- [ ] FullEpisodeEnv snapshot 适配 / Controllability P0 smoke；
- [ ] 多几何定向采集（仅在硬门需要时）；
- [ ] Observability / Semantic P0；
- [ ] 任何 policy 训练。

## 当前硬门结论

- Data / Identifiability：**partial**；
- Observability hard gate：**未评测 / 暂不可完整开始**；
- Controllability hard gate：**未评测 / 允许准备 smoke**；
- Semantic hard gate：**未评测 / 单几何阻塞**；
- Policy training：**禁止**；
- `busy=false`。

## P0-C1 Calibrated Finger Intervention（2026-08-13）

- verdict: **no_effect**
- allow_extended_controllability_p0: false
- Observability / Semantic / policy: 仍禁止
- 校准: 16/16 flexion pass
- roots: 6 eps 扫描 → 8 unstable → smoke 8 roots × 2 contexts × 6 interventions
- 成对: close 多增大 drift；open_low 局部改善但仅 2 episodes，未达 promising
- 详见 `docs/FINGER_CONTROLLABILITY_CALIBRATED_SMOKE.md`

## P0-C1.1 Root/Load Validation（2026-08-13）

- verdict: **load_calibration_fail / screening_fail**
- constant mild_transport dose 未过门（最多约 2 unstable，median Δdrift≈0.29mm）
- multi-profile calibration：**已中止**
- `selection_ok=false`；不得使用 fallback delta
- **正式停止 P0-C1.1**：不再调 transport load / wrist pulse，不跑 unstable root screening，不跑 finger smoke，不训练
- 停止原因：无法构造足够的、可恢复的真实不稳定 root；继续调 load 易无限调参
- 详见 `docs/TRANSPORT_LOAD_CALIBRATION.md`

## 阶段汇总（硬门）

| 阶段 | 结论 | 含义 |
|---|---|---|
| P0-A | partial | 单几何 + 缺部署抓持字段 |
| P0-C0 | harmful_only（scoped） | 稳定 demo roots + 24 步固定增量下未优于 hold |
| P0-C1 | no_effect | 校准后低剂量干预未达 promising |
| P0-C1.1 | load_calibration_fail | 无法筛出合格 unstable/stable 分化；已停止 |
| P0-S0 | pass | 多几何临时 plumbing；≠ Semantic P0；仍禁采集/训练 |
| P0-S0.1 | numerical_only（作废物理 settle） | 阈值过松；弹飞/初接触未检 |
| P0-S0.1b | pass | support+clearance 严格 8/8；仍禁采集/训练 |

- allow_*：**全 false**
- phase：`semantic_geometry_infrastructure`
- 下一步可选：参数化 arena 设计（未做全量采集）
- 详见 `docs/GEOMETRY_ASSET_AUDIT.md`、`docs/GEOMETRY_COMPILE_SMOKE.md`、`docs/GEOMETRY_ENV_SETTLE_SMOKE.md`
- 不删除任何 P0-C0 / P0-C1 / P0-C1.1 产物

## P0-S0 Multi-Geometry Simulator Plumbing（2026-08-14）

- verdict: **pass**
- 8/8 compile+lookup；round+rect 全覆盖
- raw mesh≈yaml 米制；XML scale=4.5；collision=measured×scale（未抄 8mm）
- 8 family 有可插入区间；3 mismatch 负对照有效
- 临时 XML 在 `outputs/geometry_xml_tmp/`；未改正式 arena
- 训练/全量采集/Semantic P0：仍禁止

## P0-S0.1 Env Settle Smoke（2026-08-14）

- 原结论 pass **已作废（作为物理 settle）**
- 更正：仅为 **numerical execution pass**（不 NaN、不超速阈值）；11/16 tip XY>5cm，socket 弹飞至 0.26–0.65m
- 详见作废说明：`docs/GEOMETRY_ENV_SETTLE_SMOKE.md`

## P0-S0.1b Strict Support + Clearance（2026-08-14）

- verdict: **pass**（support 8/8 + clearance 8/8 + mismatch 3）
- 拆分：`support_settle`（分开放置、低速窗口）与 `insertion_frame_clearance`（固定 socket、运动学放置）
- tip 改为 collision 插入端；要求 8/8，不允许 6/8 凑 pass
- 仍禁训练与全量采集；`allow_formal_arena_edit=true`（可选，尚未改正式 arena）
- 详见 `docs/GEOMETRY_ENV_SETTLE_SMOKE_V1B.md`
