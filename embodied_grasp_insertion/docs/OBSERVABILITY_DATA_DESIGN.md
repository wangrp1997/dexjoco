# Observability 数据设计（P0-Obs-Design，只设计不落盘）

- 日期：2026-08-14
- 状态：**设计评审通过**（2026-08-14）
- 前置：P0-L0 pass；P0-L1 `overall_verdict=pass`（特权标签契约已冻结）
- **本文件不生成训练数据集、不训练模型、不打开写盘、不采集新 episode**
- 目的：定义部署输入、标签窗口、split、有效样本与消融；下一步为 `P0-Obs-D0` 只读可行性盘点
- 明确：**单几何 sidecar 覆盖 ≠ Observability P0 通过**

## 0. 边界

| 是 | 否 |
|---|---|
| 部署输入档 A–D 的字段契约 | 把 privilege tip/axis/cfrc 当部署输入结论 |
| 使用 P0-L1 特权标签作教师 | 生成 slip truth / 细 contact-mode（无独立契约前） |
| episode/root held-out 规则 | geometry held-out 冒充已通过（单几何 sidecar 仍不够） |
| 输入消融表 | 直接训 Obs 模型或写 dataloader |

完整 Obs 硬门通过仍须见 `docs/P0_ACCEPTANCE_CHECKLIST.md` §1；本设计只回答「数据长什么样」。

---

## 1. 部署输入（学生侧）

历史长度集合：`H ∈ {1, 4, 8, 16}`（至少报告全部；默认主表用 H=8）。

| 档 | 名称 | 字段 | 来源 | 禁止混入 |
|---|---|---|---|---|
| A | proprio | 双臂+手指指令/关节历史，对齐 `act44` 语义 | FullEpisodeEnv / 85D 的 act44 段 | peg7、tray7、tip/axis、flags |
| B | + wrist FT | A + `ft12` 腕力历史 | 85D ft12 | 同上 |
| C | + finger contact proxy | B + **部署口径**逐指接触/触觉 | 真机 tactile 或明确标注的仿真 tactile；**评测结论不得把 MuJoCo `cfrc_ext` 写成部署已解决** | 特权指力当真机传感 |
| D | + vision/geometry | C + 视觉或几何描述子 | 相机/编码；几何 descriptor 不得直接喂 privilege tip/axis | 固定 tip/lateral/axis 当语义 |

v0 评测优先：**A / B / B+privilege-finger-ablation(标注为非部署)**。  
档 C 若仅有仿真 cfrc，只能作为 **oracle sensing ceiling**，不得写「部署 Obs 通过」。

---

## 2. 标签窗口（教师侧，对齐 P0-L1）

每样本 = 连续控制帧窗口 `W`（设计默认 **W=8**，与 L1 smoke 一致）。

### 2.1 必选标签（schema `privileged_label_v1`）

- `object_in_hand_pose_6d`（palm 系）
- `object_in_hand_velocity`（有限差分契约；窗内第 1 帧 `available=false`）
- `peg_hand_contact` total / by_finger
- `finger_force` norm / contact_active（特权；监督用）
- `outcome_raw`：`tray_ok` / `peg_ok` / `insert_ok`
- `provenance`：episode / frame / root_id / family / instance / socket_site / dt

### 2.2 明确不导出（直至独立契约）

- `slip` / `slip_truth`（仅允许未来 schema 中的 `slip_proxy`）
- capture / rim / jam / partial / seated / backout
- regrasp-needed / peg-loss risk 正式概率标签

### 2.3 监督任务（设计态）

| 任务 | 目标 | 备注 |
|---|---|---|
| T1 | 回归 o2h pose（窗末或逐步） | 主任务 |
| T2 | 回归 o2h velocity（available 帧） | 依赖契约 dt |
| T3 | 预测 contact_active / by_finger counts | 分类或回归计数 |
| T4 | 预测 outcome_raw（可选辅任务） | 不得替代 T1 |

---

## 3. Split

### 3.1 必需

| Split | 规则 |
|---|---|
| episode-held-out | 整 episode ∈ train 或 val/test，禁止跨集合 |
| root-held-out | 同一 `root_id` 及同 snapshot 派生窗口不得跨 split |

### 3.2 推荐（完整 Obs 硬门）

| Split | 规则 |
|---|---|
| object/geometry-held-out | 需多族数据；当前 sidecar 单几何 → **不可宣称 Obs 硬门通过** |

### 3.3 泄漏禁令

- 同 episode 的相邻重叠窗口若 root 相同，必须同 split  
- 禁止按帧随机切分导致同 root 泄漏  
- 禁止用 tip 距离阈值当 geometry held-out

### 3.4 建议比例（设计，不采）

在现有 100 ep 单几何上：train/val/test ≈ 70/15/15 **按 episode**；  
每个 ep 取 1–3 个 transport/early_grasp root 窗口，总窗口数作报告，不作「多样本伪独立」。

---

## 4. 有效样本规则

样本有效当且仅当：

1. 窗口长度 = W，帧号连续：`f, f+1, …, f+W-1`  
2. `sim_time` 单调增加；`control_dt_s` 与契约一致  
3. 特权标签通过 `validate_privileged_label`  
4. 窗内第 0 帧 velocity.available=false；其后 true  
5. restore 后再导出标签 bit-exact（导出流水线门禁；非每样本在线）  
6. `geometry_family_id` / `target_instance_id` / `root_id` 非空  
7. 不落在正式 `pilot_micro_demo_v0`  

无效 / 丢弃：

- episode 在窗内 early terminate  
- peg 已 `insert_ok` 且任务定义为 grasp-transport 观测（可选过滤，须预注册）  
- contact 全 0 且任务要求 grasp 接触（可选；预注册）

---

## 5. 输入消融（必须报告的对照）

| ID | 输入 | 目的 |
|---|---|---|
| A0 | proprio H | 下限 |
| B0 | proprio + wrist FT | 腕力增益 |
| B_shufFT | B0 + 打乱 FT 时间对齐 | 防泄漏/伪相关 |
| P_finger | B0 + **特权**指力（标 privilege） | sensing ceiling，非部署结论 |
| P_o2h | 直接喂 o2h（作弊上限） | 标签可学性天花板 |
| G_shuf | 若含 geometry：打乱配对 | 语义泄漏对照（多几何后） |

判定叙述：

- 仅当 **部署档（A/B，及将来真 tactile C）** 在 episode/root-held-out 上稳定接近合理上限，才谈 Obs 进展  
- `P_finger` / `P_o2h` 优而部署档差 → **sensing gap**，停训控制策略

---

## 6. 与现有产物的关系

| 产物 | 角色 |
|---|---|
| P0-L1 schema / smoke | 教师标签契约与可重复性 |
| sidecar 100 ep | 候选源；单几何 |
| formal multi-family arena | 未来 geometry held-out 前提；**尚未**进 Obs 导出 |
| micro-demo pilot | 无关；写盘仍关 |
| P0-Obs-D0 | 只读可行性盘点（统计 + 最多 /tmp 3-ep 样例） |

---

## 7. 就绪检查

- [x] 特权标签契约冻结（L1）  
- [x] `/tmp` 审计输出路径兼容（path helper）  
- [x] 本设计评审通过  
- [ ] P0-Obs-D0 可行性盘点完成  
- [ ] 明确授权「完整只读 Obs 评测包导出」（仍非训练）  
- [ ] 仍禁止：训练、全量采集、`WRITE_IMPLEMENTATION_ENABLED=True`、重开 C0/C1/C1.1  

当前：**设计通过 → 执行 D0**；未授权完整评测包导出 / 训练。

---

## 8. 相关文件

- `docs/P0_ACCEPTANCE_CHECKLIST.md`
- `docs/PRIVILEGED_LABEL_SCHEMA_V1.md`
- `docs/OBSERVABILITY_PRIVILEGED_LABEL_SMOKE.md`
- `docs/OBSERVABILITY_DATASET_FEASIBILITY.md`（D0 报告）
- `labels/privileged_schema.py`
- `docs/SEMANTIC_DATA_DESIGN.md`（几何 held-out 另线）

*版本：v0-design-approved · 2026-08-14 · no dataset / no train / no write*
