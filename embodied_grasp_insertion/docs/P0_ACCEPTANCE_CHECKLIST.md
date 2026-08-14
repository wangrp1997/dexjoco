# P0 正式验收清单：Observability / Controllability / Semantic

- 日期：2026-08-14
- 状态：**设计文档**（不采集、不训练、不改写盘开关）
- `WRITE_IMPLEMENTATION_ENABLED=False`；正式 `data/pilot_micro_demo_v0/` 不创建
- 依据：`MOTIVATION_AND_PLAN.md` §9–12、`SYSTEM_IDENTIFIABILITY_AUDIT.md`、`SEMANTIC_DATA_DESIGN.md`、P0-C0/C1/C1.1 与 P0-S0.* 已有结论

## 0. 总则

### 0.1 策略准入（三项全过才允许）

| 门 | 证明什么 | 当前 |
|---|---|---|
| Observability P0 | 部署可用历史能估计抓持/接触 belief | **未开评测**（缺标签与部署字段） |
| Controllability P0 | 全手 44D 能因果改善 retention / 物理后果 | **未通过**（C0/C1/C1.1 停在失败/无效应） |
| Semantic P0 | 对象—孔几何条件在 held-out geometry 有效 | **未开评测**（plumbing≠语义） |

未过任一门：禁止 policy / critic / actor / Diffusion / Flow / 大 Transformer 训练。

### 0.2 明确不等于这些门的工作

| 已做 | 是什么 | 不是什么 |
|---|---|---|
| P0-A | 可解性审计 `partial` | Observability 通过 |
| P0-C0/C1/C1.1 | 手指干预 smoke；已停止调参 | Controllability 通过 |
| P0-S0–S0.4c | 多几何 arena / 目标 metadata / 物理抓取配方 | Semantic / Observability / Controllability 通过 |
| micro-demo pilot | dry-run + mock 写盘链路 | 算法数据；`insert_phase=skipped` |

### 0.3 本清单禁止

- 采集、训练、打开 `WRITE_IMPLEMENTATION_ENABLED`
- 把 privilege（peg7/tray7、tip/axis、flags、cfrc）写成部署观测
- 把固定 tip/lateral/axis 误差称为通用插孔语义
- 用 tip 变近替代 retention / `insert_ok` / contact-mode
- 用 episode 随机切分冒充 geometry held-out
- 用 object ID 记忆冒充几何语义

---

## 1. Observability P0（可观测天花板）

### 1.1 问题

部署可用历史能否预测特权真值中的抓持与接触状态？（能预测特权、不能预测部署 → 先修 sensing，不训控制）

### 1.2 输入字段（部署侧，按对照递增）

| 档 | 字段 | 来源约束 |
|---|---|---|
| A | 双臂+手指 proprioception 历史（关节/指令） | `act44` 及历史；**不含** peg7/tray7 |
| B | A + 双腕 wrench 历史 | `ft12` 历史 |
| C | B + 逐指接触/触觉历史 | 真机或仿真 tactile；**不得**把 MuJoCo `cfrc_ext` 当部署输入写进结论 |
| D | C + vision / geometry descriptor | 视觉或几何编码；**不得**直接喂 privilege tip/axis |

禁止作为部署输入结论：`peg7`、`tray7`、`lat_vec3`、`along_tip_axis`、`hole_axis3`、`peg_axis3`、`flags3`、特权 o2h 真值。

历史窗：至少报告 H1 / H4 / H8 / H16。

### 1.3 标签（特权教师，仅监督/评测）

| 标签 | 要求 |
|---|---|
| `object_in_hand_pose_6d` | peg 相对抓持手/腕的相对位姿序列 |
| `object_in_hand_vel` | 相对速度（可选但推荐） |
| `per_finger_contact_retention` | 每指 binary 或力阈值；定义固定后不得事后改口径 |
| `slip_proxy_or_truth` | 显式定义（相对位移阈值 / 接触切向等）；禁止无定义伪造 |
| `contact_mode` | 至少：free / grasp / capture / rim / jam / partial / seated / backout（可子集起步，须写清） |
| `regrasp_needed` / `peg_loss_risk` | 短期风险标签，与 retention 一致 |
| `object_target_mating_T` | object insertion feature → target mating feature 变换 |

每步还须绑定：`episode_id`、`root_id`、`geometry_family_id`、`target_instance_id`、`seed`、snapshot 引用。

### 1.4 Held-out split

| Split | 规则 | Observability 是否必需 |
|---|---|---|
| episode-held-out | 整 episode 不跨 train/val | 必需 |
| root-held-out | 同 snapshot 派生分支不得跨 split | 必需 |
| object-instance-held-out | 未见过的 peg 尺寸/截面 | 硬门推荐；单几何时**不可宣称门通过** |
| geometry-family-held-out | 整族 object+socket 未见 | Semantic 硬门必需；Obs 完整通过也需要 |

泄漏禁令：同一 snapshot、近重复 root、同派生分支不得跨 split。

### 1.5 任务与对照

任务：o2h 回归；finger retention/slip；contact-mode 分类；regrasp/peg-loss 风险；mating transform。

对照（均须报告）：

1. proprio only  
2. proprio + wrist FT  
3. + finger contact/tactile  
4. + vision/geometry  
5. shuffled tactile / shuffled geometry / mismatch pairing  

### 1.6 通过条件

1. episode/root-held-out 上，含接触或几何的对照**稳定优于**无触觉、无几何与 shuffle。  
2. 指标按 episode（及有多对象时按 object）等权，不被单 episode 支配。  
3. 不确定性与误差相关（不得只报过度自信点估计）。  
4. 若特权可预测而最佳部署档不可预测 → **失败**，标记 sensing gap，不进入控制训练。

### 1.7 停止条件

- 缺少 o2h / finger retention / contact-mode 标签且无法从 snapshot 合法派生。  
- 仅单几何且无法构造 object/geometry held-out，却宣称 Observability 硬门通过。  
- 部署结论混入 privilege tip/axis。  
- 仅靠更大网络/更多 epoch 在失败部署输入上“抢救”。

### 1.8 当前缺口（相对清单）

- 85D 无部署逐指接触、无 o2h 字段、无 slip、无细 contact-mode。  
- sidecar 原 100 ep 单几何（P0-A）；正式多族 arena 已有 plumbing，**尚无** Observability 评测集。  
- → **不可完整开始正式 Obs 评测**，直至标签契约与至少 episode/root split 数据就绪。

---

## 2. Controllability P0（全手因果可控）

### 2.1 问题

在公平 matched snapshot 下，手指动作是否能改变抓持/插入物理后果（非仅 tip）？

### 2.2 输入 / 动作接口

| 项 | 规定 |
|---|---|
| 状态根 | 同一 `MjData`(+ wrapper) snapshot；bit-exact restore 已验证才可扩规模 |
| 腕动作预算 | wrist-only 与 full-44D **相同**腕轨迹/时长 |
| 手指动作 | 44D 中 finger16；须先有 flexion 符号校准（见 P0-C1） |
| 对照分支 | `hold_finger` / `demo_finger_replay` / `calibrated_close|open` / `random_finger` / `wrist_only` |

### 2.3 标签 / 指标（主次分明）

| 优先级 | 指标 | 说明 |
|---|---|---|
| 主 | `peg_retained` / contact retention | 相对 **root 时刻** 接触，非“第一步后”重定义 |
| 主 | `object_in_hand_drift` | 手内相对位姿漂移 |
| 主 | `insert_ok`（若进入插入段） | 不得用 tip 替代 |
| 次 | jam / overload / slip_proxy | 诊断用 |
| 禁作主结论 | tip distance 单独改善 | 可记录，不可作通过理由 |

### 2.4 Held-out / 覆盖

| 要求 | 最小 |
|---|---|
| episodes / seeds | ≥3 |
| roots | 不稳定但未掉杆的 matched roots；须有筛选协议且 `selection_ok=true` |
| contexts | ≥2（如 hold vs mild transport）；transport dose 须先验校准 |
| fairness | 腕预算、时长、restore 一致性 = 1.0 |

禁止：稳定 demo root 上天花板效应却宣称“手指无用”；无限调 load/pulse。

### 2.5 通过条件

1. 存在**可重复、跨 root、跨 episode** 的手指因果效应。  
2. 效应表现在 retention / drift / `insert_ok`，**不是**仅 tip。  
3. full-44D 在公平预算下优于或异于 wrist-only / hold（方向取决于干预假设，但须预注册）。  
4. 效应在校准 flexion 与合理剂量下可复现（非单次噪声）。

### 2.6 停止条件

- 44D 无法稳定改变抓持结果。  
- 收益只来自不公平腕动作/时长/restore 差异。  
- 无法构造足够可恢复 unstable roots（→ 修筛选/接口/接触模型，**不**无限调参）。  
- 已触发：P0-C1.1 `load_calibration_fail`（正式停止调 load）。

### 2.7 当前状态（相对清单）

| 子阶段 | 结论 | 对 Controllability 门 |
|---|---|---|
| snapshot restore | bit-exact pass（有限 ep） | 基础设施可用 |
| P0-C0 | scoped `harmful_only` | 未通过 |
| P0-C1 | `no_effect` | 未通过 |
| P0-C1.1 | `load_calibration_fail` | **停止该路线调参** |

→ Controllability **未通过**。重开须新干预设计（新 root 定义或接口修复），不得续调旧 load。

---

## 3. Semantic P0（对象—孔几何语义）

### 3.1 问题

模型是否学习配合关系，而非固定资产的 tip 坐标反馈？

### 3.2 输入字段

| 允许 | 禁止当作“语义” |
|---|---|
| object/target geometry descriptor（截面、尺度、clearance、depth、symmetry） | 仅资产 ID / episode ID |
| canonical frames + mating features | 单一固定 peg/socket 的 tip/lateral/axis 当通用语义 |
| 正确配对与 mismatch 负对照 | round×rect 错配当正样本 |
| privilege mating 真值（教师） | 把 privilege tip 误差写成部署已学会语义 |

### 3.3 标签

| 标签 | 用途 |
|---|---|
| `geometry_family_id` / section / nominal_mm | 族身份（评测轴，非唯一特征） |
| `target_instance_id` / `socket_site` | 多实例孔 |
| `mating_feasibility` | 当前相对位姿可否 capture/insert |
| `legal_mating_set` / 对称性约束 | round≈SO(2)；rect 需 yaw |
| `contact_mode_transition` | 动作后模式转移 |
| `insert_ok` / insert depth | 物理后果 |
| `claim_matches_env` | 错误目标 claim 负对照（plumbing 已有；语义评测需策略侧） |

### 3.4 Held-out split（冻结后再评测/采集）

设计（见 `SEMANTIC_DATA_DESIGN.md`；正式族名以 `configs/geometry_families.yaml` 为准）：

| 角色 | 建议 family |
|---|---|
| Train（≥3） | 如 `round_8mm`、`round_12mm`、`rectangular_8mm` |
| Held-out object/geometry（≥2） | 如 `round_16mm`、`rectangular_12mm` 或 `rectangular_4mm` |

规则：

- **geometry-family-held-out**：整族未进训练；评 insert_ok / mode / 对准分布。  
- **object-held-out**：未见 peg 尺寸/截面。  
- 禁止仅 episode-id 随机切分冒充几何 held-out。  
- 禁止同截面错配 socket。

规模（设计态，**本轮不采集**）：smoke ≥10 ep/family；正式 Semantic 量级另审。

### 3.5 任务与对照

任务：合法 mating；可否 capture/insert；mode transition；held-out family 上的后果预测。

对照：无 geometry / 仅 ID / shuffled-pair / mismatch pairing。

### 3.6 通过条件

1. geometry-conditioned 表示在 held-out family 上**稳定优于**无几何、仅 ID、shuffled-pair。  
2. mismatch 负对照显著变差（证明非盲目 tip 伺服记忆）。  
3. 同族多 instance（若启用）能区分 `target_instance_id`，而非只认 family 名。

### 3.7 停止条件

- 训练族少到无法形成 held-out geometry。  
- 仅靠资产 ID / episode 泄漏过关。  
- 把 P0-S0.* plumbing 或 tip 误差下降写成 Semantic 通过。

### 3.8 当前状态（相对清单）

| 项 | 状态 |
|---|---|
| 多族 mesh/XML/arena plumbing | P0-S0–S0.2 pass |
| 目标孔 metadata / 多 instance | S0.3 / S0.3b pass；`claims_policy_knows_hole=false` |
| 物理抓取配方 | S0.4c-hardened 4/4；`claims_stable_grasp_policy=false` |
| geometry held-out 评测集 + 表示学习 | **未做** |
| Semantic P0 | **未通过 / 未开评测** |

---

## 4. 三门依赖与推荐顺序

```text
P0-A (partial) ──► 标签契约 + snapshot
        │
        ├─► Controllability：公平干预（已试 C0–C1.1，失败/停止）
        │         重开条件：新 root/接口假设，禁止续调旧 load
        │
        ├─► Observability：部署输入 vs 特权标签（缺字段则先补标签协议）
        │
        └─► Semantic：多族 held-out（plumbing 已备；需数据与评测，≠ S0.*）
```

算法准入：Obs ∧ Ctrl ∧ Sem 全通过，且 snapshot/split/标签真实性审计通过。

真实写 1 条 pilot traj：对三门**几乎无增益**（`insert_phase=skipped`）；仅工程验证；**暂不授权**。

---

## 5. 算法准入前检查表（勾选）

- [ ] Observability：部署档对照 + held-out 通过 §1.6  
- [ ] Controllability：跨 root 因果效应通过 §2.5（非 C0/C1/C1.1 旧结论）  
- [ ] Semantic：geometry held-out 通过 §3.6  
- [ ] 无 privilege 伪装部署；无 tip 替代主指标；无 ID 泄漏  
- [ ] `WRITE_IMPLEMENTATION_ENABLED` 与训练仍按单独授权  

当前：**全部未勾选**。下一步做清单落地设计/标签契约，不采集、不训练、不写盘。

---

## 6. 相关文件

| 文件 | 作用 |
|---|---|
| `docs/MOTIVATION_AND_PLAN.md` | 原假设与 P0-B/C/D |
| `docs/SYSTEM_IDENTIFIABILITY_AUDIT.md` | P0-A |
| `docs/SEMANTIC_DATA_DESIGN.md` | 多几何设计 |
| `docs/FINGER_CONTROLLABILITY_*.md` / `TRANSPORT_LOAD_CALIBRATION.md` | Ctrl 已试路径 |
| `docs/TARGET_HOLE_*.md` / `GRASP_STABILITY_*.md` | Semantic plumbing / 抓取配方 |
| `configs/geometry_families.yaml` | 正式族定义 |
| `outputs/state.json` | 运行态 |

*文档版本：v0-acceptance · 2026-08-14 · 不授权采集/训练/真实写盘。*
