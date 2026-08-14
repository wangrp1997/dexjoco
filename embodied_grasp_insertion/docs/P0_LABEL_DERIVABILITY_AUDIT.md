# P0-L0 Label Derivability Audit（只读）

- 日期：2026-08-14
- 协议：`P0-L0`
- 范围：从现有 `FullEpisodeSnapshot` + MuJoCo model/data + sidecar 是否能**确定性派生**三门共用标签
- **不**采集、**不**训练、**不**打开写盘、**不**重开 C0/C1/C1.1
- 依据代码：`simulation/full_episode_snapshot.py`、`physics/grasp_metrics.py`、`geometry/target_hole.py`、`FullEpisodeEnv` / `AssemblyContactLabeler` / `FingerForceLabeler`

## 0. 总判定（摘要）

| # | 标签 | 判定 | 一句话 |
|---|---|---|---|
| 1 | object-in-hand 6D pose/velocity | **derivable**（pose）/ **partial**（vel） | pose 由 body xpos/xmat 相对 palm 确定性算；vel 需时间差分或 `cvel`，口径须冻结 |
| 2 | per-finger contact retention | **partial** | 接触计数/力阈值可派生；“retention” 相对 root 的定义已有，但力阈值≠几何接触真值 |
| 3 | slip truth/proxy | **partial**（proxy only） | 仅有 o2h 有限差分 proxy；无接触切向真值 |
| 4 | contact mode | **partial** / 细模式 **blocked** | 仅有 tray_ok/peg_ok/insert_ok+计数；capture/rim/jam/backout 未定义 |
| 5 | regrasp-needed / peg-loss risk | **partial** | 有 `_peg_lost` / `peg_ok` 启发；无正式风险标签契约 |
| 6 | object-target mating transform | **partial** | tip/socket/hole_axis 可派生；完整 6D mating SE(3) 未统一出口 |
| 7 | provenance | **derivable**（episode/family/instance/snapshot 核心）/ **partial**（root_id 需约定） | snapshot 含 episode/zarr；family/instance 从 raw names；root_id 未内建字段 |

**总评**：多数核心特权标签 **可派生或可部分派生**；阻塞点是 **细 contact-mode 契约未写** 与 **slip 仅有 proxy**。不阻止下一步小型 Observability **label smoke**（特权标签侧），但仍 **禁止** 部署 Obs 模型训练与 Controllability 重开。

---

## 1. object-in-hand 6D pose / velocity

### Pose

| 项 | 内容 |
|---|---|
| 来源字段 | `MjData.xpos` / `xmat`：`peg_body`、`allegro_palm_right` |
| 公式 | \(R_{rel}=R_{palm}^{-1} R_{peg}\)，\(t_{rel}=R_{palm}^{-1}(p_{peg}-p_{palm})\)；输出 `translation(3)` + `rotvec(3)` |
| 单位/坐标系 | m / rad；相对参考体 `allegro_palm_right`（手掌体坐标系） |
| 阈值 | 任意；单帧可算 |
| 部署 vs privilege | **仅 privilege label**（部署 85D 无此字段） |
| snapshot 一致性 | restore 后 `xpos/xmat` 随 MjData 恢复；bit-exact smoke 已证 |
| 已知歧义 | 参考体固定右手掌；左手抓持未定义；peg 子树原点≠ tip |
| 判定 | **derivable**（`physics/grasp_metrics.object_in_hand_pose`） |

### Velocity

| 项 | 内容 |
|---|---|
| 来源字段 | 相邻帧 o2h pose 差分，或 `MjData.cvel`（未封装） |
| 公式（现行 proxy） | \(\Delta t_{rel}/\Delta t_{ctrl}\)，\(\Delta\)rotvec 角速率；`control_dt_seconds`≈`timestep×frame_skip` |
| 阈值 | 需 ≥2 帧；单 snapshot **不够** |
| 歧义 | 有限差分 vs 空间速度；palm 系切向定义粗糙 |
| 判定 | **partial**（可派生但须冻结差分口径；单帧 blocked） |

---

## 2. per-finger contact retention

| 项 | 内容 |
|---|---|
| 来源 A | `data.contact[]`：peg geom ↔ 右手 body（palm/index/middle/ring/thumb 前缀）→ 计数 |
| 来源 B | `FingerForceLabeler`：`cfrc_ext` → `right_finger_force(12)`；`norm>=eps` → `contact_active(4)` |
| retention 公式（v2） | 相对 **root 时刻** 总接触数：`min(c_t / c_root, 1)`；root 无接触时用 binary |
| 单位 | 无量纲比例；力阈默认 `eps=0.05`（牛顿量级，依赖 labeler 标定） |
| 部署 vs privilege | privilege（接触计数与 cfrc 力均非 85D） |
| snapshot | contact buffer 在 MjData deepcopy 内；restore 后可再算 |
| 歧义 | 计数 vs 力 active 可不一致；unknown geom 归类；相对 root vs 相对第一步（C0 legacy） |
| 判定 | **partial**（可确定性算 proxy retention；非“物理真 retention 传感器”） |

---

## 3. slip truth / proxy

| 项 | 内容 |
|---|---|
| 现行 proxy | o2h 平移速率范数；姿态漂移率相对 root |
| 真值 | **无**：无接触点切向速度 / 摩擦锥滑移标志序列 |
| 部署 | privilege proxy only |
| 歧义 | 已写明 “NOT ground truth”；不得当 slip 真值训部署 |
| 判定 | **partial**（proxy derivable）；**truth blocked** |

---

## 4. contact mode

| 项 | 内容 |
|---|---|
| 现有 | `AssemblyContactLabeler` → `tray_ok` / `peg_ok` / `insert_ok` + 接触计数 |
| 缺失 | capture / rim / jam / partial / seated / backout 无统一枚举与几何阈值 |
| 判定 | 粗 outcome：**partial**；细 mode：**blocked**（需先写契约再导出） |

---

## 5. regrasp-needed / peg-loss risk

| 项 | 内容 |
|---|---|
| 来源 | `FullEpisodeEnv._peg_lost`、`peg_ok`、`peg_ok_seen`；summary 中 `drop_z` / 漂移阈值；legacy `peg_loss := not terminal_peg_ok` |
| 问题 | legacy `peg_loss` ≠ 物理掉落；无 horizon 风险概率标签 |
| 判定 | **partial**（启发可派生）；正式 risk 标签 **blocked** 直至契约 |

---

## 6. object-target mating transform

| 项 | 内容 |
|---|---|
| 来源 | `_insert_geometry`：tip / socket / hole_axis；`target_hole.semantic_target_features`；peg/socket body pose |
| 可派生 | tip→socket 平移；hole/peg 轴；family/instance 身份 |
| 缺口 | 统一 `T_object_feature←target_feature` SE(3) 未封装为单一 label API |
| 部署 | privilege（85D 中 tip/axis 亦为 privilege，不得当语义） |
| 判定 | **partial**（几何要素 derivable；完整 mating SE(3) 出口未定） |

---

## 7. provenance（episode / root / family / instance / snapshot）

| 字段 | 来源 | 判定 |
|---|---|---|
| `episode_index` | `FullEpisodeSnapshot.episode_index` / `_spec` | **derivable** |
| `zarr_path` | snapshot | **derivable** |
| `t` / `raw_env_step` | snapshot | **derivable** |
| `geometry_family_id` | `names_from_raw` / arena | **derivable**（正式多族 env）；sidecar 旧 100ep 仅 round_8mm |
| `target_instance_id` / `socket_site` | `target_hole_from_raw` | **derivable** |
| `root_id` | 无内建；需约定 `f"{episode}:{frame}:{phase}"` | **partial**（可约定，未写入 snapshot） |
| snapshot 引用 | `FullEpisodeSnapshot` 对象 / 磁盘序列化未标准化 | **partial**（内存可；磁盘 schema 未定） |

---

## 8. Sidecar 现状 vs 在线派生

| 内容 | sidecar 时序 | 在线从 restore 派生 |
|---|---|---|
| 85D obs | 有 | 有 |
| o2h 6D | **无** | **有** |
| 逐指力/retention | **无**（labeler 可算但未落盘） | **有** |
| slip series | **无** | proxy **有** |
| 细 contact mode | **无** | **无** |
| family held-out 覆盖 | 单族 | 正式 arena 多族可用，但 demo zarr 仍单族 |

结论：标签缺口主要是 **未导出/未契约**，不是物理不可算（细 mode 除外）。

---

## 9. 只读 smoke 结果

- 脚本：`scripts/audit_label_derivability.py`
- manifest：`data/manifests/label_derivability_audit_v1.json`
- MuJoCo：ep0 frame=287；restore 一致性 o2h/contact/target/outcome **all_ok**
- 接触样例：total=4（index=2, thumb=2）
- `WRITE_IMPLEMENTATION_ENABLED=False`；未采集/训练/写盘
- overall：`ready_for_observability_label_smoke`（1 derivable + 7 partial；0 blocked 项为整类，细 mode/slip-truth 记在 partial notes）

## 10. 分支建议（审计后）

1. **本轮**：可启动小型 **Observability label smoke**（特权标签导出契约），仍不训模型。  
2. 细 contact-mode / slip-truth：先定契约再导出，禁止用粗 outcome 冒充。  
3. Controllability：**不重开**，除非新 root/动作接口假设。  
4. 真实写盘：仍不授权。

*版本：v0 · 2026-08-14 · readonly smoke pass*
