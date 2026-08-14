# Semantic Data Design（只读审计，不采集、不训练）

- 日期：2026-08-13
- 背景：P0-C0=`harmful_only`，P0-C1=`no_effect`，P0-C1.1=`load_calibration_fail`；手指稳定化路线暂停。
- 目的：设计多物体/多孔几何覆盖，解除 Semantic / Observability held-out 阻塞。
- **本文件不做采集、不改仿真、不训练。**

## 1. 当前数据事实

- sidecar 100/100 episodes 只有一族几何：
  - peg：`industreal_round_peg_8mm`
  - socket：`industreal_tray_insert_round_peg_8mm`
- arena 硬编码 include 上述两个 XML：
  - `dexjoco/dexjoco/sim/envs/xmls/arena_arm_hand_bimanual_assembly.xml`
- 接触/插入标签与 env 也硬编码 8mm body / site / geom 名（如 `assembly_contacts.py`、`panda_bimanual_assembly_env.py`、hybrid/pose insert）。

结论：当前策略即使过 Controllability，也无法学到跨几何插孔语义；固定 tip/lateral/axis 误差≠通用语义。

## 2. 仓库已有资产盘点

### 2.1 Mesh（`industreal/mesh/industreal_pegs/`）

| 截面 | peg | tray_insert（孔/托盘） | tray_pick（抓取托盘，非插入孔） |
|---|---|---|---|
| round | 4 / 8 / 12 / 16 mm | 4 / 8 / 12 / 16 mm | 4 / 8 / 12 / 16 mm |
| rectangular | 4 / 8 / 12 / 16 mm | 4 / 8 / 12 / 16 mm | 4 / 8 / 12 / 16 mm |

### 2.2 URDF + 官方尺寸表

- URDF：`industreal_round_peg_*`、`industreal_round_hole_*`、`industreal_rectangular_peg_*`、`industreal_rectangular_hole_*`（各 4/8/12/16）。
- 尺寸真值：`industreal/yaml/industreal_asset_info_pegs.yaml`。

### 2.3 MuJoCo XML wrapper

- **仅有** `industreal_round_peg_8mm.xml` 与 `industreal_tray_insert_round_peg_8mm.xml`。
- 其他尺寸只有 mesh/URDF，**没有**可直接 include 的 arena XML。
- 注意：现有 8mm XML 的 collision 是近似 primitive（peg cylinder `size≈0.01785×0.0675`，mesh `scale=4.5`），与 yaml 中名义直径 ~8mm **不是同一套数字**；多几何接入时必须统一命名、site、collision 与 clearance 口径，不能混用“资产名 mm”与“collision 米制”。

## 3. 可组成的真实 object–hole family

只允许**配对**使用（同截面、同名义尺寸）：

| family_id | object | socket | 截面 | 对称性 | mating feature |
|---|---|---|---|---|---|
| round_4 | round_peg_4mm | tray_insert_round_peg_4mm | 圆 | 绕孔轴近 SO(2) | 径向 clearance 插入 |
| round_8 | round_peg_8mm | tray_insert_round_peg_8mm | 圆 | 同上 | 同上（当前唯一在线） |
| round_12 | round_peg_12mm | tray_insert_round_peg_12mm | 圆 | 同上 | 同上 |
| round_16 | round_peg_16mm | tray_insert_round_peg_16mm | 圆 | 同上 | 同上 |
| rect_4 | rectangular_peg_4mm | tray_insert_rectangular_peg_4mm | 矩形 | 需 yaw 对齐 | 宽/深双轴 clearance |
| rect_8 | rectangular_peg_8mm | tray_insert_rectangular_peg_8mm | 矩形 | 需 yaw | 矩形宽深不对称（yaml depth≠width） |
| rect_12 | … | … | 矩形 | 需 yaw | 同上 |
| rect_16 | … | … | 矩形 | 需 yaw | 同上 |

禁止：
- round peg × rectangular hole 等错配；
- 把 `tray_pick_*` 当插入 socket（那是抓取托盘 mesh，不是 mating hole）；
- 仅换轨迹却声称多几何。

## 4. Clearance / 长度 / 截面（来自 yaml，单位 m）

### Round（peg diameter vs hole diameter）

| size | peg Ø | hole Ø | 径向间隙（直径差） | peg length | hole height/depth |
|---|---|---|---|---|---|
| 4mm | 0.003988 | 0.0041 | ≈0.112 mm | 0.050 | 0.028 / 0.023 |
| 8mm | 0.007986 | 0.0081 | ≈0.114 mm | 0.050 | 0.028 / 0.023 |
| 12mm | 0.011983 | 0.0122 | ≈0.217 mm | 0.050 | 0.028 / 0.023 |
| 16mm | 0.015983 | 0.0165 | ≈0.517 mm | 0.050 | 0.028 / 0.023 |

### Rectangular（peg width×depth vs hole width；depth 表列为孔深）

| size | peg W×D | hole W | 备注 |
|---|---|---|---|
| 4mm | 0.00397×0.00397 | 0.00411 | 近方截面 |
| 8mm | 0.007964×0.006910 | 0.0081444 | **扁矩形**，yaw 语义更强 |
| 12mm | 0.011957×0.007910 | 0.0121778 | 扁矩形 |
| 16mm | 0.015957×0.009910 | 0.0162182 | 扁矩形 |

语义含义：
- round 系列主要考 **径向对准 + 轴向插入**；
- rectangular 系列额外考 **绕孔轴朝向**；
- 尺寸变大 clearance 相对更松（尤其 16mm round），可做难度轴，但不能只靠 8mm 一种。

## 5. 最小多几何数据规模（设计，不采集）

建议 **最小 Semantic P0 数据集**（在现有 8mm 之外新增）：

1. **Train families（≥3）**  
   - `round_8`（已有可复用）  
   - `round_12`  
   - `rect_8`（引入朝向 mating）
2. **Object/geometry held-out（≥2）**  
   - `round_16`（同截面尺寸 OOD）  
   - `rect_12` 或 `rect_4`（截面+尺寸 OOD）
3. **每 family 轨迹规模（烟测→正式）**  
   - smoke：每 family ≥10 完整 demo（含 grasp→transport→insert）  
   - Semantic P0：每 train family ≥40–50；每个 held-out ≥20  
   - 总计量级：约 150–250 episodes（远小于“无限调 load”成本，但必须多 XML）

Split 规则（冻结后再采）：
- **object-held-out**：训练未见过的 peg 尺寸/截面；socket 与之配对但策略不得记忆单一 tip 误差模板。
- **geometry-held-out**：整族 object+socket 未进训练；评估 insert_ok / 接触模式 / 对准误差分布。
- 禁止按 episode id 随机切分冒充几何 held-out。

## 6. 需要修改的工程接口（清单，本轮不改代码）

| 层级 | 路径/模块 | 改什么 |
|---|---|---|
| Arena XML | `arena_arm_hand_bimanual_assembly.xml` | 参数化 include peg/socket；或生成每 family 的 arena 变体 |
| Asset XML | `xmls/industreal_*_{4,12,16}mm.xml` 等 | **新建** MuJoCo wrapper（现仅 8mm）；统一 body/site/geom 命名约定 |
| Collision | 各 peg/tray XML | 按 yaml clearance 设置 primitive 或 mesh collision；记录 scale 与名义 mm 的映射 |
| Env | `panda_bimanual_assembly_env.py` | 去掉硬编码 8mm body/joint/geom id |
| Labels | `hybrid_insert/assembly_contacts.py` 等 | peg/tray/bottom/site 名配置化 |
| Controllers | `hybrid_insert/*`, `pose_insert/*` | socket_site / bottom_contact 配置化（即使后续禁 servo，标签也需要） |
| Task config | `CONFIG_MAPPING` / assembly task yaml | family_id、asset 路径、随机化范围 |
| Sidecar metadata | interaction_sidecar manifest | 每 episode 写 `object_asset`, `socket_asset`, `family_id`, `section`, `nominal_size_mm` |
| Obs contract | `full_obs.py` 文档 | 明确 peg7/tray7/tip 误差是 privilege，不叫部署语义；语义评估用 family held-out 指标 |
| Dataset split | 新 manifest | `train_families` / `heldout_object` / `heldout_geometry` |

## 7. 验收门槛（设计态）

在未采集前，Semantic 数据设计通过需满足文档级检查：
- ≥3 train + ≥2 held-out 配对 family 已定义；
- 每 family 有 mesh + 计划 XML 命名 + clearance 表；
- hardcoded 8mm 调用点清单完整；
- split 规则不依赖 tip 距离 gate 冒充几何。

采集与训练仍禁止，直到该设计审阅通过且 Observability/Controllability 硬门允许。

## 8. 与 Controllability 路线的关系

- 手指稳定化 / wrist load 调参：**停止**。
- 原因：无法在单几何、难构造可恢复 unstable root 的设定下继续获得稳定干预证据；继续调 load 易沦为无限调参。
- 更大阻塞：单 peg/socket → 无法学插孔语义。
- P0-C0/C1/C1.1 产物全部保留，不作覆盖删除。
