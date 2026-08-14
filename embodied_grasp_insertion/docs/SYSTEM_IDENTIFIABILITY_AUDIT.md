# System Identifiability Audit (P0-A)

- 日期（UTC）：2026-08-13T13:39:01Z
- 结论：**partial**
- 有效 episode：100；排除：0
- object/socket/geometry family：1/1/1
- 85D/44D 与代码一致：True

## 1. 85D observation 分段

| 段 | slice | dim | 类别 | 来源 |
|---|---|---|---|---|
| `act44` | [0:44] | 44 | deployment_observable | current_action44(raw) via read_arm_action + dual_arm23_to_action44 |
| `peg7` | [44:51] | 7 | simulator_privileged | raw._data.qpos[raw._peg_qpos_adr:adr+7] freejoint |
| `tray7` | [51:58] | 7 | simulator_privileged | raw._data.qpos[raw._socket_qpos_adr:adr+7] freejoint |
| `lat_vec3` | [58:61] | 3 | simulator_privileged | privileged_full_features → lateral_error(tip, socket, hole) |
| `along_tip_axis` | [61:64] | 3 | simulator_privileged | [along, tip_dist, axis_err] from privileged_full_features |
| `hole_axis3` | [64:67] | 3 | simulator_privileged | unit hole opening axis from socket site + bottom geom |
| `peg_axis3` | [67:70] | 3 | simulator_privileged | unit peg body z-axis |
| `ft12` | [70:82] | 12 | deployment_observable | FingerForceLabeler wrist_ft_right/left minus baseline |
| `flags3` | [82:85] | 3 | simulator_privileged | FullEpisodeEnv._flags: tray_ok_seen, peg_ok_seen, peg_ok_seen(duplicate) |

### 部署可用 vs 特权

- **deployment_observable**：`act44`（本体/指令）、`ft12`（腕力；若真机有腕力传感器）。
- **simulator_privileged**：`peg7`、`tray7`、相对几何、孔/销轴、outcome flags。
- **unavailable（相对本项目目标）**：逐指接触不在 85D；无 slip；无 object-in-hand 6D 字段；无物体/孔几何描述符；无 capture/rim/jam/backout 模式。

## 2. object-in-hand 6D

- 85D/sidecar 是否直接提供：否。
- 能否从仿真状态推导：是（peg freejoint 相对腕/手坐标系），类别为特权派生。 详见 conclusions.object_in_hand_6d。

## 3. 逐指接触与滑移

- 逐指力：`FingerForceLabeler` 可从 `cfrc_ext` 得到 4×3×双手，但**未写入 85D**，也不在 sidecar 时序里。
- 滑移真值：无现成标签；不得伪造。
- `AssemblyOutcome` 仅有 tray_ok/peg_ok/insert_ok 与接触计数，无逐指 retention。

## 4. 44D 动作是否控制全部手指

- **FullEpisodeEnv：是**。右/左各 16 维手指增量，`finger_scale=0.15`，代码写明 fingers free。
- **InsertHandoffEnv：否**。wrist12（或 riva）只动腕，手指在 handoff 冻结。

## 5. Snapshot 与 matched intervention

- `InsertEnvSnapshot`：深拷贝 `MjData` + wrapper 字段，可精确 restore → **insert 阶段 matched intervention：可用**。
- `CompactInsertSnapshot`：recovery 磁盘上有 pkl（insert 根）；同样是 insert 阶段。
- `FullEpisodeEnv`：**无** snapshot/restore API；只能 zarr `initial_state`+动作重放 → 抓持阶段精确分支 **部分可用/缺口**。

## 6. 几何多样性

- family `industreal_round_peg_8mm__industreal_tray_insert_round_peg_8mm`：100 episodes
- 结论：True （同一 `industreal_round_peg_8mm` + `industreal_tray_insert_round_peg_8mm` 的不同轨迹/位姿）。
- 仓库 mesh 目录另有 4/12/16mm 与 rectangular 资产，但**未出现在本 100 episode 覆盖中**。

## 7. Split 可行性

- episode-held-out：True
- object-instance-held-out：False
- geometry-family-held-out：False

## 8. P0 准入判断

- Observability P0 可否完整开始：**False**
  - 原因：85D lacks deployment finger contact / object-in-hand; single geometry cannot support object/geometry-held-out controls required by project hard gates.
- Controllability P0 smoke 可否开始：**True**
  - 原因：FullEpisodeEnv exposes 44D finger control; InsertEnvSnapshot enables matched insert-phase restore. Controllability smoke can run on single geometry, but Semantic P0 still blocked.

## 9. 最小缺失与定向重放

- 缺失：object_in_hand_relative_pose_6d_as_labeled_series
- 缺失：per_finger_contact_force_or_binary_retention_series
- 缺失：slip_proxy_or_truth_series
- 缺失：contact_mode_beyond_tray_ok_peg_ok_insert_ok
- 缺失：multiple_object_hole_geometry_families_in_episode_coverage
- 缺失：FullEpisodeEnv.snapshot/restore for grasp-phase matched intervention
- 需求：Optional: single-episode FullEpisodeEnv smoke to dump privileged o2h + finger forces alongside 85D
- 需求：Optional: add snapshot API to FullEpisodeEnv before Controllability P0
- 需求：Required for Semantic/Observability hard gates: new episodes with alternate IndustReal peg/tray meshes already present in repo assets

## 10. 与停止条件

- MOTIVATION §8 停止条件（多样性仅为同位姿变化）：**触发=True**
- 所谓多样性实际只是同一几何的位姿/轨迹变化（100/100 同为 round_peg_8mm + tray_insert_round_peg_8mm）。
- 同时 snapshot 可恢复手/物/接触物理态，且 episode split 可行 → 总评 **partial**，不是全面 fail。

