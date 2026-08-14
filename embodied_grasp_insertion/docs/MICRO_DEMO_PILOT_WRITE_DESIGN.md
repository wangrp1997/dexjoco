# Micro-demo Pilot：单条 Trajectory 写入实现设计（评审稿）

状态：**设计 + mock 单测** — `WRITE_IMPLEMENTATION_ENABLED=False`；不授权真实写盘；不采集；不写正式 `out_root`。

前置：dry-run 安全加固已审定通过。

---

## 1. 目标与非目标

### 目标

- 定稿单条 trajectory 目录布局、`meta.json` / `labels.json` / `states.npz` / run manifest schema。
- 定稿原子写入流程：隔离临时目录 → 校验 → `fsync` → 原子 `rename` → 失败回滚。
- 用 **临时目录 mock** 单测写入器；不触碰正式 `data/pilot_micro_demo_v0/`。

### 非目标

- 不把 `WRITE_IMPLEMENTATION_ENABLED` 设为 `True`。
- 不接真实 MuJoCo 采集进写盘路径。
- 不训练、不把 pilot 数据挂进 dataloader。

---

## 2. 目录与文件布局（单条）

正式根（仅未来授权后）：

```
embodied_grasp_insertion/data/pilot_micro_demo_v0/
  README.md
  PILOT_BANNER.json          # {"training_forbidden":true,"revocable":true,"pilot_tag":"..."}
  trajectories/
    <uuid>/
      meta.json
      labels.json
      states.npz             # 可选；v0 mock 可写最小数组
      COMMITTED              # 空标志文件；仅 rename 成功后存在
  manifests/
    run_<utc>_<run_id>.json
  .tmp/                      # 写入进行中；成功后清空；崩溃可扫删
    <run_id>/
      traj_<uuid>/
      manifest.json.partial
```

约束：

- `traj_id` 必须为 UUID；路径不得由用户字符串拼接。
- 首次写入：`out_root` 不存在或为空（仅允许我们创建的脚手架文件策略见下）。
- **禁止覆盖**：`trajectories/<uuid>/` 或同名 run manifest 已存在 → 拒绝。
- 禁止 symlink（逐级 `lstat`）。

脚手架（首次成功 commit 时原子创建）：`README.md`、`PILOT_BANNER.json`。若根已存在且非空且缺脚手架 → 拒绝（防混入脏目录）。

---

## 3. Schema（必填）

### 3.1 `meta.json`

| 字段 | 类型 | 约束 |
|------|------|------|
| `traj_id` | str | UUID |
| `pilot_tag` | str | `micro_demo_pilot_v0` |
| `training_forbidden` | bool | 必须 `true` |
| `dry_run` | bool | 真实写入必须 `false` |
| `geometry_family_id` | str | 非空 |
| `target_instance_id` | str | 非空 |
| `socket_site` | str | 非空 |
| `root_source` | str | `demo_transport` \| `oracle_establish_formal` |
| `matched_snapshot_branch` | bool | 必须 `true` |
| `snap_call_count_after_establish` | int | 必须 `0` |
| `is_insertion_demo` | bool | 必须 `false`（v0） |
| `created_at` | str | UTC `...Z` |
| `horizon_steps_used` | int | `>=0` |
| `horizon_budget_max` | int | `MIN..MAX` |
| `oracle_usage` | object | 含 snap 计数说明 |
| `episode_index` / `root_frame` | int\|null | demo 路径必填 |

### 3.2 `labels.json`

| 字段 | 约束 |
|------|------|
| `gates` | 列表，含三门 `name/passed` |
| `all_gates_passed` | 必须 `true` 才允许 commit |
| `insert_phase` | `skipped` |
| `insert_ok` | `false` |
| `is_insertion_demo` | `false` |
| `stop_reason` | 成功时 `null` |

### 3.3 `states.npz`

- 仅允许数值 dtype（禁 `object`）。
- 总字节 ≤ `MAX_STATES_NPZ_BYTES`。
- v0 mock 最小键：`t` (int64, shape `[T]`)、`dummy` (float64, shape `[T,1]`)。
- 真实采集格式可后扩，但不得 pickle。

### 3.4 run manifest

| 字段 | 说明 |
|------|------|
| `protocol` | `micro_demo_pilot_v0` |
| `run_id` | UUID |
| `created_at` | UTC |
| `dry_run` | `false`（真实写） |
| `WRITE_IMPLEMENTATION_ENABLED` | 当时常量 |
| `config_sha256` / `code_caps` / `seed` | 复现 |
| `trajectories` | `[{traj_id, path, gates_ok}]` |
| `verdict` | `write_ok` \| `aborted` \| `refused` |
| `rollback` | 推荐命令（allowlist 脚本） |

校验实现：`pilot/traj_schema.py`。

---

## 4. 原子写入流程

```
assert WRITE_IMPLEMENTATION_ENABLED  # 生产入口；当前恒 False → refused
assert out_root allowlist + 无 symlink + 空/可初始化
assert traj_id UUID 且最终路径不存在
assert meta/labels schema + all_gates_passed

tmpdir = out_root/.tmp/<run_id>/traj_<uuid>/   # 先建 .tmp（同根 allowlist 内）
write meta.json, labels.json, states.npz into tmpdir
fsync each file + fsync tmpdir
validate bytes on disk (re-read schema)

# 最终 traj 目录原子出现：
os.rename(tmpdir, out_root/trajectories/<uuid>)   # 同文件系统
write COMMITTED into final dir (或 rename 前写入 tmp 再整体 rename)
fsync parent trajectories/

# run manifest 最后写：
write manifests/run_....json.partial → fsync → rename → fsync manifests/
```

失败：

1. 任意步骤异常 → 删除本次 `.tmp/<run_id>/`（allowlist 校验后）。
2. 若 `trajectories/<uuid>` 已出现但不含完整文件 → 视为脏；删除该 traj（仅当缺 `COMMITTED` 或 schema 失败）。
3. **不**留下半条可被训练误读的轨迹（无 `COMMITTED` 的目录不算有效）。

禁止覆盖：`rename` 目标存在则失败（`O_EXCL` 语义）；不 `rm` 后重写。

---

## 5. API 分层（保持开关关闭）

| API | 行为 |
|-----|------|
| `commit_trajectory(...)` | 生产入口；若 `WRITE_IMPLEMENTATION_ENABLED=False` → 立即 `PilotWriteRefused` |
| `commit_trajectory_mock(out_root=/tmp/..., ...)` | **仅单测**；要求 `out_root` 在 `/tmp` 下；不读正式 allowlist 根；不要求开关为 True |
| `rollback_micro_demo_pilot` | 已有 allowlist 删除；设计不变 |

正式 runner **不得**调用 `commit_trajectory_mock`。

---

## 6. 单测计划（无 MuJoCo / 无正式 out_root）

`tests/test_pilot_atomic_write.py`：

1. schema 接受/拒绝（缺字段、`insert_ok=true`、非 UUID）。
2. mock 根下成功 commit：最终存在 `COMMITTED`，无残留 `.tmp`。
3. 目标已存在 → 拒绝覆盖。
4. 写入中途注入失败 → `.tmp` 清理，无最终 traj。
5. `states.npz` object dtype / 超字节 → 拒绝。
6. `commit_trajectory` 在开关 False 时 refused（不创建任何目录）。
7. mock 拒绝把 `out_root` 指到正式 `ALLOWED_OUT_ROOT`。

---

## 7. 评审通过后仍不自动写盘

勾选后才可另开一轮讨论：

- [ ] 本设计评审通过  
- [ ] mock 单测全绿  
- [ ] 显式授权将 `WRITE_IMPLEMENTATION_ENABLED=True`（单独变更）  
- [ ] 用户同意写 **1** 条真实 trajectory  

---

## 8. 本轮交付文件

| 文件 | 作用 |
|------|------|
| `docs/MICRO_DEMO_PILOT_WRITE_DESIGN.md` | 本文 |
| `pilot/traj_schema.py` | meta/labels/manifest/npz 校验 |
| `pilot/atomic_write.py` | 原子写入 + mock 入口 |
| `tests/test_pilot_atomic_write.py` | 临时目录单测 |

*不授权真实 trajectory 落盘。*
