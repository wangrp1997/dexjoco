# Micro-Demo Pilot 方案 v0（有条件通过后契约修订）

状态：**写入实现设计评审中** — dry-run 加固已通过；`WRITE_IMPLEMENTATION_ENABLED=False`；mock 单测可用；不自动真实写盘。

前提：P0-S0.4c-hardened 三项回归 **4/4**。  
证明范围：recipe 基础设施。不证明策略自主抓取；禁止常规采集/训练。

---

## 0. 本修订相对原稿的契约修正

1. **dry-run 零写入（repo）**  
   - dry-run 只打 stdout。  
   - 可选 `--dry-run-report /tmp/...`（必须在 `/tmp` 下，且不在 repo 内）。  
   - **不得**写 `out_root`、repo 内文件或 `data/manifests/micro_demo_pilot_v0.json`。  
   - 正式/独立 manifest 仅在未来授权写入版本生成。

2. **硬上限（代码常量，YAML 不可放宽）**  
   - `MAX_FAMILIES=1`  
   - `MAX_TOTAL_TRAJECTORIES=1`  
   - `MAX_EPISODES_PER_FAMILY=2`  
   - `MAX_TRAJECTORIES_PER_EPISODE=1`  
   - `MAX_HORIZON_STEPS=80`  
   - 更大的「≤4」属于**未来新版本方案**，不是本 runner 可配置上限。

3. **输出路径 allowlist**  
   - 唯一允许根：`embodied_grasp_insertion/data/pilot_micro_demo_v0/`（`Path.resolve()` 后校验）。  
   - 禁止 symlink、`..`、绝对路径覆盖、前缀字符串绕过。  
   - 采用 allowlist，不以 denylist 为主。

4. **禁止覆盖**（未来写入）  
   - `out_root` 须不存在或为空。  
   - `traj_id` / run id 已存在则拒绝。  
   - 删除前必须重新 `assert_under_allowlisted_out_root`。

5. **原子落盘**（未来写入；本版未实现）  
   - 隔离临时目录 → 三门+schema+`fsync` → 原子 rename。  
   - 失败删临时目录；run manifest 最后原子写。

6. **写入四条件**（本版一律 `verdict=refused`）  
   - `dry_run: false`  
   - `--allow-write`  
   - `--i-understand-pilot-is-revocable`  
   - `WRITE_IMPLEMENTATION_ENABLED=True`（当前为 False）

7. **禁训**  
   - banner/metadata **不能**单独阻止读取。  
   - 共享 guard：`pilot.paths.assert_not_pilot_path_for_training`。  
   - 已在 `make_full_env` 入口调用。  
   - **在更多训练入口接入前，不得声称“所有训练脚本必然 raise”**；仅写成实施要求。

### 0.1 dry-run 安全加固（2026-08-14，仍不写盘）

- 严格 schema：`pilot/config_schema.py`；未知字段/错误类型/非正整数 cap → 创建环境前 `aborted`
- v0 全部 gates 强制 `true`；任一门为 false → `aborted`（不可静默忽略）
- horizon：按实际 `env.step` 计数；schema 要求 `MIN_HORIZON_STEPS=5` … `MAX_HORIZON_STEPS=80`（`<5` 在校验阶段 code 3，不再落到 plan/执行）
- 路径：原始路径逐级 `lstat` 拒 symlink，再 resolve；禁训同时查字符串与 resolved∈`ALLOWED_OUT_ROOT`
- `/tmp` 报告：`O_CREAT|O_EXCL|O_NOFOLLOW`，拒绝覆盖与 symlink
- 单测：`tests/test_pilot_dry_run_guards.py`（无 MuJoCo）
- **`WRITE_IMPLEMENTATION_ENABLED` 仍为 False；不进入写入实现**

---

## 1. v0 runner 范围（当前唯一授权实现）

> **仅 dry-run、全程内存、不创建 `out_root`、不写正式 manifest；所有写入旗标均 `refused`。**

命令：

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= \
  python -m embodied_grasp_insertion.scripts.run_micro_demo_pilot \
  --config embodied_grasp_insertion/configs/micro_demo_pilot.yaml
```

可选：

```bash
  --dry-run-report /tmp/micro_demo_pilot_dry_run.json
```

---

## 2. 目录布局（仅未来写入；本版不创建）

```
embodied_grasp_insertion/data/pilot_micro_demo_v0/
  README.md
  PILOT_BANNER.json
  trajectories/<uuid>/
  manifests/run_<utc>.json
```

回滚：使用 allowlist 命令，不鼓励裸 `rm -rf`：

```bash
python -m embodied_grasp_insertion.scripts.rollback_micro_demo_pilot --target ... --yes
```

---

## 3. 记录与门检（dry-run 内存缓冲）

- `traj_id`：随机 UUID（禁止用户字符串拼路径）。  
- meta：family / instance / site / root_source / oracle / snap_after / `training_forbidden` / `dry_run`。  
- 三门：物理抓取、目标语义、插入标签一致性。  
- `insert_phase=skipped` ⇒ `insert_ok=false`，`is_insertion_demo=false`。  
- 任一门失败 ⇒ `verdict=aborted`，不再开下一条。  
- dry-run **只用内存缓冲**，不靠「写临时文件再删」模拟。

未来 `states.npz`：禁止 pickle/object dtype；限制总字节数（常量 `MAX_STATES_NPZ_BYTES`）。

Manifest（未来）应含：代码版本、配置哈希、arena XML 哈希、seed、依赖 smoke manifest。

---

## 4. 放行清单（第一次真正写入前）

- [ ] dry-run runner `verdict=dry_run_ok`  
- [ ] 写入实现单独评审（原子化 + allowlist + 禁覆盖）  
- [ ] `WRITE_IMPLEMENTATION_ENABLED` 变更经显式授权  
- [ ] 用户同意写 **1** 条  
- [ ] 四条件同时满足  
- [ ] 回滚命令演练通过  

未勾满 → 禁止写盘。

---

## 5. 相关文件

| 文件 | 作用 |
|------|------|
| `pilot/__init__.py` | 代码硬顶与 `WRITE_IMPLEMENTATION_ENABLED=False` |
| `pilot/paths.py` | allowlist / 禁训 guard / dry-run report 路径 |
| `pilot/dry_run.py` | 内存三门 |
| `scripts/run_micro_demo_pilot.py` | dry-run-only CLI |
| `scripts/rollback_micro_demo_pilot.py` | allowlist 删除 |
| `docs/MICRO_DEMO_PILOT_WRITE_DESIGN.md` | 单条写入设计评审稿 |
| `pilot/traj_schema.py` | meta/labels/manifest/npz schema |
| `pilot/atomic_write.py` | 原子写入；生产拒绝；mock 仅 `/tmp` |
| `tests/test_pilot_atomic_write.py` | 写入 mock 单测 |

---

*文档版本：v0-dry-run · 2026-08-14 · 不授权任何 trajectory 落盘。*
