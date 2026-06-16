# DexQuery 实现步骤

## 外部依赖：

```
*/lerobot/    # pip install -e . 一次即可
*/dexjoco/    # PYTHONPATH 含此目录，以便 import dexquery
```

---

## Step 0：环境与协议对齐

- [x] Conda 环境：与 DexJoCo / LeRobot 训练环境一致（已有 ACT 训练环境即可）
- [x] LeRobot：`pip install -e /home/wangrenpeng/lerobot`
- [x] 读 [`docs/custom_policy_integration.md`](../docs/custom_policy_integration.md)：
  - 输入：三相机 + `state[:46]` + `prompt`（不用 privileged pose）
  - 输出：44d rotvec action；chunk horizon 与 baseline 一致（30）
- [x] 首任务：`bimanual_assembly`；数据集：`/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly`

---

## Step 1：Sim contact 标注（阻塞训练）

**目标**：为每一帧自动生成 `tray_ok`、`peg_ok`（训练用特权标签；推理只用视觉 outcome 头）。

**输出（sidecar，不改原数据）**：

```
<mnt/ssd/.../bimanual_assembly/dexquery_labels/
  outcomes.parquet      # index, episode_index, frame_index, tray_ok, peg_ok, insert_ok
  manifest.json
  summary.json
```

- [x] `scripts/label_contact.py` — 主入口
- [x] `data/assembly_contacts.py` — MuJoCo contact 检测（左 hand ↔ tray，右 hand ↔ peg）
- [x] `data/episode_replay.py` — sim replay + 可选 `restore_initial_state`
- [x] `data/lerobot_io.py` — 读 LeRobot parquet 对齐 `index`
- [x] `data/zarr_io.py` — 从 `replay.zarr` 读完整 `state[0]`（**推荐**）
- [x] `data/label_store.py` / `data/outcome_labels.py` — 读写 sidecar

**推荐命令（准确）**：

```bash
cd /home/wangrenpeng/dexjoco
export PYTHONPATH=/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco
MUJOCO_GL=egl conda run -n lerobot python dexquery/scripts/label_contact.py \
  --task bimanual_assembly \
  --dataset-root /mnt/ssd/datasets/dexjoco_lerobot_datasets \
  --zarr-input-dir /path/to/raw/bimanual_assembly/demos
```

**无 zarr 时（近似，不推荐）**：加 `--allow-seed-only`（场景与 demo 不对齐，标签稀疏/不准）。

- [ ] 提供/确认 raw zarr demo 路径后跑全量 100 episodes
- [ ] 抽检：`summary.json` 中 `tray_ok_rate` / `peg_ok_rate` 合理；成功 demo 中段应有连续 1

---

## Step 2：核心模型 `models/`

**架构（v1）**：

```
3 × RGB → 共享 SigLIP/CLIP ViT-B → patch tokens
3 条子任务 text（config 模板）→ text encoder → query
query cross-attn patch tokens → z_tray, z_peg, z_insert
z_tray, z_peg → outcome BCE
z_* + proprio(46) → action head → 44d × chunk_size
Loss = BC(action) + λ · BCE(outcome)
```

- [ ] `models/vision_backbone.py` — ViT patch 提取（参考 OpenPI `pi0_pytorch.py` SigLIP 用法）
- [ ] `models/subtask_query.py` — text encoder + cross-attn（参考 PerAct **写法**，视觉 token 用 2D patch 非体素）
- [ ] `models/outcome_head.py` — `z_tray/z_peg` → sigmoid BCE
- [ ] `models/action_head.py` — phase 条件或 3 head；v1 可先单 head + phase embedding
- [ ] `models/dexquery_model.py` — 组装 forward / loss
- [ ] 子任务句模板（v1 写死在 config，不用 test-time VLM）：
  - `Grasp the tray with the left hand.`
  - `Grasp the peg with the right hand.`
  - `Insert the peg into the hole.`

**参考（只抄模块思路，不 fork 整 repo）**：

| 模块 | 参考 |
|------|------|
| cross-attn | [peract/perceiver_lang_io.py](https://github.com/peract/peract) |
| ViT / SigLIP | `openpi/src/openpi/models_pytorch/pi0_pytorch.py` |
| action chunk | LeRobot ACT 输出形状对齐 |

---

## Step 3：数据层 `data/`

- [ ] `data/dataset.py` — `from lerobot.datasets.lerobot_dataset import LeRobotDataset`，读取三相机 + state + action + outcome 字段
- [ ] `data/collator.py` — batch 组装、normalize（可复用 LeRobot dataset stats）
- [ ] `data/subtask_prompts.py` — 任务 → 3 条子任务句映射

---

## Step 4：训练 `scripts/train.py`

- [ ] **自写训练循环**（不调用 `lerobot-train --policy.type=...`）
- [ ] 从 `configs/bimanual_assembly.yaml` + `configs/training/policies/dexquery.yaml` 读超参
- [ ] 与 pi0.5 baseline 对齐：batch=32, steps=60k, seed=0, action_horizon=30
- [ ] checkpoint 保存到 `checkpoints/dexquery_dexjoco_ckpt/<task>/`
- [ ] 修改 `scripts/train_dexjoco_lerobot.py`：`--policy dexquery` 时分发执行 `dexquery/scripts/train.py`（复用 config 合并逻辑，不改 lerobot）
- [ ] 添加 `configs/training/policies/dexquery.yaml`

**启动示例（完成后）**：

```bash
cd /home/wangrenpeng/dexjoco
export PYTHONPATH=/home/wangrenpeng/dexjoco:$PYTHONPATH

python scripts/train_dexjoco_lerobot.py \
  --policy dexquery \
  --task bimanual_assembly
```

---

## Step 5：推理接口 `policy/` + `inference/`

- [ ] `policy/dexquery_policy.py` — `select_action(obs)`、`load_checkpoint` / `save_checkpoint`（不必注册进 LeRobot factory）
- [ ] `inference/phase_controller.py` — 根据预测的 `tray_ok`、`peg_ok` 选择 active subtask / head：
  - `¬tray_ok` → π_tray
  - `tray_ok ∧ ¬peg_ok` → π_peg
  - `tray_ok ∧ peg_ok` → π_insert
- [ ] chunk 执行与 replan：复用 `dexjoco/dexjoco_lerobot_client/` 的 action buffer 逻辑，或 `scripts/eval.py` 独立评测

---

## Step 6：评测

- [ ] `scripts/eval.py` 或扩展 `dexjoco-lerobot-eval` 加载 DexQuery checkpoint
- [ ] 配置：`configs/rand_obj/bimanual_assembly.yaml`，50 episodes，与 ACT/GR00T 一致
- [ ] 记录：success rate、phase 切换是否合理、outcome 预测曲线（debug）

---

## Step 7：Ablation（论文用，非 v1 阻塞）

- [ ] 无 outcome BCE（纯 BC + query）
- [ ] 无 subtask query（三图 concat，同 backbone）
- [ ] FiLM 全局语言调制 vs cross-attn query
- [ ] 换 action 骨干：DexQuery + Diffusion / DiT

---

## 依赖关系小结

```
Step 1 (contact 标注)
    ↓
Step 2 (models) + Step 3 (data)  可并行
    ↓
Step 4 (train)
    ↓
Step 5 (inference) → Step 6 (eval)
```

---

## 明确不做（v1）

- 修改 `/home/wangrenpeng/lerobot` 源码或注册 `factory.py`
- GroundingDINO / bbox / crop 主路径
- Test-time VLM 分解子任务
- 保证 OOD 复杂 recovery（主要靠 outcome + rand-obj 分布内 re-grasp）
