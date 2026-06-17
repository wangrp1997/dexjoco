# pi0.5 + ForceVLA（DexJoCo）

在 **pi0.5 LoRA** 上增加 sim 特权 **力/触觉 proxy**，与纯视觉 `bimanual_assembly` 同骨干，做公平消融。

代码：`src/openpi/forcevla/`（模型来源见各文件首行 `Source:` 注释）。

## 0. 环境

```bash
cd ~/dexjoco/openpi
conda activate openpi
pip install -e ~/dexjoco/refs/ForceVLA/flaxformer   # LIMoE 依赖，装一次即可
```

确认 `config.yaml` 里 `dataset_root` 指向真实本地路径（例如 `/mnt/ssd/datasets/dexjoco_lerobot_datasets`）。

## 1. 力标签（DexJoCo 仓库，非 openpi）

在 **lerobot** 环境 replay 写 sidecar（不改 LeRobot 本体）：

```bash
cd ~/dexjoco
conda activate lerobot
export PYTHONPATH=/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco
export MUJOCO_GL=egl

python -u dexquery/scripts/label_forces.py \
  --task bimanual_assembly \
  --dataset-root /mnt/ssd/datasets/dexjoco_lerobot_datasets \
  --zarr-input-dir /mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly \
  --overwrite
```

输出：`<dataset>/bimanual_assembly/force_labels/forces.parquet`  
字段：`wrist_ft_right(6)`、`wrist_ft_left(6)`、`right_finger_force(12)`、`left_finger_force(12)`，按 `index` 与 LeRobot 对齐。

## 2. 归一化统计

纯视觉 baseline（46 维 state，与 `bimanual_assembly` 一致）：

```bash
cd ~/dexjoco/openpi
conda activate openpi

python scripts/compute_norm_stats.py bimanual_assembly --batch-size=64 --num-workers=16
```

ForceVLA 需先把 46 维 state 转成 44 维 proprio（quat→rotvec），并对 **state / actions / force** 一起做 quantile 归一化（对齐原版 ForceVLA 对 wrench+proprio 的 norm 方式）。**每个力输入模式单独算一份 stats**（维度不同）：

```bash
# 训哪个模式就跑哪个（both 示例）
python scripts/compute_norm_stats.py bimanual_assembly_forcevla_both --batch-size=64 --num-workers=16

# wrist / finger 若也要训，分别再跑：
# python scripts/compute_norm_stats.py bimanual_assembly_forcevla_wrist  ...
# python scripts/compute_norm_stats.py bimanual_assembly_forcevla_finger ...
```

输出目录（含 `state`、`actions`、`force` 三项）：

| 配置 | norm stats 路径 |
|------|-----------------|
| `*_forcevla_wrist` | `assets/bimanual_assembly/forcevla_wrist/local_repo/` |
| `*_forcevla_finger` | `assets/bimanual_assembly/forcevla_finger/local_repo/` |
| `*_forcevla_both` | `assets/bimanual_assembly/forcevla_both/local_repo/` |

## 3. 训练配置（消融）

| 配置名 | 力输入 |
|--------|--------|
| `bimanual_assembly` | 无（**pi0.5 纯视觉 baseline**） |
| `bimanual_assembly_forcevla_wrist` | 双腕 6D×2 = 12D |
| `bimanual_assembly_forcevla_finger` | 指尖 12D×2 = 24D |
| `bimanual_assembly_forcevla_both` | 腕 + 指尖 = 36D |

```bash
cd ~/dexjoco/openpi
conda activate openpi
export CUDA_VISIBLE_DEVICES=1   # 按需选空闲卡

# baseline（若尚未训过）
python scripts/train.py bimanual_assembly --exp-name=pi05_vision --project-name=dexjoco --overwrite

# 消融（默认 exp_name 已按力输入区分，可直接跑）
python scripts/train.py bimanual_assembly_forcevla_wrist  --project-name=dexjoco --overwrite
python scripts/train.py bimanual_assembly_forcevla_finger --project-name=dexjoco --overwrite
python scripts/train.py bimanual_assembly_forcevla_both   --project-name=dexjoco --overwrite
```

**W&B**：`config.yaml` 里 `wandb_enabled: true` 时会自动记录；默认项目是 `openpi`，与 ACT/Diffusion 等同表对比请加上 `--project-name=dexjoco`（run 名为 `exp_name`，如 `forcevla_both`）。

Checkpoint 目录（均在 SSD，与 `dexquery_dexjoco_ckpt` 等同层）：

| 配置 | 保存路径 |
|------|----------|
| `bimanual_assembly` | `/mnt/ssd/checkpoints/pi05_dexjoco_ckpt/bimanual_assembly/pi05_vision/<step>/` |
| `bimanual_assembly_forcevla_wrist` | `/mnt/ssd/checkpoints/forcevla_dexjoco_ckpt/bimanual_assembly/forcevla_wrist/<step>/` |
| `bimanual_assembly_forcevla_finger` | `/mnt/ssd/checkpoints/forcevla_dexjoco_ckpt/bimanual_assembly/forcevla_finger/<step>/` |
| `bimanual_assembly_forcevla_both` | `/mnt/ssd/checkpoints/forcevla_dexjoco_ckpt/bimanual_assembly/forcevla_both/<step>/` |

路径由 `config.yaml` 的 `ckpts_root` / `forcevla_ckpts_root` 控制。

## 4. 评估

纯视觉与力模型均用现有 DexJoCo eval 流程（`dexjoco-openpi-eval` / `serve_policy.py`）。  
**力模型推理**需在 sim 中提供与训练一致的力输入（腕部 sensor）；eval 侧接线若未完成，先以训练 loss / 少量 sim rollout 验证。

## 5. 建议实验顺序

1. `force_labels` 全量完成  
2. `bimanual_assembly` baseline 训完并 eval  
3. `forcevla_wrist` → 有增益再加 `finger` / `both`  
4. 记录同一 seed、同一 eval 协议下的成功率对比
