# DexQuery

DexQuery：灵巧操作的**子任务语义查询**框架（subtask text query + multi-view cross-attn + per-object outcome）。

- 实现步骤：[`IMPLEMENTATION_STEPS.md`](IMPLEMENTATION_STEPS.md)
- 独立目录，**不修改**外部 LeRobot（`/home/wangrenpeng/lerobot`）
- 首任务：`bimanual_assembly`

## 环境

```bash
cd ~/dexjoco
export PYTHONPATH=/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco
conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com   # 国内下载 SigLIP 建议开启
```

首次训练需 `pip install sentencepiece`（SigLIP tokenizer 依赖）。

## Step 1：Contact 标注

写入 `<dataset>/dexquery_labels/`（sidecar，不改原 LeRobot 数据）：

```bash
MUJOCO_GL=egl python -u dexquery/scripts/label_contact.py \
  --task bimanual_assembly \
  --dataset-root /mnt/ssd/datasets/dexjoco_lerobot_datasets \
  --zarr-input-dir /mnt/ssd/datasets/dexjoco_raw/dexjoco_raw_datasets/bimanual_assembly
```

## Step 2：训练

```bash
export CUDA_VISIBLE_DEVICES=1   # 按需指定 GPU

python -u scripts/train_dexjoco_lerobot.py \
  --policy dexquery \
  --task bimanual_assembly \
  --dataset-root /mnt/ssd/datasets/dexjoco_lerobot_datasets \
  --output-dir /mnt/ssd/checkpoints/dexquery_dexjoco_ckpt/bimanual_assembly \
  --device cuda
```

- 默认 wandb project：`dexjoco`；关闭加 `--no-wandb`
- checkpoint 每 **10k** step 保存；输出目录含 `config.yaml`、`dataset_stats.json`
- 超参：`configs/bimanual_assembly.yaml` + `configs/training/dual_arm_baseline.yaml`

## Step 3：评估

与 ACT/GR00T 相同目录结构：`outputs/dexquery/<task>_seed<N>_ckptXXXXXX/episode_XX_{success|failure}/*.mp4`

```bash
export CUDA_VISIBLE_DEVICES=1   # 按需指定 GPU
export MUJOCO_GL=egl

python -u dexquery/scripts/eval.py \
  --task bimanual_assembly \
  --checkpoint /mnt/ssd/checkpoints/dexquery_dexjoco_ckpt/bimanual_assembly/checkpoint_step_060000.pt \
  --episodes 50 \
  --seed 0
```

- 默认输出：`outputs/dexquery/bimanual_assembly_seed0_ckpt060000/`（可用 `--output` 覆盖）
- 每个 episode 保存 `ego.mp4`、`wrist_left.mp4`、`wrist_right.mp4`
- 另含 `success_rate_*_*.txt`、`eval_summary.json`、`phase_traces.json`
- rand-obj 全随机：加 `--rand-full`；覆盖已有目录：加 `--overwrite`
