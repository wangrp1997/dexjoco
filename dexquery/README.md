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

推理用 **预测的** `tray_ok/peg_ok` + phase 切换（见 `inference/phase_controller.py`）：

```bash
python -u dexquery/scripts/eval.py \
  --task bimanual_assembly \
  --checkpoint /mnt/ssd/checkpoints/dexquery_dexjoco_ckpt/bimanual_assembly/checkpoint_last.pt \
  --episodes 50 \
  --seed 0 \
  --output-dir outputs/dexquery/bimanual_assembly_seed0
```

- 输出：`eval_summary.json`（成功率）、`phase_traces.json`（每步 phase / `p_tray` / `p_peg`）
- phase 滞后阈值：`configs/bimanual_assembly.yaml` → `inference.phase_controller`
