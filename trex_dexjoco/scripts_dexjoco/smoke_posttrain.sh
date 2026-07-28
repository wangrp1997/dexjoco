#!/bin/bash
# Smoke test: load midtrain + DexJoCo data + run 3 train steps.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}/scripts"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

accelerate launch \
  --config_file ../config/sft_qwen_single.yaml \
  --num_processes 1 \
  train.py \
  --model_path /mnt/hdd/checkpoints/trex/Qwen3-VL-2B-Instruct \
  --data_format dexjoco \
  --lerobot_root /mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly \
  --force_labels_path /mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly/force_labels/forces.parquet \
  --norm_stats_path /mnt/ssd/datasets/trex_dexjoco/bimanual_assembly/trex_norm_stats.json \
  --n_epochs 1 --max_train_steps 3 --save_freq 999 \
  --action_dim 44 --action_chunk 16 \
  --train_bsz_per_gpu 1 --learning_rate 1e-4 \
  --min_lr_ratio 0 --weight_decay 0 --gradient_accumulation_steps 1 \
  --output_dir /tmp/trex_smoke --log_dir /tmp/trex_smoke \
  --experiment_name smoke --run_name smoke_ok \
  --use_robot_state 0 --use_tactile_vec 1 --use_tactile_deform 0 \
  --use_tactile_vqvae 1 \
  --vqvae_ckpt /mnt/hdd/checkpoints/trex/vqvae_dexjoco_allegro8x3/latest.pt \
  --tactile_intermediate_size 1536 --training_stage 2 \
  --tactile_loss_weight 1.0 \
  --cascaded_total_steps 10 --cascaded_split_step 6 \
  --cascaded_tactile_dropout 0.1 --cascaded_loss_weight 1.0 \
  --resume_checkpoint /mnt/hdd/checkpoints/trex/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6 \
  --resume_source midtrain --use_flare 0 \
  --image_size 384 288 --val_ratio 0

echo "SMOKE OK"
