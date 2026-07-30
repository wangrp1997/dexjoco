#!/bin/bash
# DexJoCo post-train (B: action 44 + T1: native tactile VQ-VAE).
# Short fine-tune: few epochs, save every epoch, lower LR (anti-overfit).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}/scripts"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

ORIGIN_MODEL_PATH="${ORIGIN_MODEL_PATH:-/mnt/hdd/checkpoints/trex/Qwen3-VL-2B-Instruct}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/mnt/hdd/checkpoints/trex/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6}"
# REQUIRED: DexJoCo-retrained VQ-VAE (not midtrain F6)
VQVAE_CKPT="${VQVAE_CKPT:-/mnt/hdd/checkpoints/trex/vqvae_dexjoco_allegro8x3/vqvae_f6_20260727_231133/latest.pt}"

LEROBOT_ROOT="${LEROBOT_ROOT:-/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly}"
FORCE_LABELS="${FORCE_LABELS:-${LEROBOT_ROOT}/force_labels/forces.parquet}"
NORM_STATS="${NORM_STATS:-/mnt/ssd/datasets/trex_dexjoco/bimanual_assembly/trex_norm_stats.json}"

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/hdd/checkpoints/trex_dexjoco_ckpt/bimanual_assembly}"
EXPERIMENT_NAME="trex_posttrain_bimanual_assembly"
RUN_NAME="${EXPERIMENT_NAME}_$(date +%m%d_%H%M)"

ACTION_DIM=44
NUM_PROCESSES="${NUM_PROCESSES:-1}"
TRAIN_BSZ="${TRAIN_BSZ:-16}"
# Lower LR + short schedule: val peaked ~epoch0–1 under 1e-4×100ep.
LR="${LR:-3e-5}"
N_EPOCHS="${N_EPOCHS:-10}"
# Few epochs → save every epoch so we can pick best by val.
SAVE_FREQ="${SAVE_FREQ:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

NUM_WORKERS="${NUM_WORKERS:-4}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

if [[ ! -f "${NORM_STATS}" ]]; then
  echo "Missing ${NORM_STATS}; run: python -m adapters.compute_norm_stats"
  exit 1
fi
if [[ ! -f "${VQVAE_CKPT}" ]]; then
  echo "Missing ${VQVAE_CKPT}; run scripts_dexjoco/train_vqvae.sh first"
  exit 1
fi

ACCEL_CONFIG="${ACCEL_CONFIG:-../config/sft_qwen_single.yaml}"

echo "post-train: epochs=${N_EPOCHS} save_freq=${SAVE_FREQ} lr=${LR} wd=${WEIGHT_DECAY} run=${RUN_NAME}"

accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  train.py \
  --model_path "${ORIGIN_MODEL_PATH}" \
  --data_format dexjoco \
  --lerobot_root "${LEROBOT_ROOT}" \
  --force_labels_path "${FORCE_LABELS}" \
  --norm_stats_path "${NORM_STATS}" \
  --n_epochs "${N_EPOCHS}" \
  --save_freq "${SAVE_FREQ}" \
  --action_dim "${ACTION_DIM}" \
  --action_chunk 16 \
  --train_bsz_per_gpu "${TRAIN_BSZ}" \
  --learning_rate "${LR}" \
  --min_lr_ratio 0 \
  --weight_decay "${WEIGHT_DECAY}" \
  --gradient_accumulation_steps 1 \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${OUTPUT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --run_name "${RUN_NAME}" \
  --use_robot_state 0 \
  --use_tactile_vec 1 \
  --use_tactile_deform 0 \
  --use_tactile_vqvae 1 \
  --vqvae_ckpt "${VQVAE_CKPT}" \
  --tactile_intermediate_size 1536 \
  --training_stage 2 \
  --tactile_loss_weight 1.0 \
  --cascaded_total_steps 10 \
  --cascaded_split_step 6 \
  --cascaded_tactile_dropout 0.1 \
  --cascaded_loss_weight 1.0 \
  --resume_checkpoint "${RESUME_CHECKPOINT}" \
  --resume_source midtrain \
  --use_flare 1 \
  --n_flare_tokens_per_frame 4 \
  --n_flare_steps 8 \
  --flare_loss_weight 0.5 \
  --flare_frame_stride 4 \
  --flare_layer_index -1 \
  --image_size 384 288 \
  --num_workers "${NUM_WORKERS}" \
  --video_backend "${VIDEO_BACKEND}" \
  --val_ratio 0.05 \
  --val_freq 500 \
  --max_val_batches 30 \
  --save_best 1
