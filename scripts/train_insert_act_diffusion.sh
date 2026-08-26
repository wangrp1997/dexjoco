#!/bin/bash
# Parallel ACT vs Diffusion BC on 800 insert episodes (vision+state, no force).
# Usage: bash scripts/train_insert_act_diffusion.sh [act|diffusion|both]
set -euo pipefail

POLICY="${1:-both}"
DATASET_ROOT=/mnt/hdd/dexjoco/datasets/bimanual_assembly_insert_force_lerobot
REPO_ID=bimanual_assembly_insert_force_lerobot
OUT_ROOT=/mnt/hdd/dexjoco/outputs/insert_bc_act_diffusion
STEPS=50000
BATCH=4
SEED=0
SAVE_FREQ=10000
LOG_FREQ=100

source /home/wangrenpeng/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$OUT_ROOT"

launch_one() {
  local policy="$1"
  local gpu="$2"
  local out="$OUT_ROOT/${policy}_steps${STEPS}_bs${BATCH}_seed${SEED}"
  mkdir -p "$out"
  echo "[train] policy=$policy gpu=$gpu out=$out"
  CUDA_VISIBLE_DEVICES="$gpu" lerobot-train \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --policy.type="$policy" \
    --output_dir="$out" \
    --job_name="insert_${policy}" \
    --policy.device=cuda \
    --policy.repo_id="local/insert_${policy}" \
    --batch_size="$BATCH" \
    --steps="$STEPS" \
    --seed="$SEED" \
    --num_workers=4 \
    --log_freq="$LOG_FREQ" \
    --save_freq="$SAVE_FREQ" \
    --save_checkpoint=true \
    --use_policy_training_preset=true \
    --wandb.enable=false \
    2>&1 | tee "$out/train.log"
}

case "$POLICY" in
  act)
    launch_one act 2
    ;;
  diffusion)
    launch_one diffusion 1
    ;;
  both)
    # ACT on GPU2, Diffusion on GPU1 — matched data/steps/batch/seed
    CUDA_VISIBLE_DEVICES=2 lerobot-train \
      --dataset.repo_id="$REPO_ID" \
      --dataset.root="$DATASET_ROOT" \
      --policy.type=act \
      --output_dir="$OUT_ROOT/act_steps${STEPS}_bs${BATCH}_seed${SEED}" \
      --job_name=insert_act \
      --policy.device=cuda \
      --policy.repo_id=local/insert_act \
      --batch_size="$BATCH" \
      --steps="$STEPS" \
      --seed="$SEED" \
      --num_workers=4 \
      --log_freq="$LOG_FREQ" \
      --save_freq="$SAVE_FREQ" \
      --save_checkpoint=true \
      --use_policy_training_preset=true \
      --wandb.enable=false \
      2>&1 | tee "$OUT_ROOT/act_steps${STEPS}_bs${BATCH}_seed${SEED}/train.log" &
    ACT_PID=$!
    mkdir -p "$OUT_ROOT/act_steps${STEPS}_bs${BATCH}_seed${SEED}"

    mkdir -p "$OUT_ROOT/diffusion_steps${STEPS}_bs${BATCH}_seed${SEED}"
    CUDA_VISIBLE_DEVICES=1 lerobot-train \
      --dataset.repo_id="$REPO_ID" \
      --dataset.root="$DATASET_ROOT" \
      --policy.type=diffusion \
      --output_dir="$OUT_ROOT/diffusion_steps${STEPS}_bs${BATCH}_seed${SEED}" \
      --job_name=insert_diffusion \
      --policy.device=cuda \
      --policy.repo_id=local/insert_diffusion \
      --batch_size="$BATCH" \
      --steps="$STEPS" \
      --seed="$SEED" \
      --num_workers=4 \
      --log_freq="$LOG_FREQ" \
      --save_freq="$SAVE_FREQ" \
      --save_checkpoint=true \
      --use_policy_training_preset=true \
      --wandb.enable=false \
      2>&1 | tee "$OUT_ROOT/diffusion_steps${STEPS}_bs${BATCH}_seed${SEED}/train.log" &
    DIFF_PID=$!

    echo "ACT_PID=$ACT_PID DIFF_PID=$DIFF_PID"
    wait $ACT_PID
    wait $DIFF_PID
    ;;
  *)
    echo "usage: $0 [act|diffusion|both]"
    exit 1
    ;;
esac
