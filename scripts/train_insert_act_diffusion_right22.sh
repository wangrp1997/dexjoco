#!/bin/bash
# Parallel ACT vs Diffusion BC on right-arm-only (22-dim) insert actions.
# Usage:
#   bash scripts/train_insert_act_diffusion_right22.sh [act|diffusion|both]
#   bash scripts/train_insert_act_diffusion_right22.sh diffusion resume
set -euo pipefail

POLICY="${1:-both}"
RESUME="${2:-}"
DATASET_ROOT=/mnt/hdd/dexjoco/datasets/bimanual_assembly_insert_force_lerobot_right22
REPO_ID=bimanual_assembly_insert_force_lerobot_right22
OUT_ROOT=/mnt/hdd/dexjoco/outputs/insert_bc_act_diffusion_right22
LOG_DIR="$OUT_ROOT/_train_logs"
STEPS=50000
BATCH=4
SEED=0
SAVE_FREQ=10000
LOG_FREQ=100

if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  echo "missing dataset; run: python scripts/slice_insert_lerobot_right22.py"
  exit 1
fi

source /home/wangrenpeng/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$LOG_DIR"

launch_one() {
  local policy="$1"
  local gpu="$2"
  local out="$OUT_ROOT/${policy}_steps${STEPS}_bs${BATCH}_seed${SEED}"
  local resume_args=(--resume=false)
  if [[ "${RESUME:-}" == "resume" ]]; then
    local cfg="$out/checkpoints/last/pretrained_model/train_config.json"
    if [[ ! -f "$cfg" ]]; then
      echo "missing resume config: $cfg"
      exit 1
    fi
    resume_args=(--resume=true --config_path="$cfg")
  fi
  echo "[train right22] policy=$policy gpu=$gpu out=$out ${resume_args[*]}"
  CUDA_VISIBLE_DEVICES="$gpu" lerobot-train \
    "${resume_args[@]}" \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --policy.type="$policy" \
    --output_dir="$out" \
    --job_name="insert_right22_${policy}" \
    --policy.device=cuda \
    --policy.repo_id="local/insert_right22_${policy}" \
    --batch_size="$BATCH" \
    --steps="$STEPS" \
    --seed="$SEED" \
    --num_workers=4 \
    --log_freq="$LOG_FREQ" \
    --save_freq="$SAVE_FREQ" \
    --save_checkpoint=true \
    --use_policy_training_preset=true \
    --wandb.enable=false \
    2>&1 | tee -a "$LOG_DIR/${policy}.log"
}

case "$POLICY" in
  act)
    launch_one act 2
    ;;
  diffusion)
    launch_one diffusion 1
    ;;
  both)
    CUDA_VISIBLE_DEVICES=2 lerobot-train \
      --dataset.repo_id="$REPO_ID" \
      --dataset.root="$DATASET_ROOT" \
      --policy.type=act \
      --output_dir="$OUT_ROOT/act_steps${STEPS}_bs${BATCH}_seed${SEED}" \
      --job_name=insert_right22_act \
      --policy.device=cuda \
      --policy.repo_id=local/insert_right22_act \
      --batch_size="$BATCH" \
      --steps="$STEPS" \
      --seed="$SEED" \
      --num_workers=4 \
      --log_freq="$LOG_FREQ" \
      --save_freq="$SAVE_FREQ" \
      --save_checkpoint=true \
      --use_policy_training_preset=true \
      --wandb.enable=false \
      2>&1 | tee "$LOG_DIR/act.log" &
    ACT_PID=$!

    CUDA_VISIBLE_DEVICES=1 lerobot-train \
      --dataset.repo_id="$REPO_ID" \
      --dataset.root="$DATASET_ROOT" \
      --policy.type=diffusion \
      --output_dir="$OUT_ROOT/diffusion_steps${STEPS}_bs${BATCH}_seed${SEED}" \
      --job_name=insert_right22_diffusion \
      --policy.device=cuda \
      --policy.repo_id=local/insert_right22_diffusion \
      --batch_size="$BATCH" \
      --steps="$STEPS" \
      --seed="$SEED" \
      --num_workers=4 \
      --log_freq="$LOG_FREQ" \
      --save_freq="$SAVE_FREQ" \
      --save_checkpoint=true \
      --use_policy_training_preset=true \
      --wandb.enable=false \
      2>&1 | tee "$LOG_DIR/diffusion.log" &
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
