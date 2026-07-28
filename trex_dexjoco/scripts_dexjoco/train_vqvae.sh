#!/bin/bash
# Step 1: prepare DexJoCo force → VQ-VAE hdf5 tree + train VQ-VAE (T1).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

FORCE_PARQUET="${FORCE_PARQUET:-/mnt/ssd/datasets/dexjoco_lerobot_datasets/bimanual_assembly/force_labels/forces.parquet}"
VQVAE_DATA="${VQVAE_DATA:-/mnt/ssd/datasets/trex_dexjoco/vqvae_f6_data}"
VQVAE_OUT="${VQVAE_OUT:-/mnt/hdd/checkpoints/trex/vqvae_dexjoco_allegro8x3}"

python -m adapters.prep_vqvae_data \
  --force_parquet "${FORCE_PARQUET}" \
  --output_root "${VQVAE_DATA}" \
  --task_name dexjoco_assembly

# DATA_ROOT must contain pretrain_manifest.json (single-task layout)
accelerate launch -m tactile_vqvae.train \
  --data_root "${VQVAE_DATA}/dexjoco_assembly" \
  --output_dir "${VQVAE_OUT}" \
  --window 16 \
  --stride 4 \
  --codebook_size 64 \
  --granularity finger \
  --n_fingers 4 \
  --per_finger_dim 3 \
  --epochs 30 \
  --batch_size 256 \
  --lr 3e-4

echo "VQ-VAE ckpt: ${VQVAE_OUT}/latest.pt"
