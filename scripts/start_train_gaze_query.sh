#!/usr/bin/env bash
# Train GazeQueryNet (ViT + 2 learnable queries, RF-DETR-style spatial head).
set -euo pipefail
source /home/wangrenpeng/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH=/home/wangrenpeng/dexjoco
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

DATA=/mnt/hdd/dexjoco/datasets/gaze_spiral_ego_100
OUT=/mnt/hdd/dexjoco/outputs/gaze_query_train_vit
mkdir -p "$OUT"
PY=/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python
cd /home/wangrenpeng/dexjoco

exec "$PY" -u -m gaze_heatmap.train_query \
  --data-root "$DATA" \
  --out-dir "$OUT" \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --image-size 224 \
  --sigma 5.0 \
  --val-ratio 0.1 \
  --num-workers 4 \
  --device cuda:0 \
  2>&1 | tee "$OUT/train.log"
