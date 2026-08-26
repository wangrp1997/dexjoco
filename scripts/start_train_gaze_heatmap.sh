#!/usr/bin/env bash
set -euo pipefail
DATA=/mnt/hdd/dexjoco/datasets/gaze_spiral_ego_100
OUT=/mnt/hdd/dexjoco/outputs/gaze_heatmap_train
LOG="$OUT/train.log"
mkdir -p "$OUT"
export PYTHONPATH=/home/wangrenpeng/dexjoco
export CUDA_VISIBLE_DEVICES=1
PY=/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python
cd /home/wangrenpeng/dexjoco
exec "$PY" -u -m gaze_heatmap.train \
  --data-root "$DATA" \
  --out-dir "$OUT" \
  --epochs 30 \
  --batch-size 16 \
  --device cuda:0 \
  2>&1 | tee -a "$LOG"
