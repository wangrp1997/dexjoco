#!/usr/bin/env bash
# Batch gaze spiral labels: ego JPEG + parquet, no video. Uses idle GPU for EGL.
set -euo pipefail
OUT=/mnt/hdd/dexjoco/datasets/gaze_spiral_ego_100
LOG="$OUT/batch.log"
mkdir -p "$OUT"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco:/home/wangrenpeng/dexjoco/scripts:/home/wangrenpeng/lai
PY=/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python
cd /home/wangrenpeng/dexjoco
exec "$PY" -u scripts/batch_gaze_spiral_collect.py --out-dir "$OUT" 2>&1 | tee -a "$LOG"
