#!/usr/bin/env bash
# Poll action44 checkpoints and eval ep35 until insert_ok.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="/mnt/hdd/dexjoco/poseinsert_sim/checkpoints/action44"
VIDEO_DIR="/mnt/hdd/dexjoco/outputs/poseinsert_sim/videos"
LOG="/mnt/hdd/dexjoco/outputs/poseinsert_sim/overnight_eval.log"
mkdir -p "$VIDEO_DIR"
cd "$ROOT"

last_ckpt=""
while true; do
  ckpt="$(ls -t "$CKPT_DIR"/policy_epoch_*.ckpt 2>/dev/null | head -1 || true)"
  if [[ -z "$ckpt" || "$ckpt" == "$last_ckpt" ]]; then
    sleep 120
    continue
  fi
  last_ckpt="$ckpt"
  tag="$(basename "$ckpt" .ckpt)"
  out="$VIDEO_DIR/ep35_${tag}_seed0.mp4"
  echo "$(date -Is) eval $ckpt" | tee -a "$LOG"
  if python scripts/eval_pose_insert_sim.py --ep 35 --insert-mode action44 --ckpt "$ckpt" \
    --video --video-out "$out" 2>&1 | tee -a "$LOG"; then
    echo "$(date -Is) SUCCESS $out" | tee -a "$LOG"
    exit 0
  fi
  sleep 60
done
