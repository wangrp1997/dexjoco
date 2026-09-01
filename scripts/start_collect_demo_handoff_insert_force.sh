#!/usr/bin/env bash
# Batch: original 100 human demos -> handoff -> demo open-loop insert + force npz/mp4.
set -euo pipefail
OUT=/mnt/hdd/dexjoco/outputs/pi05_hybrid_insert_collect_raw_force
LOG="$OUT/collect.log"
mkdir -p "$OUT"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco:/home/wangrenpeng/dexjoco/scripts:/home/wangrenpeng/reach_insert_rl
PY=/home/wangrenpeng/miniconda3/envs/dexjoco/bin/python
cd /home/wangrenpeng/dexjoco

SESSION=demo_handoff_insert_force
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists; attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "$PY -u scripts/collect_demo_handoff_insert_force.py \
    --all \
    --seed 0 \
    --skip-existing \
    --output $OUT \
    2>&1 | tee -a $LOG"

echo "started tmux session: $SESSION"
echo "log: $LOG"
echo "attach: tmux attach -t $SESSION"
