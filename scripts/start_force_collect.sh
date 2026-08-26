#!/bin/bash
set -euo pipefail
OUT=/mnt/hdd/dexjoco/outputs/pi05_hybrid_insert_collect_raw_force
CKPT=/mnt/hdd/dexjoco/shared_checkpoints/pi05_dexjoco_ckpt/bimanual_assembly
PORT=8012
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/wangrenpeng/dexjoco:/home/wangrenpeng/dexjoco/dexjoco
cd /home/wangrenpeng/dexjoco/openpi
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
CUDA_VISIBLE_DEVICES=1 \
/home/wangrenpeng/miniconda3/envs/openpi/bin/python scripts/serve_policy.py --port "$PORT" policy:checkpoint \
  --policy.config bimanual_assembly \
  --policy.dir "$CKPT" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
for i in $(seq 1 60); do
  if rg -q "server listening" "$OUT/server.log" 2>/dev/null; then
    echo server_ready
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo server_died
    tail -40 "$OUT/server.log"
    exit 1
  fi
  sleep 2
done
source /home/wangrenpeng/miniconda3/etc/profile.d/conda.sh
conda activate dexjoco
cd /home/wangrenpeng/dexjoco
python scripts/collect_pi05_hybrid_insert_pilot.py \
  --host 127.0.0.1 --port "$PORT" \
  --start-index 0 \
  --seed-base 251400 \
  --max-rollouts 2500 \
  --target-successes 800 \
  --grasp-deadline 700 \
  --max-policy-steps 1200 \
  --output "$OUT" \
  2>&1 | tee -a "$OUT/collect.log"
echo "[collect] finished" | tee -a "$OUT/collect.log"
