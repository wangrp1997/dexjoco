#!/usr/bin/env bash
# Restart action44 holdout + batch eval after PoseInsert path fix.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="/mnt/hdd/dexjoco/poseinsert_sim/checkpoints/action44"
OUT="/mnt/hdd/dexjoco/outputs/poseinsert_sim"
mkdir -p "$OUT"

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/dexjoco"
export MUJOCO_GL=egl

PYTHON="${PYTHON:-python}"
if command -v conda >/dev/null 2>&1; then
  PYTHON="conda run -n dexjoco --no-capture-output python"
fi

HOLDOUT_LOG="$OUT/holdout_eval_restart.log"
BATCH_LOG="$OUT/batch_eval_action44_restart.log"

echo "$(date -Is) holdout restart (epoch 500 + last)" | tee "$HOLDOUT_LOG"
for ckpt in policy_epoch_500_seed_233.ckpt policy_last.ckpt; do
  path="$CKPT_DIR/$ckpt"
  [[ -f "$path" ]] || continue
  echo "=== $ckpt ===" | tee -a "$HOLDOUT_LOG"
  $PYTHON scripts/validate_action44_holdout.py --ckpt "$path" 2>&1 | tee -a "$HOLDOUT_LOG"
done

echo "$(date -Is) batch eval restart (all ckpts)" | tee "$BATCH_LOG"
$PYTHON scripts/batch_eval_action44.py \
  --ckpt-dir "$CKPT_DIR" \
  --ckpts all \
  --out "/mnt/hdd/dexjoco/poseinsert_sim/bimanual_assembly/batch_eval_action44.json" \
  2>&1 | tee -a "$BATCH_LOG"
