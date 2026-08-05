#!/bin/bash
# Open-loop action MSE on DexJoCo demos (multi-ckpt).
# Saves under outputs/trex/{env}_seed{seed}_{ckpt_label}_openloop/ (no videos).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TREX_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${TREX_ROOT}"

export PYTHONPATH="${TREX_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

CKPT_ROOT="${CKPT_ROOT:-/mnt/hdd/checkpoints/trex_dexjoco_ckpt/bimanual_assembly/trex_posttrain_bimanual_assembly/trex_posttrain_bimanual_assembly_0730_1103}"
CHECKPOINTS="${CHECKPOINTS:-${CKPT_ROOT}/checkpoint-best ${CKPT_ROOT}/checkpoint-4-15945}"
CUDA_ID="${CUDA_ID:-0}"
SEED="${SEED:-0}"
N_TRAIN="${N_TRAIN:-64}"
N_VAL="${N_VAL:-64}"
# Optional override parent; leave empty → default outputs/trex/..._openloop/
DUMP_DIR="${DUMP_DIR:-}"

EXTRA=()
if [[ -n "${DUMP_DIR}" ]]; then
  EXTRA+=(--dump_dir "${DUMP_DIR}")
fi

# shellcheck disable=SC2086
python -m eval_sim.openloop \
  --checkpoints ${CHECKPOINTS} \
  --cuda "${CUDA_ID}" \
  --seed "${SEED}" \
  --n_train "${N_TRAIN}" \
  --n_val "${N_VAL}" \
  "${EXTRA[@]}"
