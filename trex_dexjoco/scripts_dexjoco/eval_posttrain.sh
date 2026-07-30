#!/bin/bash
# DexJoCo sim eval for T-Rex post-train (output layout = pi0.5 / ForceVLA).
# Example result dir:
#   outputs/trex/bimanual_assembly_seed0_ckpt000013/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TREX_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${TREX_ROOT}:${REPO_ROOT}:${REPO_ROOT}/dexjoco:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

CHECKPOINT="${CHECKPOINT:-/mnt/hdd/checkpoints/trex_dexjoco_ckpt/bimanual_assembly/trex_posttrain_bimanual_assembly/trex_posttrain_bimanual_assembly_0728_2128/checkpoint-13-44646}"
CONFIG="${CONFIG:-./configs/rand_obj/bimanual_assembly.yaml}"
SEED="${SEED:-0}"
EPISODES="${EPISODES:-50}"
CUDA_ID="${CUDA_ID:-0}"

EXTRA=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRA+=(--overwrite)
fi
if [[ "${HYBRID_INSERT:-0}" == "1" ]]; then
  EXTRA+=(--hybrid-insert)
fi
if [[ "${SKILL_GRAPH:-0}" == "1" ]]; then
  EXTRA+=(--skill-graph-recovery)
fi

python -m eval_sim.evaluate \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --seed "${SEED}" \
  --episodes "${EPISODES}" \
  --cuda "${CUDA_ID}" \
  "${EXTRA[@]}"
