#!/usr/bin/env bash
# Sync refs → interaction_retarget mirror tree (same subdir names).
# Re-run after updating refs/ clones.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REFS="$ROOT/refs"
IR="$ROOT/interaction_retarget"

mkdir -p "$IR/DexGraspBench/src/task/eval_func/fc_metric"
mkdir -p "$IR/DexGraspBench/src/util"
mkdir -p "$IR/Dexonomy/dexonomy/sim"
mkdir -p "$IR/Dexonomy/dexonomy/config/op"
mkdir -p "$IR/Dexonomy/dexonomy/op"
mkdir -p "$IR/holosoma/holosoma_retargeting/src"
mkdir -p "$IR/GenHand/simulation"
mkdir -p "$IR/contactopt/contactopt"
mkdir -p "$IR/GraspTTA/utils"
mkdir -p "$IR/spider/spider/preprocess"

# DexGraspBench — verbatim math + eval stages
cp "$REFS/DexGraspBench/src/task/eval_func/fc_metric/qp.py" \
   "$IR/DexGraspBench/src/task/eval_func/fc_metric/"
cp "$REFS/DexGraspBench/src/util/rot_util.py" "$IR/DexGraspBench/src/util/"
cp "$REFS/DexGraspBench/src/task/eval_func/tabletop_mocap.py" \
   "$IR/DexGraspBench/src/task/eval_func/"
cp "$REFS/DexGraspBench/src/task/eval_func/fc_mocap.py" \
   "$IR/DexGraspBench/src/task/eval_func/"
cp "$REFS/DexGraspBench/src/util/hand_util.py" "$IR/DexGraspBench/src/util/"
# GenHand contact+FC optimisation (reference only; PyTorch — adapter in grasp/qpos_refine.py)
mkdir -p "$IR/GenHand/optimisation"
cp "$REFS/GenHand/optimisation/loss.py" "$IR/GenHand/optimisation/" 2>/dev/null || true
cp "$REFS/GenHand/optimisation/icp.py" "$IR/GenHand/optimisation/" 2>/dev/null || true

# Dexonomy — filter config + reference impl
cp "$REFS/Dexonomy/dexonomy/config/op/grasp.yaml" "$IR/Dexonomy/dexonomy/config/op/"
cp "$REFS/Dexonomy/dexonomy/op/gen_grasp.py" "$IR/Dexonomy/dexonomy/op/"

# GenHand / spider — reference copies (PyBullet / MANO deps; use via adapters)
cp "$REFS/GenHand/simulation/robot_base.py" "$IR/GenHand/simulation/"
cp "$REFS/GenHand/simulation/trajectory.py" "$IR/GenHand/simulation/"
cp "$REFS/spider/spider/preprocess/detect_contact.py" "$IR/spider/spider/preprocess/"

# contactopt / GraspTTA — keep numpy ports in mirror (do not overwrite with torch originals)
# holosoma laplacian — maintained in holosoma/.../laplacian_utils.py (numpy subset)

# Fix DexGraspBench qp imports after copy
sed -i 's|from util.rot_util import|from interaction_retarget.DexGraspBench.src.util.rot_util import|' \
  "$IR/DexGraspBench/src/task/eval_func/fc_metric/qp.py" || true
sed -i '/sys.path.append/d' "$IR/DexGraspBench/src/task/eval_func/fc_metric/qp.py" || true

echo "sync_refs: done (see interaction_retarget/REFS.md)"
