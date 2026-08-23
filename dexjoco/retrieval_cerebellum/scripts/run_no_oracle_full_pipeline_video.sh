#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENPI_DIR="$REPO_ROOT/openpi"
DEXJOCO_DIR="$REPO_ROOT/dexjoco"
PORT="${PORT:-8011}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-17 23 31 42 78 81 83 88 97}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEXJOCO_DIR/outputs/retrieval_cerebellum/no_oracle_full_pipeline_20260823}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}_logs}"
SERVER_LOG="$LOG_DIR/server.log"
SERVER_PID=""

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        echo "ERROR: policy server port $PORT is still listening" >&2
        exit 1
    fi
    exit "$exit_code"
}

trap cleanup EXIT INT TERM

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "ERROR: port $PORT is already in use" >&2
    exit 1
fi

setsid bash -lc "
    cd '$OPENPI_DIR'
    export CUDA_VISIBLE_DEVICES='$GPU'
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export PYTHONUNBUFFERED=1
    exec conda run --no-capture-output -n openpi python scripts/serve_policy.py \
        --port '$PORT' \
        policy:checkpoint \
        --policy.config bimanual_assembly \
        --policy.dir ../checkpoints/pi05_dexjoco_ckpt/bimanual_assembly
" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        tail -80 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        break
    fi
    sleep 1
done

if ! ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    tail -80 "$SERVER_LOG" >&2 || true
    exit 1
fi

cd "$DEXJOCO_DIR"
export MUJOCO_GL=egl
export PYTHONPATH="$REPO_ROOT:$DEXJOCO_DIR"

successful_seed=""
for seed in $SEEDS; do
    output="$OUTPUT_ROOT/seed_$seed"
    rm -rf "$output"
    timeout --signal=INT --kill-after=30s 900s \
        conda run --no-capture-output -n dexjoco dexjoco-openpi-eval \
            --config ../configs/rand_obj/bimanual_assembly.yaml \
            --seed "$seed" \
            --port "$PORT" \
            --host 127.0.0.1 \
            --output "$output" \
            --episodes 1 \
            --render-mode rgb_array \
            --retrieval-cerebellum-intent-chunk \
            --retrieval-cerebellum-auto-handoff \
            --retrieval-cerebellum-contact-response \
            --overwrite \
        2>&1 | tee "$LOG_DIR/seed_$seed.log"
    if [[ -f "$output/success_rate_1_1.txt" ]]; then
        audit="$output/episode_00_success/retrieval_cerebellum_intent_chunk.json"
        if [[ -f "$audit" ]] && conda run --no-capture-output -n dexjoco python - "$audit" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if audit.get("deployable_handoff_observed") is True else 1)
PY
        then
            successful_seed="$seed"
            break
        fi
        echo "seed $seed succeeded without deployable cerebellum ownership; continuing"
    fi
done

if [[ -z "$successful_seed" ]]; then
    echo "No successful full pipeline rollout found" >&2
    exit 2
fi

episode_dir="$OUTPUT_ROOT/seed_$successful_seed/episode_00_success"
audit="$episode_dir/retrieval_cerebellum_intent_chunk.json"
video="$episode_dir/ego.mp4"
if [[ ! -f "$audit" || ! -f "$video" ]]; then
    echo "ERROR: successful rollout lacks audit or ego video" >&2
    exit 1
fi

conda run --no-capture-output -n dexjoco python - "$audit" "$OUTPUT_ROOT" "$successful_seed" <<'PY'
import json
import shutil
import sys
from pathlib import Path

audit_path = Path(sys.argv[1])
output = Path(sys.argv[2])
seed = int(sys.argv[3])
audit = json.loads(audit_path.read_text())
if audit.get("privileged_evaluator_enabled") is not False:
    raise SystemExit("privileged evaluator was enabled")
if audit.get("deployable_handoff_observed") is not True:
    raise SystemExit("deployable auto handoff was not observed")
if audit.get("synthetic_handoff_observed") is not False:
    raise SystemExit("synthetic handoff contaminated the rollout")
source_video = audit_path.parent / "ego.mp4"
target_video = output / "full_pipeline_no_oracle.mp4"
shutil.copy2(source_video, target_video)
delivery = {
    "seed": seed,
    "video": str(target_video),
    "policy_handoff_observed": audit.get("policy_handoff_observed"),
    "deployable_handoff_observed": audit.get("deployable_handoff_observed"),
    "synthetic_handoff_observed": audit.get("synthetic_handoff_observed"),
    "privileged_evaluator_enabled": audit.get("privileged_evaluator_enabled"),
    "control_inputs": audit.get("control_inputs"),
    "events": audit.get("events"),
}
(output / "delivery_audit.json").write_text(
    json.dumps(delivery, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(delivery, ensure_ascii=False, indent=2))
PY
