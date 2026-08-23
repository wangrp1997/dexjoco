#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENPI_DIR="$REPO_ROOT/openpi"
DEXJOCO_DIR="$REPO_ROOT/dexjoco"
PORT="${PORT:-8011}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
SYNTHETIC_HANDOFF_FRAME="${SYNTHETIC_HANDOFF_FRAME:-}"
OUTPUT="${OUTPUT:-$DEXJOCO_DIR/outputs/retrieval_cerebellum/explicit_handoff_smoke_20260823}"
LOG_DIR="${LOG_DIR:-${OUTPUT}_logs}"
SERVER_LOG="$LOG_DIR/pi05_server.log"
EVAL_LOG="$LOG_DIR/eval.log"
SERVER_PID=""

mkdir -p "$LOG_DIR"

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

echo "Starting π0.5 server on GPU $GPU, port $PORT"
setsid bash -lc "
    cd '$OPENPI_DIR'
    export CUDA_VISIBLE_DEVICES='$GPU'
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export PYTHONUNBUFFERED=1
    exec conda run --no-capture-output -n openpi python scripts/serve_policy.py \\
        --port '$PORT' \\
        policy:checkpoint \\
        --policy.config bimanual_assembly \\
        --policy.dir ../checkpoints/pi05_dexjoco_ckpt/bimanual_assembly
" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: policy server exited during startup" >&2
        tail -80 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        break
    fi
    sleep 1
done

if ! ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "ERROR: policy server did not become ready" >&2
    tail -80 "$SERVER_LOG" >&2 || true
    exit 1
fi

echo "Running one explicit-handoff sensor-only episode"
cd "$DEXJOCO_DIR"
export MUJOCO_GL=egl
EVAL_EXTRA_ARGS=()
if [[ -n "$SYNTHETIC_HANDOFF_FRAME" ]]; then
    EVAL_EXTRA_ARGS+=(
        --retrieval-cerebellum-synthetic-handoff-frame
        "$SYNTHETIC_HANDOFF_FRAME"
    )
fi
timeout --signal=INT --kill-after=30s 900s \
    conda run --no-capture-output -n dexjoco dexjoco-openpi-eval \
        --config ../configs/rand_obj/bimanual_assembly.yaml \
        --seed "$SEED" \
        --port "$PORT" \
        --host 127.0.0.1 \
        --output "$OUTPUT" \
        --episodes 1 \
        --render-mode rgb_array \
        --retrieval-cerebellum-intent-chunk \
        --overwrite \
        "${EVAL_EXTRA_ARGS[@]}" \
    2>&1 | tee "$EVAL_LOG"
