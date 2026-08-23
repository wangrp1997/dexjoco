#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENPI_DIR="$REPO_ROOT/openpi"
DEXJOCO_DIR="$REPO_ROOT/dexjoco"
PORT="${PORT:-8011}"
GPU="${GPU:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEXJOCO_DIR/outputs/retrieval_cerebellum/contact_response_ablation9_20260823}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}_logs}"
SERVER_LOG="$LOG_DIR/server.log"
SERVER_PID=""

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

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

run_variant() {
    local name=$1
    local contact_flag=$2
    timeout --signal=INT --kill-after=30s 1200s \
        conda run --no-capture-output -n dexjoco python \
            retrieval_cerebellum/scripts/run_natural_handoff_baseline.py \
            --output "$OUTPUT_DIR/$name.json" \
            --intent-mode online_pi05_chunk \
            --openpi-config "$REPO_ROOT/configs/multi_task/bimanual_assembly.yaml" \
            --host 127.0.0.1 \
            --port "$PORT" \
            --steps 100 \
            "$contact_flag" \
        2>&1 | tee "$LOG_DIR/$name.log"
}

run_variant without_contact_response --no-contact-response
run_variant with_contact_response --contact-response

conda run --no-capture-output -n dexjoco python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
without = json.loads((output / "without_contact_response.json").read_text())
with_response = json.loads((output / "with_contact_response.json").read_text())

def compact(payload):
    return {
        "successes": payload["num_successes"],
        "requested": payload["num_requested_held_out_episodes"],
        "episodes": [
            {
                "episode": row["episode"],
                "success": row["success"],
                "exit_reason": row["exit_reason"],
                "steps": row["steps_executed"],
                "peak_force_n": row["peak_force_n"],
                "contact_response_steps": row["contact_response_steps"],
                "maximum_contact_rotation_correction_rad": row[
                    "maximum_contact_rotation_correction_rad"
                ],
                "final_lateral_error_m": row["final_lateral_error_m"],
                "final_axis_error_rad": row["final_axis_error_rad"],
            }
            for row in payload["results"]
        ],
    }

comparison = {
    "experiment": "frozen_coarse_alignment_contact_response_ablation",
    "without_contact_response": compact(without),
    "with_contact_response": compact(with_response),
}
(output / "comparison.json").write_text(
    json.dumps(comparison, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(comparison, ensure_ascii=False, indent=2))
PY
