#!/usr/bin/env bash
# ============================================================================
# Quickstart — single-command setup + launch.
# ============================================================================
#
# What this does, in order:
#   1. install uv (skip if already present)
#   2. uv sync from uv.lock                                (creates ./.venv)
#   3. clone & install externals (slime / Megatron-LM / sglang) into ./external/
#   4. start a frozen teacher SGLang server on port 30001
#      (skip if SKIP_TEACHER=1; reuse if already up)
#   5. launch training via scripts/launch_train.sh
#
# Usage:
#   bash scripts/quickstart.sh                              # default config
#   bash scripts/quickstart.sh configs/qwen3_30b_a3b_8gpu.yaml
#
# Environment knobs:
#   SKIP_SETUP=1          skip step 1+2 (Python env)
#   SKIP_EXTERNALS=1      skip step 3 (slime/Megatron/sglang clone+install)
#   SKIP_TEACHER=1        skip step 4 (assume teacher already running)
#   TEACHER_PORT=30001    teacher SGLang port (default 30001)
#   CUDA_VISIBLE_DEVICES  defaults to 0..NGPUS-1 from nvidia-smi
# ============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG="${1:-configs/qwen3_30b_a3b_8gpu.yaml}"
if [ ! -f "${CONFIG}" ]; then
    echo "[quickstart] ERROR: config '${CONFIG}' not found"
    exit 1
fi

TEACHER_PORT="${TEACHER_PORT:-30001}"
TEACHER_HOST="${TEACHER_HOST:-localhost}"
TEACHER_URL="http://${TEACHER_HOST}:${TEACHER_PORT}"

echo "============================================================"
echo "  Cache-aware MoE distillation — quickstart"
echo "  Config:        ${CONFIG:=${CONFIG}}"
echo "  Teacher URL:   ${TEACHER_URL}"
echo "============================================================"

# ---- 1+2. Python env ------------------------------------------------------
if [ "${SKIP_SETUP:-0}" != "1" ]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "[quickstart] installing uv ..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    echo "[quickstart] uv sync (this can take a few min on first run)"
    uv sync --frozen
fi

# ---- 3. External deps -----------------------------------------------------
if [ "${SKIP_EXTERNALS:-0}" != "1" ]; then
    bash "${ROOT}/scripts/install_externals.sh"
fi

# ---- 4. Teacher SGLang ----------------------------------------------------
if [ "${SKIP_TEACHER:-0}" != "1" ]; then
    if curl -s "http://${TEACHER_HOST}:${TEACHER_PORT}/get_model_info" >/dev/null 2>&1; then
        echo "[quickstart] teacher already up at ${TEACHER_URL}"
    else
        TEACHER_MODEL="${TEACHER_MODEL:-$(grep -E '^\s*hf_checkpoint:' "${1:-configs/qwen3_30b_a3b_8gpu.yaml}" | awk '{print $2}')}"
        if [ -z "${TEACHER_MODEL}" ]; then
            echo "[quickstart] WARN: TEACHER_MODEL unset and no hf_checkpoint in config; skipping teacher launch"
        else
            mkdir -p logs
            LOG="logs/teacher_$(date +%Y%m%d_%H%M%S).log"
            echo "[quickstart] starting teacher SGLang ${TEACHER_MODEL} on :${TEACHER_PORT} ..."
            nohup uv run --frozen --no-sync python -m sglang.launch_server \
                --model-path "${TEACHER_MODEL}" \
                --port "${TEACHER_PORT}" \
                --mem-fraction-static 0.7 \
                --disable-cuda-graph \
                > "${LOG}" 2>&1 &
            echo "[quickstart] teacher PID=$!  log=${LOG}"
            for i in $(seq 1 60); do
                if curl -s "${TEACHER_URL}/get_model_info" >/dev/null 2>&1; then
                    echo "[quickstart] teacher ready"
                    break
                fi
                sleep 5
                if [ "$i" = 60 ]; then
                    echo "[quickstart] ERROR: teacher did not come up in 5min; see ${LOG}"
                    exit 2
                fi
            done
        fi
    fi
fi

# ---- 5. Launch training ---------------------------------------------------
echo "[quickstart] launching trainer with config ${CONFIG}"
exec bash scripts/launch_train.sh "${CONFIG}"
