#!/usr/bin/env bash
# ============================================================================
# Cache-aware MoE distillation — production launcher.
# ============================================================================
#
# Usage:
#     # 8-GPU (default): all GPUs for training+student, teacher external
#     bash scripts/launch_train.sh
#
#     # 6-GPU: 4 train + 2 teacher (auto-launched)
#     TRAIN_GPUS=0,1,2,3 TEACHER_GPUS=4,5 LAUNCH_TEACHER=1 \
#         bash scripts/launch_train.sh configs/qwen3_30b_a3b_6gpu.yaml
#
#     # 8-GPU: 8 train, teacher already running elsewhere
#     TEACHER_URL=http://teacher-host:30001 \
#         bash scripts/launch_train.sh configs/qwen3_30b_a3b_8gpu.yaml
#
# Environment variables:
#     TRAIN_GPUS      GPUs for Megatron training + SGLang student (default: CUDA_VISIBLE_DEVICES or 0..7)
#     TEACHER_GPUS    GPUs for teacher SGLang server (default: none — teacher must be external)
#     LAUNCH_TEACHER  Set to 1 to auto-launch teacher on TEACHER_GPUS (requires TEACHER_GPUS)
#     TEACHER_URL     Teacher SGLang base URL (default: http://localhost:30001)
#     TEACHER_TP      Teacher tensor-parallel size (default: number of TEACHER_GPUS)
#
# Pre-reqs (run once):
#     bash scripts/install_externals.sh                # slime + Megatron-LM + sglang
#
# Tensorboard:
#     tensorboard --logdir runs/
# ============================================================================

set -euo pipefail

# --- locate repo root ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# --- pick config ---
CONFIG="${1:-configs/qwen3_30b_a3b_8gpu.yaml}"
if [ ! -f "${CONFIG}" ]; then
    echo "[launch_train] ERROR: config not found: ${CONFIG}"
    exit 1
fi

# --- env: external repos on PYTHONPATH ---
EXT="${ROOT}/external"
export PYTHONPATH="${EXT}/Megatron-LM:${EXT}/slime:${PYTHONPATH:-}"
export ENABLE_ROUTING_REPLAY=1

# --- GPU layout ---
TRAIN_GPUS="${TRAIN_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
TEACHER_GPUS="${TEACHER_GPUS:-}"
NGPUS=$(echo "${TRAIN_GPUS}" | awk -F',' '{print NF}')

# --- run dir ---
RUN_NAME="$(grep -E '^run_name:' "${CONFIG}" | head -1 | awk '{print $2}')"
RUN_NAME="${RUN_NAME:-cache_distill_$(date +%Y%m%d_%H%M%S)}"
TB_DIR="${ROOT}/runs/${RUN_NAME}"
mkdir -p "${TB_DIR}"

# --- bootstrap teacher SGLang server (background) if requested ---
TEACHER_URL="${TEACHER_URL:-http://localhost:30001}"
if [ "${LAUNCH_TEACHER:-0}" = "1" ]; then
    if [ -z "${TEACHER_GPUS}" ]; then
        echo "[launch_train] ERROR: LAUNCH_TEACHER=1 requires TEACHER_GPUS (e.g. TEACHER_GPUS=4,5)"
        exit 1
    fi
    TEACHER_NGPUS=$(echo "${TEACHER_GPUS}" | awk -F',' '{print NF}')
    TEACHER_TP="${TEACHER_TP:-${TEACHER_NGPUS}}"
    TEACHER_PORT="$(echo "${TEACHER_URL}" | grep -oE '[0-9]+$')"
    TEACHER_PORT="${TEACHER_PORT:-30001}"
    HF_CKPT="$(grep -E '^\s*hf_checkpoint:' "${CONFIG}" | awk '{print $2}')"

    echo "[launch_train] starting teacher SGLang on ${TEACHER_URL} ..."
    echo "[launch_train]   GPUs: ${TEACHER_GPUS} (TP=${TEACHER_TP})"
    CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}" \
    nohup uv run --frozen --no-sync python -m sglang.launch_server \
        --model-path "${HF_CKPT}" \
        --tp "${TEACHER_TP}" \
        --port "${TEACHER_PORT}" \
        --mem-fraction-static 0.85 \
        --disable-cuda-graph \
        > "${TB_DIR}/teacher.log" 2>&1 &
    TEACHER_PID=$!
    echo "[launch_train]   PID=${TEACHER_PID} (logs: ${TB_DIR}/teacher.log)"
    for i in $(seq 1 60); do
        if curl -s "${TEACHER_URL}/get_model_info" >/dev/null 2>&1; then
            echo "[launch_train]   teacher ready"
            break
        fi
        if [ "$i" = "60" ]; then
            echo "[launch_train] WARNING: teacher not responding after 5 min; continuing anyway"
        fi
        sleep 5
    done
fi

# --- launch trainer ---
echo "[launch_train] config:        ${CONFIG}"
echo "[launch_train] train GPUs:    ${TRAIN_GPUS} (${NGPUS} GPUs)"
if [ -n "${TEACHER_GPUS}" ]; then
    echo "[launch_train] teacher GPUs:  ${TEACHER_GPUS}"
fi
echo "[launch_train] teacher_url:   ${TEACHER_URL}"
echo "[launch_train] tensorboard:   ${TB_DIR}"
echo

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export SLIME_ADAPTER_TB_DIR="${TB_DIR}"
export TEACHER_URL

exec uv run --frozen --no-sync python scripts/run_train.py --config "${CONFIG}"
