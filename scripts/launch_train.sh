#!/usr/bin/env bash
# ============================================================================
# Cache-aware MoE distillation — production launcher (single node, 8 GPUs).
# ============================================================================
#
# Usage:
#     bash scripts/launch_train.sh                                # default config
#     bash scripts/launch_train.sh configs/qwen3_30b_a3b_8gpu.yaml
#     CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/launch_train.sh ...   # 4-GPU
#
# Pre-reqs (run once, see scripts/install_externals.sh):
#     bash scripts/setup_env.sh
#     bash scripts/install_externals.sh                # slime + Megatron-LM + sglang
#
# Tensorboard:
#     tensorboard --logdir runs/                       # then open http://host:6006
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
export ENABLE_ROUTING_REPLAY=1                    # slime register_routing_replay needs this

# --- GPU layout ---
GPUS_AVAILABLE="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NGPUS=$(echo "${GPUS_AVAILABLE}" | awk -F',' '{print NF}')
export CUDA_VISIBLE_DEVICES="${GPUS_AVAILABLE}"

# --- run dir ---
RUN_NAME="$(grep -E '^run_name:' "${CONFIG}" | head -1 | awk '{print $2}')"
RUN_NAME="${RUN_NAME:-cache_distill_$(date +%Y%m%d_%H%M%S)}"
TB_DIR="${ROOT}/runs/${RUN_NAME}"
mkdir -p "${TB_DIR}"

# --- bootstrap teacher SGLang server (background) if requested ---
TEACHER_URL="${TEACHER_URL:-http://localhost:30001}"
if [ "${LAUNCH_TEACHER:-0}" = "1" ]; then
    echo "[launch_train] starting teacher SGLang on ${TEACHER_URL} ..."
    HF_CKPT="$(grep -E '^\s*hf_checkpoint:' "${CONFIG}" | awk '{print $2}')"
    nohup uv run --frozen --no-sync python -m sglang.launch_server \
        --model-path "${HF_CKPT}" \
        --port 30001 \
        --mem-fraction-static 0.8 \
        --disable-cuda-graph \
        > "${TB_DIR}/teacher.log" 2>&1 &
    TEACHER_PID=$!
    echo "[launch_train] teacher PID=${TEACHER_PID} (logs at ${TB_DIR}/teacher.log)"
    # wait for teacher to come up
    for i in $(seq 1 60); do
        if curl -s "${TEACHER_URL}/get_model_info" >/dev/null 2>&1; then
            echo "[launch_train] teacher ready"
            break
        fi
        sleep 5
    done
fi

# --- launch trainer ---
echo "[launch_train] config:        ${CONFIG}"
echo "[launch_train] gpus:          ${CUDA_VISIBLE_DEVICES} (${NGPUS} GPUs)"
echo "[launch_train] tensorboard:   ${TB_DIR}"
echo "[launch_train] teacher_url:   ${TEACHER_URL}"
echo

export SLIME_ADAPTER_TB_DIR="${TB_DIR}"
export TEACHER_URL

# slime expects a python entry point; we use slime/train.py with overrides from yaml.
# For the v1 single-node smoke we use our own driver script that loads the yaml
# and dispatches to slime.train (or to scripts/single_gpu_real_qwen3.py for the
# in-process smoke).
exec uv run --frozen --no-sync python scripts/run_train.py --config "${CONFIG}"
