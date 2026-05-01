#!/usr/bin/env bash
# Bootstrap a fresh server: install uv, create venv from uv.lock, install externals.
#
# Usage:
#   bash scripts/setup_env.sh                    # full setup (train + serve + dev)
#   bash scripts/setup_env.sh train              # train deps only
#   bash scripts/setup_env.sh serve              # serve deps only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# 1. Install uv if not already present.
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

uv --version

# 2. Create venv + install Python deps from lockfile.
profile="${1:-all}"
case "${profile}" in
    all)
        echo "[setup] uv sync --all-extras"
        uv sync --all-extras
        ;;
    train)
        echo "[setup] uv sync --extra train"
        uv sync --extra train --extra dev
        ;;
    serve)
        echo "[setup] uv sync --extra serve"
        uv sync --extra serve --extra dev
        ;;
    dev)
        echo "[setup] uv sync --extra dev"
        uv sync --extra dev
        ;;
    *)
        echo "Unknown profile: ${profile}. Use one of: all train serve dev"
        exit 1
        ;;
esac

# Activate the venv so subsequent commands hit it.
source .venv/bin/activate

# Install externals that are not on PyPI (slime / Megatron / sglang).
if [ "${profile}" = "all" ] || [ "${profile}" = "train" ]; then
    bash scripts/install_externals.sh --no-sglang
fi
if [ "${profile}" = "all" ] || [ "${profile}" = "serve" ]; then
    bash scripts/install_externals.sh --no-megatron
fi

echo
echo "[setup] done. To activate the env later:"
echo "  source ${ROOT}/.venv/bin/activate"
