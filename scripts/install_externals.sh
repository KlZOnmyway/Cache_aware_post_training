#!/usr/bin/env bash
# Install external dependencies that are not on PyPI (or whose PyPI builds are
# CUDA-version-specific): slime, Megatron-LM, transformer-engine, sglang.
#
# Run this AFTER ``uv sync --extra train`` so that torch + the rest of the
# Python deps are already installed in .venv.
#
# Usage:
#   bash scripts/install_externals.sh                   # install everything
#   bash scripts/install_externals.sh slime             # install one package only
#   bash scripts/install_externals.sh --no-sglang       # skip sglang
#
# Each external is cloned into ./external/<name>/ and installed editable so you
# can ``git pull`` + ``uv pip install -e ./external/<name>`` to refresh.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNALS="${ROOT}/external"
mkdir -p "${EXTERNALS}"

# Pin commits/branches here so reproducibility is locked in source control.
SLIME_REPO="https://github.com/THUDM/slime.git"
SLIME_REF="${SLIME_REF:-main}"

MEGATRON_REPO="https://github.com/NVIDIA/Megatron-LM.git"
# slime expects a particular commit; bump as slime upgrades.
MEGATRON_REF="${MEGATRON_REF:-core_v0.13.0}"

SGLANG_REPO="https://github.com/sgl-project/sglang.git"
SGLANG_REF="${SGLANG_REF:-main}"

clone_or_update() {
    local repo="$1" dir="$2" ref="$3"
    if [ ! -d "${dir}" ]; then
        echo "[clone]   ${repo} -> ${dir}"
        git clone --depth=1 --filter=blob:none "${repo}" "${dir}"
        git -C "${dir}" fetch --depth=1 origin "${ref}" || true
        git -C "${dir}" checkout "${ref}" || true
    else
        echo "[update]  ${dir} (ref=${ref})"
        git -C "${dir}" fetch --depth=1 origin "${ref}" || true
        git -C "${dir}" checkout "${ref}" || true
    fi
}

uv_install_editable() {
    local path="$1"
    echo "[install] uv pip install -e ${path}"
    uv pip install -e "${path}" --no-build-isolation
}

want_slime=1
want_megatron=1
want_sglang=1

for arg in "$@"; do
    case "${arg}" in
        --no-slime)    want_slime=0    ;;
        --no-megatron) want_megatron=0 ;;
        --no-sglang)   want_sglang=0   ;;
        slime)         want_slime=1; want_megatron=0; want_sglang=0 ;;
        megatron)      want_slime=0; want_megatron=1; want_sglang=0 ;;
        sglang)        want_slime=0; want_megatron=0; want_sglang=1 ;;
        -h|--help)
            echo "Usage: $0 [slime | megatron | sglang | --skip-X]"
            exit 0
            ;;
    esac
done

if [ "${MEGATRON_LM_PATH:-}" = "" ] && [ "${want_megatron}" = "1" ]; then
    clone_or_update "${MEGATRON_REPO}" "${EXTERNALS}/Megatron-LM" "${MEGATRON_REF:-main}"
    # Megatron-LM does not ship a setup.py for plain pip; it's expected to be
    # on PYTHONPATH instead. Slime's docker image sets MEGATRON_LM_PATH.
    echo "Megatron-LM at ${EXTERNALS}/Megatron-LM (set MEGATRON_LM_PATH=$(realpath "${EXTERNALS}/Megatron-LM") in your shell)"
fi

if [ "${want_slime}" = "1" ]; then
    if [ ! -d "${EXTERNALS}/slime" ]; then
        echo "[install_externals] cloning slime"
        git clone --depth=1 --filter=blob:none "${SLIME_REPO:-${SLIME_REPO}}" "${EXTERNALS}/slime"
    fi
    git -C "${EXTERNALS}/slime" fetch --depth=1 origin "${SLIME_REF:-main}" || true
    git -C "${EXTERNALS}/slime" checkout "${SLIME_REF:-main}" || true
    uv pip install -e "${EXTERNALS}/slime" --no-build-isolation
fi

if [ "${want_sglang}" = "1" ]; then
    if [ ! -d "${EXTERNALS}/sglang" ]; then
        git clone --depth=1 "${SGLANG_REPO}" "${EXTERNALS}/sglang"
    fi
    git -C "${EXTERNALS}/sglang" fetch --depth=1 origin "${SGLANG_REF}" || true
    git -C "${EXTERNALS}/sglang" checkout "${SGLANG_REF}" || true
    # sglang ships its python under ./python
    uv pip install -e "${EXTERNALS}/sglang/python" --no-build-isolation
fi

echo
echo "[install_externals] done."
echo "External repos in:    ${EXTERNALS}"
if [ "${want_megatron:-1}" = "1" ]; then
    echo "  Megatron-LM:   ${EXTERNALS}/Megatron-LM"
    echo "  → set MEGATRON_LM_PATH=${EXTERNALS}/Megatron-LM in your env"
fi
if [ "${want_slime}" = "1" ]; then
    echo "  slime:         ${EXTERNALS}/slime"
fi
if [ "${want_sglang}" = "1" ]; then
    echo "  sglang:        ${EXTERNALS}/sglang"
fi
