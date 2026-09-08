#!/usr/bin/env bash
# run_flm.sh — Robust FastFlowLM launcher with proper XRT and multiarch environment
set -e

MODEL_TAG="${1:-minicpm5:2b}"
PORT="${2:-8001}"
HOST="${3:-127.0.0.1}"

# Locate FLM binary directory
FLM_BIN="$(which flm 2>/dev/null || true)"
if [ -z "$FLM_BIN" ]; then
    if [ -f "/tmp/opencode/flm102/flm" ]; then
        FLM_DIR="/tmp/opencode/flm102"
        FLM_BIN="${FLM_DIR}/flm"
    elif [ -f "${HOME}/.config/flm/flm" ]; then
        FLM_DIR="${HOME}/.config/flm"
        FLM_BIN="${FLM_DIR}/flm"
    elif [ -f "/var/cache/lemonade/bin/flm/npu/flm" ]; then
        FLM_DIR="/var/cache/lemonade/bin/flm/npu"
        FLM_BIN="${FLM_DIR}/flm"
    else
        echo "[ERROR] Could not find 'flm' binary in PATH or standard directories."
        echo "Please install FastFlowLM or set FLM_DIR."
        exit 1
    fi
else
    FLM_DIR="$(cd "$(dirname "$FLM_BIN")" && pwd)"
fi

echo "=================================================="
echo " 🧠 FastFlowLM NPU Server Launcher"
echo "    Binary: ${FLM_BIN}"
echo "    Model:  ${MODEL_TAG}"
echo "    Host:   ${HOST}:${PORT}"
echo "=================================================="

# Check /dev/accel/accel0
if [ ! -e "/dev/accel/accel0" ]; then
    echo "[WARN] /dev/accel/accel0 not found. Check amdxdna driver."
fi

# Multiarch XRT library resolution
export LD_LIBRARY_PATH="${FLM_DIR}/lib:${LD_LIBRARY_PATH}"
export XILINX_XRT="${FLM_DIR}"

MULTIARCH_DIR="${FLM_DIR}/lib/x86_64-linux-gnu"
if [ ! -d "$MULTIARCH_DIR" ] && [ -d "${FLM_DIR}/lib" ]; then
    mkdir -p "$MULTIARCH_DIR"
    cd "${FLM_DIR}/lib"
    for lib in libxrt*.so* libxrt++.so*; do
        if [ -f "$lib" ] || [ -L "$lib" ]; then
            ln -sf "../$lib" "$MULTIARCH_DIR/$lib" 2>/dev/null || true
        fi
    done
    cd - >/dev/null
fi

echo "[INFO] Starting FLM server..."
exec "${FLM_BIN}" serve "${MODEL_TAG}" --host "${HOST}" --port "${PORT}"
