#!/usr/bin/env bash
# run_flm.sh — Robust FastFlowLM launcher with proper XRT and multiarch environment
set -e

MODEL_TAG="${1:-minicpm5:2b}"
PORT="${2:-8001}"
HOST="${3:-127.0.0.1}"

# Locate FLM binary directory (FLM_DIR env wins, then PATH, then known locations)
FLM_BIN=""
if [ -n "${FLM_DIR:-}" ] && [ -f "${FLM_DIR}/flm" ]; then
    FLM_BIN="${FLM_DIR}/flm"
else
    FLM_BIN="$(which flm 2>/dev/null || true)"
fi
if [ -z "$FLM_BIN" ]; then
    for cand in "${HOME}/.local"/flm*/flm \
                /tmp/opencode/flm*/flm \
                "${HOME}/.config/flm/flm" \
                "/var/cache/lemonade/bin/flm/npu/flm"; do
        if [ -f "$cand" ]; then
            FLM_BIN="$cand"
            break
        fi
    done
    if [ -z "$FLM_BIN" ]; then
        echo "[ERROR] Could not find 'flm' binary. Install it first:"
        echo "  ./scripts/setup_flm.sh"
        echo "or set FLM_DIR to your FastFlowLM install directory."
        exit 1
    fi
fi
FLM_DIR="$(cd "$(dirname "$FLM_BIN")" && pwd)"

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

# FLM mmaps ~2.4 GB with MAP_LOCKED; the default 8 MB memlock cap kills serve.
if [ "$(ulimit -l)" != "unlimited" ]; then
    echo "[WARN] memlock limit is $(ulimit -l) KB (need unlimited). Serve will fail with mmap err=-11."
    echo "       Fix: sudo sh -c 'printf \"* soft memlock unlimited\n* hard memlock unlimited\n\" >> /etc/security/limits.conf'"
    echo "       then log out/in (or reboot). See README troubleshooting."
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
