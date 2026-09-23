#!/usr/bin/env bash
# run_oflm.sh — serve MiniCPM5-2B via the open-kernel engine.
# Usage: ./scripts/run_oflm.sh [model_tag] [port] [host]
#   ./scripts/run_oflm.sh minicpm5:2b 8001
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/oflm_env.sh"

MODEL_TAG="${1:-minicpm5:2b}"
PORT="${2:-8001}"
HOST="${3:-127.0.0.1}"
OFLM_BIN="${OFLM_SRC}/build/src/oflm"

[ -x "${OFLM_BIN}" ] || { echo "[ERROR] oflm not built. Run ./scripts/setup_oflm.sh first."; exit 1; }
[ -e /dev/accel/accel0 ] || echo "[WARN] /dev/accel/accel0 missing (amdxdna driver?)"
[ "$(ulimit -l)" = "unlimited" ] || echo "[WARN] memlock is $(ulimit -l) KB, need unlimited (see README)."

echo "=================================================="
echo " 🧠 oflm open-kernel server"
echo "    Binary: ${OFLM_BIN}"
echo "    Model:  ${MODEL_TAG}"
echo "    Host:   ${HOST}:${PORT}"
echo "=================================================="
exec "${OFLM_BIN}" serve "${MODEL_TAG}" --host "${HOST}" --port "${PORT}"
