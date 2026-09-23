#!/usr/bin/env bash
# setup_flm.sh — One-shot FastFlowLM + MiniCPM5-2B-NPU2 setup (no sudo needed,
# except for the one-time memlock fix described below).
#
# What it does:
#   1. Checks NPU device (/dev/accel/accel0) and memlock limit.
#   2. Downloads the FastFlowLM Linux tarball (static build, bundled XRT) into
#      ~/.local/flm<ver> (persistent across reboots, unlike /tmp).
#   3. Downloads the prebuilt MiniCPM5-2B-NPU2 weights from Hugging Face.
#   4. Registers minicpm5:2b in the FLM install's own model_list.json
#      (FLM >= 1.0.x ignores ~/.config/flm/model_list.json), links the model
#      weights, and copies the *.xclbin kernels into place.
#   5. Runs `flm validate` and confirms the model is listed.
#
# Usage:
#   ./scripts/setup_flm.sh
#   FLM_VERSION=v1.0.6 ./scripts/setup_flm.sh   # pin a different release
#
# NOTE: serving still requires memlock=unlimited (root, one time):
#   sudo sh -c 'printf "* soft memlock unlimited\n* hard memlock unlimited\n" >> /etc/security/limits.conf'
#   # then fully log out/in (or reboot) and verify: ulimit -l  ->  unlimited
set -e

FLM_VERSION="${FLM_VERSION:-v1.0.6}"
FLM_VER_NODOTS="${FLM_VERSION#v}"
FLM_VER_NODOTS="${FLM_VER_NODOTS//./}"
FLM_ROOT="${FLM_ROOT:-${HOME}/.local/flm${FLM_VER_NODOTS}}"
MODEL_DIR="${MODEL_DIR:-${HOME}/.config/flm/models/MiniCPM5-2B-NPU2}"
MODEL_URL="https://huggingface.co/julianmb/MiniCPM5-2B-NPU2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=================================================="
echo " 🛠️  MiniCPM5-2B-NPU2 setup (FLM ${FLM_VERSION})"
echo "     FLM root:  ${FLM_ROOT}"
echo "     Model dir: ${MODEL_DIR}"
echo "=================================================="

# 1. NPU device + memlock -----------------------------------------------------
if [ ! -e "/dev/accel/accel0" ]; then
    echo "[ERROR] /dev/accel/accel0 not found. Check the amdxdna driver + firmware."
    exit 1
fi
echo "[OK] /dev/accel/accel0 present"

if [ "$(ulimit -l)" != "unlimited" ]; then
    echo "[WARN] memlock limit is $(ulimit -l) KB, FLM needs ~2.4 GB locked."
    echo "       Serve will fail with: mmap(...) failed (err=-11)."
    echo "       Fix (root, one time), then log out/in or reboot:"
    echo "         sudo sh -c 'printf \"* soft memlock unlimited\n* hard memlock unlimited\n\" >> /etc/security/limits.conf'"
else
    echo "[OK] memlock unlimited"
fi

# 2. FastFlowLM runtime --------------------------------------------------------
if [ ! -x "${FLM_ROOT}/flm" ]; then
    echo "[INFO] Downloading FastFlowLM ${FLM_VERSION}..."
    mkdir -p "${FLM_ROOT}"
    TARBALL="${HOME}/.local/fastflowlm_${FLM_VERSION}_linux.tar.gz"
    if [ ! -f "${TARBALL}" ]; then
        curl -sL --max-time 600 \
            "https://github.com/ROCm/FastFlowLM/releases/download/${FLM_VERSION}/fastflowlm_${FLM_VERSION}_linux.tar.gz" \
            -o "${TARBALL}"
    fi
    tar xzf "${TARBALL}" -C "${FLM_ROOT}"
    echo "[OK] Extracted to ${FLM_ROOT}"
else
    echo "[OK] FLM already at ${FLM_ROOT} ($("${FLM_ROOT}/flm" --version 2>/dev/null || echo unknown))"
fi

# 3. Model weights --------------------------------------------------------------
if [ ! -f "${MODEL_DIR}/model.q4nx" ]; then
    echo "[INFO] Downloading prebuilt weights (~1.9 GB)..."
    mkdir -p "${MODEL_DIR}"
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download('julianmb/MiniCPM5-2B-NPU2', local_dir='${MODEL_DIR}')"
    elif command -v git >/dev/null; then
        git clone "${MODEL_URL}" "${MODEL_DIR}"
    else
        echo "[ERROR] Need python3+huggingface_hub or git to fetch ${MODEL_URL}"
        exit 1
    fi
else
    echo "[OK] Model weights present (${MODEL_DIR})"
fi

# 4. Registration (FLM >= 1.0.x reads the model_list.json next to its binary) --
echo "[INFO] Registering minicpm5:2b..."
python3 - "${FLM_ROOT}/model_list.json" "${REPO_DIR}/configs/model_list_entry.json" <<'EOF'
import json, sys
base = json.load(open(sys.argv[1]))
entry = json.load(open(sys.argv[2]))
base.setdefault("models", {}).update(entry["models"])
json.dump(base, open(sys.argv[1], "w"), indent=2)
print("[OK] entry merged into", sys.argv[1])
EOF

mkdir -p "${FLM_ROOT}/models" "${FLM_ROOT}/xclbins/MiniCPM5-2B-NPU2"
ln -sfn "${MODEL_DIR}" "${FLM_ROOT}/models/MiniCPM5-2B-NPU2"
cp -f "${MODEL_DIR}"/*.xclbin "${FLM_ROOT}/xclbins/MiniCPM5-2B-NPU2/"
echo "[OK] Model linked, $(ls "${FLM_ROOT}/xclbins/MiniCPM5-2B-NPU2/" | tr '\n' ' ')kernels in place"

# 5. Validate -------------------------------------------------------------------
export LD_LIBRARY_PATH="${FLM_ROOT}/lib:${LD_LIBRARY_PATH}"
export XILINX_XRT="${FLM_ROOT}"
echo "--- flm validate ---"
"${FLM_ROOT}/flm" validate || true
echo "--- flm list ---"
"${FLM_ROOT}/flm" list | grep -i -E "minicpm|Models:" || true

echo "=================================================="
echo " Setup complete. Serve with:"
echo "   FLM_DIR=${FLM_ROOT} ./scripts/run_flm.sh minicpm5:2b 8001"
echo " Known status: prefill runs on NPU; decode on the closed Qwen3"
echo " engine fails (runlist ERT error) — see README troubleshooting."
echo "=================================================="
