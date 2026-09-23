#!/usr/bin/env bash
# build_open_weights.sh — build the shim-free Q4NX container for the open-kernel
# (oflm) route: expand KV heads -> BF16 GGUF -> Q4NX (llama family, Q4_1).
#
# Output: ${OUT_DIR} (default ~/.cache/oflm-weights/MiniCPM5-2B-OFLM) with
# model.q4nx + config.json + tokenizer files. No QK-norm shim is applied:
# the open llama3 engine (qk_norm=false) must NOT see q_norm/k_norm tensors.
#
# Usage:
#   ./scripts/build_open_weights.sh
#   SRC_MODEL=openbmb/MiniCPM5-2B ./scripts/build_open_weights.sh  # local dir or HF id
#
# Requirements (checked below): python3 venv at $OFLM_VENV with torch
# (CPU is fine), safetensors, numpy, gguf, transformers, sentencepiece,
# huggingface_hub, einops, accelerate; git checkouts of llama.cpp and
# Atomic-Germ/FLM_Q4NX_Converter under $OFLM_TOOLS (cloned if missing).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/oflm_env.sh"

SRC_MODEL="${SRC_MODEL:-openbmb/MiniCPM5-2B}"
WORK_DIR="${WORK_DIR:-${HOME}/.cache/oflm-weights}"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/MiniCPM5-2B-OFLM}"
FORCE="${FORCE:-0}"

PY="${OFLM_VENV}/bin/python"
LLAMA_DIR="${OFLM_TOOLS}/llama.cpp"
CONV_DIR="${OFLM_TOOLS}/FLM_Q4NX_Converter"

echo "=================================================="
echo " 🧱 Open-kernel weight pipeline"
echo "     src: ${SRC_MODEL}"
echo "     out: ${OUT_DIR}"
echo "=================================================="

[ -x "${PY}" ] || { echo "[ERROR] No python at ${PY}. Create it and install: torch safetensors numpy gguf transformers sentencepiece huggingface_hub einops accelerate"; exit 1; }
for mod in torch safetensors gguf transformers sentencepiece huggingface_hub; do
    "${PY}" -c "import ${mod}" 2>/dev/null || { echo "[ERROR] python module '${mod}' missing in ${OFLM_VENV}. pip-install it first."; exit 1; }
done

mkdir -p "${WORK_DIR}" "${OFLM_TOOLS}"
[ -f "${LLAMA_DIR}/convert_hf_to_gguf.py" ] || git clone --depth 1 https://github.com/ggerganov/llama.cpp "${LLAMA_DIR}"
[ -f "${CONV_DIR}/convert.py" ] || git clone --depth 1 https://github.com/Atomic-Germ/FLM_Q4NX_Converter "${CONV_DIR}"

# 1. Fetch original weights (skip if a local dir was given and has safetensors)
if [ -d "${SRC_MODEL}" ]; then
    SRC_DIR="${SRC_MODEL}"
else
    SRC_DIR="${WORK_DIR}/MiniCPM5-2B-src"
    if ! ls "${SRC_DIR}"/*.safetensors >/dev/null 2>&1; then
        echo "[INFO] Downloading ${SRC_MODEL}..."
        "${PY}" -c "from huggingface_hub import snapshot_download; snapshot_download('${SRC_MODEL}', local_dir='${SRC_DIR}')"
    fi
fi

# 2. KV expansion 2 -> 8 (16:2 -> 16:8 GQA, math-identical under GQA)
ADAPT_DIR="${WORK_DIR}/MiniCPM5-2B-gqa8"
if [ "${FORCE}" = "1" ] || ! ls "${ADAPT_DIR}"/*.safetensors >/dev/null 2>&1; then
    echo "[INFO] Expanding KV heads..."
    "${PY}" "${SCRIPT_DIR}/expand_kv_heads.py" --src "${SRC_DIR}" --dst "${ADAPT_DIR}" --target-kv-heads 8
else
    echo "[OK] Adapted weights present, skipping (FORCE=1 to redo)"
fi

# 3. BF16 GGUF (llama.cpp has no direct Q4_1 outtype; the Q4NX converter
#    quantizes from BF16 itself — and rejects Q4_0 sources as corrupt)
GGUF="${WORK_DIR}/minicpm5_2b_gqa8_bf16.gguf"
if [ "${FORCE}" = "1" ] || [ ! -f "${GGUF}" ]; then
    echo "[INFO] Converting to BF16 GGUF (slow, ~5 GB)..."
    (cd "${LLAMA_DIR}" && "${PY}" convert_hf_to_gguf.py "${ADAPT_DIR}" --outfile "${GGUF}" --outtype bf16)
else
    echo "[OK] BF16 GGUF present, skipping (FORCE=1 to redo)"
fi

# 4. Q4NX container, llama family (NOT qwen3: no shim injected there)
if [ "${FORCE}" = "1" ] || [ ! -f "${OUT_DIR}/model.q4nx" ]; then
    echo "[INFO] Converting BF16 GGUF -> Q4NX (llama)..."
    mkdir -p "${OUT_DIR}"
    (cd "${CONV_DIR}" && PYTHONPATH="${CONV_DIR}" "${PY}" convert.py -i "${GGUF}" -o "${OUT_DIR}" -f llama -s "${ADAPT_DIR}")
else
    echo "[OK] Q4NX container present, skipping (FORCE=1 to redo)"
fi

echo "[INFO] Verifying container..."
"${PY}" - "${OUT_DIR}/model.q4nx" <<'EOF'
import struct, json, sys
h = json.loads(open(sys.argv[1],'rb').read(struct.unpack('<Q', open(sys.argv[1],'rb').read(8))[0]+8)[8:])
keys = [k for k in h if k != '__metadata__']
assert not any('q_norm' in k or 'k_norm' in k for k in keys), "shim tensors present — wrong family!"
print(f"[OK] {len(keys)} tensors, no q_norm/k_norm shim")
EOF

echo "=================================================="
echo " Weights ready at ${OUT_DIR}"
echo " Next: ./scripts/setup_oflm.sh  (then ./scripts/run_oflm.sh)"
echo "=================================================="
