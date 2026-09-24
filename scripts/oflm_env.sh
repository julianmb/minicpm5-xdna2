# oflm_env.sh — shared environment for the open-kernel (oflm) path.
# Source from setup_oflm.sh / run_oflm.sh, or in your shell:
#   source scripts/oflm_env.sh
#
# All locations are overridable via environment variables. Defaults match
# what scripts/setup_oflm.sh creates. No sudo required for any of it
# (except the one-time memlock fix, see README).

# Where the OpenFlowLM-Next checkout lives
: "${OFLM_SRC:=${HOME}/src/OpenFlowLM-Next}"
# No-sudo sysroot with staged -dev/-runtime .debs (Boost, XRT tools, ffmpeg, ...)
: "${OFLM_SYSROOT:=${HOME}/.local/sysroot}"
# FastFlowLM tarball install — reused as the XRT *runtime* lib source
: "${FLM_ROOT:=${HOME}/.local/flm106}"
# XRT runtime root (mirror layout XRT >= 2.25 requires; built by setup_oflm.sh)
: "${XRT_ROOT:=${HOME}/.local/xrtroot}"
# Python venv for the weight pipeline + oflm-add
: "${OFLM_VENV:=${HOME}/.venvs/npu}"
# oflm user-level state: config root (registry + models + xclbins)
: "${OFLM_ROOT:=${HOME}/.config/oflm}"
# Open-kernel model registry (written by oflm-add)
: "${OFLM_CONFIG_PATH:=${OFLM_ROOT}/model_list.json}"
: "${OFLM_XCLBIN_PATH:=${OFLM_ROOT}}"
# The runtime resolves models as $OFLM_MODEL_PATH + <registry model_path> + <name>.
# model_path is "models", so this must be the CONFIG dir, not the models dir —
# pointing it at .../oflm/models yields .../oflm/models/models/<name>.
: "${OFLM_MODEL_PATH:=${OFLM_ROOT}}"
export OFLM_MODEL_PATH
# Running from a build tree rather than an install prefix: oflm can't find
# model_info.json beside the binary, and will try to "pull" instead of using
# the locally registered model.
if [ -f "${OFLM_SRC}/src/model_info.json" ]; then
    export OFLM_MODELINFO_PATH="${OFLM_SRC}/src/model_info.json"
fi
export OFLM_SRC OFLM_SYSROOT FLM_ROOT XRT_ROOT OFLM_VENV
export OFLM_CONFIG_PATH OFLM_XCLBIN_PATH

# The (128,16,8,128) attention tuple is outside the validated catalogue,
# exactly as in the verified recipe — required at build AND serve time.
export OPEN_KERNELS_UNVALIDATED=1

export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${OFLM_SYSROOT}/usr/bin:${PATH}"
export CMAKE_PREFIX_PATH="${OFLM_SYSROOT}/usr${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export PKG_CONFIG_PATH="${OFLM_SYSROOT}/usr/lib/x86_64-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
export CPLUS_INCLUDE_PATH="${OFLM_SYSROOT}/usr/include:${OFLM_SYSROOT}/usr/include/x86_64-linux-gnu${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export C_INCLUDE_PATH="${OFLM_SYSROOT}/usr/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
export LIBRARY_PATH="${OFLM_SYSROOT}/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${XRT_ROOT}/lib/x86_64-linux-gnu:${XRT_ROOT}/lib:${OFLM_SYSROOT}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export XILINX_XRT="${XRT_ROOT}"

# Helper checkouts (llama.cpp, converters, XRT headers, aiebu sources)
: "${OFLM_TOOLS:=${HOME}/src/build-tools}"
export OFLM_TOOLS

# Must be an array: CMAKE_CXX_STANDARD_LIBRARIES contains a space, which would
# word-split if these were a plain string and hand cmake a bare "-lidn2".
OFLM_CMAKE_FLAGS=(
    "-DOFLM_KERNEL_SPECS=minicpm5-2b"
    "-DXRT_INCLUDE_DIR=${OFLM_TOOLS}/XRT/src/runtime_src/core/include"
    "-DXRT_LIB_DIR=${FLM_ROOT}/lib"
    "-DCMAKE_CXX_STANDARD_LIBRARIES=-lz -lidn2"
    "-DCURL_LIBRARY=${OFLM_SYSROOT}/usr/lib/x86_64-linux-gnu/libcurl.so"
    "-DCURL_LIBRARY_RELEASE=${OFLM_SYSROOT}/usr/lib/x86_64-linux-gnu/libcurl.so"
)
