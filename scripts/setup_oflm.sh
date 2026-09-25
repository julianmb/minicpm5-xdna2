#!/usr/bin/env bash
# setup_oflm.sh — end-to-end open-kernel (oflm) build with NO sudo.
# Idempotent: completed stages are skipped on re-runs.
#
# Stages:
#   0. checks (NPU device, memlock warning, cmake/g++/ninja/git/curl/python3)
#   1. sysroot: exact -dev/-runtime .deb closure staged to $OFLM_SYSROOT
#      (+ local BoostConfig shim for Ubuntu 26.04's Boost 1.90 packaging)
#   2. rustup (cargo, for tokenizers-cpp) if missing
#   3. aiebu from source (provides aiebu-asm) -> ~/.local/bin
#   4. XRT headers (sparse checkout) for compiling against
#   5. FastFlowLM tarball (XRT *runtime* lib source) -> $FLM_ROOT
#   6. OpenFlowLM-Next clone, configure, build `oflm` (+ tokenizers)
#   7. minicpm5-2b open kernel set export (mlir-aie/Peano, self-provisioned)
#   8. XRT runtime mirror ($XRT_ROOT) + oflm-add registration of $MODEL_DIR
#
# Usage:
#   ./scripts/build_open_weights.sh   # first (produces the container)
#   MODEL_DIR=~/.cache/oflm-weights/MiniCPM5-2B-OFLM ./scripts/setup_oflm.sh
#   ./scripts/run_oflm.sh minicpm5:2b 8001
#
# Time: engine build ~10-20 min, kernel export per design ~40s each.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/oflm_env.sh"

MODEL_DIR="${MODEL_DIR:-${HOME}/.cache/oflm-weights/MiniCPM5-2B-OFLM}"
MODEL_TAG="${MODEL_TAG:-minicpm5:2b}"
FLM_VERSION="${FLM_VERSION:-v1.0.6}"
DEB_CACHE="${DEB_CACHE:-${HOME}/.local/debcache/archives}"
OFLM_BIN="${OFLM_SRC}/build/src/oflm"

say() { echo "[$1] $2"; }

# ---- 0. checks ---------------------------------------------------------------
[ -e /dev/accel/accel0 ] || { say ERROR "/dev/accel/accel0 missing (amdxdna driver?)"; exit 1; }
[ "$(ulimit -l)" = "unlimited" ] || say WARN "memlock is $(ulimit -l) KB — serve needs unlimited (see README step 0)"
for t in cmake g++ ninja git curl python3; do
    command -v "$t" >/dev/null || { say ERROR "missing build tool: $t"; exit 1; }
done
mkdir -p "${OFLM_SYSROOT}" "${OFLM_TOOLS}" "${HOME}/.local/bin" "${DEB_CACHE}/partial"

# ---- 1. sysroot (.deb closure, download-only, no root) -----------------------
# NOTE: package names are distro-specific. libuuid-dev does not exist on
# Ubuntu 26.04 (it is uuid-dev there), and Boost's version suffix moves between
# releases — so each name is probed and unknown ones are reported, not fatal.
SYS_BASE="libboost1.90-dev libboost-program-options-dev libboost-program-options1.90-dev libboost-program-options1.90.0 libboost-filesystem1.90.0 libcurl4-openssl-dev libfftw3-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libswresample-dev libreadline-dev libncurses-dev uuid-dev libdrm-dev liblzma-dev liblz4-dev libelf-dev libidn2-dev libxrt2 libxrt-utils libxrt-npu2 libxrt-dev"
if [ ! -f "${OFLM_SYSROOT}/usr/include/boost/beast/core.hpp" ]; then
    say INFO "staging sysroot .debs (one-time, ~1-2 GB with codec closure)..."
    resolved=""
    for p in ${SYS_BASE}; do
        if apt-cache show "$p" >/dev/null 2>&1; then
            resolved="${resolved} ${p}"
        else
            say WARN "package '${p}' not in this distro's archive — skipping it"
        fi
    done
    [ -n "${resolved}" ] || { say ERROR "none of the required build packages resolved; wrong distribution?"; exit 1; }
    # -y is required: with no tty, apt-get prompts "Do you want to continue?"
    # once a download set gets large and aborts on the empty answer.
    # shellcheck disable=SC2086
    if ! apt-get -y -o Debug::NoLocking=1 -o Dir::Cache::Archives="${DEB_CACHE}" \
        --download-only --no-install-recommends install ${resolved} >/dev/null; then
        say ERROR "apt-get download failed for: ${resolved}"
        say ERROR "Check your apt sources, or stage these packages manually into ${DEB_CACHE}."
        exit 1
    fi
    # Boost's versioned names move between releases. Only if no Boost -dev
    # archive landed do we retry with the unversioned meta-packages.
    if ! ls "${DEB_CACHE}"/libboost*-dev*.deb >/dev/null 2>&1; then
        say INFO "no versioned Boost headers downloaded; retrying unversioned meta-packages"
        # shellcheck disable=SC2086
        apt-get -y -o Debug::NoLocking=1 -o Dir::Cache::Archives="${DEB_CACHE}" \
            --download-only --no-install-recommends install libboost-dev libboost-all-dev >/dev/null 2>&1 || true
    fi
    for f in "${DEB_CACHE}"/*.deb; do dpkg-deb -x "$f" "${OFLM_SYSROOT}/" 2>/dev/null; done
    # one-level codec runtime closure (libav* pull many small codec libs)
    for p in libavcodec62 libavformat62 libavutil60 libswscale9 libswresample6; do
        deps=$(apt-cache depends "$p" 2>/dev/null | grep Depends | awk '{print $2}' | grep -v -E "^<|^libc6$|^libgcc-s1$|^libstdc\+\+6$" | tr '\n' ' ')
        # shellcheck disable=SC2086
        apt-get -y -o Debug::NoLocking=1 -o Dir::Cache::Archives="${DEB_CACHE}" \
            --download-only --no-install-recommends install $deps >/dev/null 2>&1 || true
    done
    for f in "${DEB_CACHE}"/*.deb; do dpkg-deb -x "$f" "${OFLM_SYSROOT}/" 2>/dev/null; done
    # repair: copy system versioned lib if a dev symlink dangles into it
    for link in "${OFLM_SYSROOT}"/usr/lib/x86_64-linux-gnu/*.so; do
        [ -L "$link" ] || continue
        tgt=$(readlink "$link"); [ -e "$link" ] || [ -f "/usr/lib/x86_64-linux-gnu/$tgt" ] && cp -f "/usr/lib/x86_64-linux-gnu/$tgt" "$(dirname "$link")/" 2>/dev/null || true
    done
else
    say OK "sysroot present, skipping"
fi
# BoostConfig shim (Ubuntu 26.04 ships no monolithic Boost config)
SHIM="${OFLM_SYSROOT}/usr/lib/x86_64-linux-gnu/cmake/Boost/BoostConfig.cmake"
if [ ! -f "${SHIM}" ]; then
    mkdir -p "$(dirname "${SHIM}")"
    sed "s|/home/user/.local/sysroot|${OFLM_SYSROOT}|g" "${SCRIPT_DIR}/BoostConfig.cmake.shim" > "${SHIM}" 2>/dev/null || cat > "${SHIM}" <<EOF
set(Boost_FOUND TRUE)
set(Boost_VERSION "1.90.0")
set(Boost_INCLUDE_DIRS "${OFLM_SYSROOT}/usr/include")
set(Boost_LIBRARY_DIRS "${OFLM_SYSROOT}/usr/lib/x86_64-linux-gnu")
if(NOT TARGET Boost::program_options)
  add_library(Boost::program_options UNKNOWN IMPORTED)
  set_target_properties(Boost::program_options PROPERTIES
    IMPORTED_LOCATION "${OFLM_SYSROOT}/usr/lib/x86_64-linux-gnu/libboost_program_options.so"
    INTERFACE_INCLUDE_DIRECTORIES "${OFLM_SYSROOT}/usr/include")
endif()
set(Boost_program_options_FOUND TRUE)
EOF
fi

# ---- 2. rustup ----------------------------------------------------------------
# `command -v cargo` is not enough: a rustup shim exists before a toolchain is
# installed, and then every `cargo` call fails with "could not choose a version".
# Probe the toolchain itself.
cargo_works() { command -v cargo >/dev/null && cargo --version >/dev/null 2>&1; }
if ! cargo_works; then
    say INFO "installing rustup + stable toolchain (user-level)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
    export PATH="${HOME}/.cargo/bin:${PATH}"
    cargo_works || cargo rustup default stable
    cargo_works || { say ERROR "cargo installed but no usable toolchain. Try: rustup default stable"; exit 1; }
else
    say OK "cargo present, skipping"
fi

# ---- 3. aiebu (provides aiebu-asm) -------------------------------------------
if [ ! -x "${HOME}/.local/bin/aiebu-asm" ]; then
    say INFO "building aiebu from source..."
    [ -d "${OFLM_TOOLS}/aiebu" ] || git clone --depth 1 https://github.com/Xilinx/aiebu.git "${OFLM_TOOLS}/aiebu"
    (cd "${OFLM_TOOLS}/aiebu" && git submodule update --init --recursive >/dev/null 2>&1 || true)
    (cd "${OFLM_TOOLS}/aiebu/build" && rm -rf Release && cmake -B Release -DCMAKE_BUILD_TYPE=Release .. >/dev/null && cmake --build Release -j"$(nproc)" >/dev/null)
    ln -sf "${OFLM_TOOLS}/aiebu/build/Release/src/cpp/utils/asm/aiebu-asm" "${HOME}/.local/bin/aiebu-asm"
else
    say OK "aiebu-asm present, skipping"
fi
[ -x "${HOME}/.local/bin/xclbinutil" ] || ln -sf "${OFLM_SYSROOT}/usr/bin/xclbinutil" "${HOME}/.local/bin/xclbinutil"

# ---- 4. XRT headers (compile against; runtime comes from FLM below) ----------
if [ ! -f "${OFLM_TOOLS}/XRT/src/runtime_src/core/include/xrt/xrt_bo.h" ]; then
    say INFO "fetching XRT headers (sparse)..."
    git clone --depth 1 --filter=blob:none --sparse https://github.com/Xilinx/XRT.git "${OFLM_TOOLS}/XRT"
    (cd "${OFLM_TOOLS}/XRT" && git sparse-checkout set src/runtime_src/core/include)
else
    say OK "XRT headers present, skipping"
fi

# ---- 5. FLM tarball (consistent XRT 2.25 runtime lib set) ---------------------
if [ ! -x "${FLM_ROOT}/flm" ]; then
    say INFO "downloading FastFlowLM ${FLM_VERSION} (XRT runtime source)..."
    mkdir -p "${FLM_ROOT}"
    TB="${HOME}/.local/fastflowlm_${FLM_VERSION}_linux.tar.gz"
    URL="https://github.com/ROCm/FastFlowLM/releases/download/${FLM_VERSION}/fastflowlm_${FLM_VERSION}_linux.tar.gz"
    # GitHub's asset CDN intermittently answers with a 9-byte "Not Found" body;
    # verify the archive before extracting, and retry rather than untarring it.
    ok=0
    for attempt in 1 2 3 4; do
        if [ -f "${TB}" ] && tar tzf "${TB}" >/dev/null 2>&1; then ok=1; break; fi
        rm -f "${TB}"
        [ "${attempt}" -gt 1 ] && sleep $((attempt * 10))
        say INFO "  download attempt ${attempt}/4..."
        curl -sSL --fail --max-time 900 --retry 1 --retry-delay 10 "${URL}" -o "${TB}" || true
    done
    if [ "${ok}" -ne 1 ]; then
        rm -f "${TB}"
        say ERROR "could not download a valid tarball from ${URL}"
        say ERROR "Fetch it manually to ${TB} and re-run (it must be a real .tar.gz, not a 'Not Found' body)."
        exit 1
    fi
    tar xzf "${TB}" -C "${FLM_ROOT}"
else
    say OK "FLM runtime present, skipping"
fi

# ---- 6. OpenFlowLM-Next: clone, configure, build `oflm` -----------------------
if [ ! -d "${OFLM_SRC}/open_kernels" ]; then
    say INFO "cloning OpenFlowLM-Next @ ${OFLM_SRC_REF:0:12} (pinned)"
    git clone --recursive https://github.com/Atomic-Germ/OpenFlowLM-Next "${OFLM_SRC}"
    (cd "${OFLM_SRC}" && git fetch --depth 1 origin "${OFLM_SRC_REF}" && git checkout --detach FETCH_HEAD)
fi
if [ -d "${OFLM_SRC}/.git" ]; then
    have="$(git -C "${OFLM_SRC}" rev-parse HEAD)"
    if [ "${have}" != "${OFLM_SRC_REF}" ]; then
        say WARN "OpenFlowLM-Next is at ${have:0:12}, pinned ref is ${OFLM_SRC_REF:0:12}"
        say WARN "Set OFLM_SRC_REF=<sha> to move deliberately, or re-clone for the verified tree."
    fi
fi
if [ ! -x "${OFLM_BIN}" ]; then
    say INFO "configuring OpenFlowLM-Next..."
    # shellcheck disable=SC2086
    (cd "${OFLM_SRC}" && cmake --preset linux-default "${OFLM_CMAKE_FLAGS[@]}" >/dev/null)
    say INFO "building oflm (engine + tokenizers; kernels come in stage 7)..."
    (cd "${OFLM_SRC}/build" && cmake --build . --target oflm -j"$(nproc)")
else
    say OK "oflm binary present, skipping (rm it to rebuild)"
fi

# ---- 7. minicpm5-2b kernel set -------------------------------------------------
KSET="${OFLM_SRC}/src/xclbins/MiniCPM5-2B-NPU2/open_kernels/manifest.json"
if [ ! -f "${KSET}" ]; then
    # The IRON/mlir-aie toolchain lives in its own venv (same one upstream's
    # export-kernels.py would create). We create it here because that script
    # also exports the open_npue BERT sets, which need a live NPU + pyxrt and
    # are not needed for this model.
    IRON_PY="${OFLM_SRC}/ironvenv/bin/python"
    if [ ! -x "${IRON_PY}" ]; then
        say INFO "creating ironvenv (mlir-aie 1.4.2 + Peano, ~1 GB)..."
        if command -v uv >/dev/null; then
            uv venv "${OFLM_SRC}/ironvenv"
            uv pip install --python "${IRON_PY}" -r "${OFLM_SRC}/ironvenv-requirements.txt"
        else
            python3 -m venv "${OFLM_SRC}/ironvenv"
            "${IRON_PY}" -m pip install --quiet --upgrade pip
            "${IRON_PY}" -m pip install -r "${OFLM_SRC}/ironvenv-requirements.txt"
        fi
    fi
    say INFO "exporting minicpm5-2b kernels (mlir-aie/Peano, several minutes)..."
    (cd "${OFLM_SRC}" && "${IRON_PY}" open_kernels/export_qwen36_kernels.py --spec open_kernels/recipes/specs/minicpm5-2b.json)
else
    say OK "kernel set present, skipping"
fi

# ---- 8. XRT runtime mirror + registration -------------------------------------
if [ ! -f "${XRT_ROOT}/lib/x86_64-linux-gnu/libxrt_core.so.2" ]; then
    say INFO "building XRT runtime mirror at ${XRT_ROOT}..."
    mkdir -p "${XRT_ROOT}/lib/x86_64-linux-gnu"
    for f in "${FLM_ROOT}"/lib/libxrt*; do ln -sf "$f" "${XRT_ROOT}/lib/"; done
    for f in "${XRT_ROOT}"/lib/libxrt*; do ln -sf "$f" "${XRT_ROOT}/lib/x86_64-linux-gnu/"; done
else
    say OK "XRT mirror present, skipping"
fi
if "${OFLM_BIN}" list 2>/dev/null | grep -q "minicpm5:2b"; then
    say OK "minicpm5:2b already registered"
else
    [ -f "${MODEL_DIR}/model.q4nx" ] || { say ERROR "no container at ${MODEL_DIR} — run build_open_weights.sh first (or set MODEL_DIR)"; exit 1; }
    say INFO "registering ${MODEL_TAG}..."
    "${OFLM_VENV}/bin/python" "${OFLM_SRC}/utilities/oflm-add/oflm-add.py" "${MODEL_DIR}" \
        --tag "${MODEL_TAG}" --family llama3 \
        --config "${OFLM_CONFIG_PATH}" \
        --models-root "${OFLM_MODEL_PATH}" \
        --xclbin-dir "${OFLM_XCLBIN_PATH}/xclbins" \
        --system-list "${OFLM_SRC}/src/model_list.json" \
        --open-kernels "${OFLM_SRC}/src/xclbins/MiniCPM5-2B-NPU2/open_kernels" --force
fi

say OK "setup complete."
echo "Serve with:  ./scripts/run_oflm.sh ${MODEL_TAG} 8001"
echo "Note: decode needs OPEN_KERNELS_UNVALIDATED=1 (set by run_oflm.sh + oflm_env.sh)."
