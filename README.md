# MiniCPM5-2B on AMD XDNA 2 NPU (Strix Halo)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-julianmb%2FMiniCPM5--2B--NPU2-blue)](https://huggingface.co/julianmb/MiniCPM5-2B-NPU2)
[![Hardware](https://img.shields.io/badge/Hardware-AMD_XDNA_2_NPU-red)](https://github.com/julianmb/npuhalo)
[![FastFlowLM](https://img.shields.io/badge/Runtime-FastFlowLM_%E2%89%A5_v0.9.22-green)](https://github.com/ROCm/FastFlowLM)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

This repository contains the end-to-end porting pipeline, architectural adaptations, FastFlowLM configurations, and diagnostic/reproduction harnesses for running **[openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)** on the **AMD XDNA 2 NPU** (`/dev/accel/accel0`, 48 AIE-ML tiles) on **AMD Strix Halo (Ryzen AI Max+ 395)**.

Pre-quantized AMD Q4NX weights, precompiled XCLBIN firmware, and verified tokenizers are hosted on Hugging Face:
👉 **[huggingface.co/julianmb/MiniCPM5-2B-NPU2](https://huggingface.co/julianmb/MiniCPM5-2B-NPU2)**

Part of the **[npuhalo](https://github.com/julianmb/npuhalo)** research initiative on AMD Strix Halo heterogeneous inference.

> ⚠️ **Current status (verified on FLM v1.0.6, Strix Halo, FW 1.1.2.65):**
> `flm serve minicpm5:2b` loads and **prefill runs on the NPU**, but **decode fails**
> with `runlist failed execution (ERT_CMD_STATE_NEW/TIMEOUT)`.
> The quickstart below gets you to a serving endpoint for reproduction — it does not
> yet yield tokens. See [Troubleshooting](#-troubleshooting) and the
> [open-kernel route](#-serving-via-open-kernels-community-reproduction-issue-1)
> (reported working, unverified here).

---

## 📖 Table of Contents
- [The Porting Story: Overcoming Hardware Constraints](#-the-porting-story-overcoming-hardware-constraints)
  - [1. The GQA Ratio Constraint (16:2 vs AIE Kernels)](#1-the-gqa-ratio-constraint-162-vs-aie-kernels)
  - [2. Mathematical KV Head Expansion (2 → 8 Heads)](#2-mathematical-kv-head-expansion-2--8-heads)
  - [3. QK-Norm Identity Injection for RMSNorm](#3-qk-norm-identity-injection-for-rmsnorm)
- [End-to-End Conversion Pipeline](#-end-to-end-conversion-pipeline)
- [Serving Pre-built Weights (Quickstart)](#-serving-pre-built-weights-quickstart)
- [Serving via Open Kernels (Community Reproduction, Issue #1)](#-serving-via-open-kernels-community-reproduction-issue-1)
- [The 42-Layer Runlist ERT Timeout Analysis (Closed Qwen3 Engine Only)](#-the-42-layer-runlist-ert-timeout-analysis-closed-qwen3-engine-only)
  - [Disassembly & Technical Root Cause](#disassembly--technical-root-cause)
  - [Correction on the `-noert` Build Flag](#correction-on-the--noert-build-flag)
  - [Path to Resolution](#path-to-resolution)
- [Reproduction & Diagnostic Harnesses](#-reproduction--diagnostic-harnesses)
- [Troubleshooting](#-troubleshooting)
- [Hardware & Software Profile](#-hardware--software-profile)

---

## 🧩 The Porting Story: Overcoming Hardware Constraints

MiniCPM5-2B is a 42-layer transformer model with hidden size $d_{model} = 2048$, intermediate size $d_{ffn} = 6144$, and attention head dimension $d_{head} = 128$. Attempting to execute stock MiniCPM5-2B on FastFlowLM encounters two critical hardware/firmware barriers:

### 1. The GQA Ratio Constraint (16:2 vs AIE Kernels)
MiniCPM5-2B has 16 Query heads and 2 Key/Value heads ($16:2 = 8:1$ GQA ratio).
Disassembly of FastFlowLM's multi-head attention AIE kernel library (`libmha.so`) revealed that compiled XDNA 2 kernels strictly support:
- `_gen_mha_seq_d64_q4` ($d_{head}=64$, $4:1$ ratio)
- `_gen_mha_seq_d128_q2` ($d_{head}=128$, $2:1$ ratio)
- `_gen_mha_seq_d128_q3` ($d_{head}=128$, $3:1$ ratio)
- `_gen_mha_seq_d128_q4` ($d_{head}=128$, $4:1$ ratio)

There is **no compiled 8:1 AIE kernel** for $d_{head}=128$. Additionally, FastFlowLM's Llama engine (`libllama_npu.so`) hardcodes `hidden_size == 2048` to $d_{head}=64$.

### 2. Mathematical KV Head Expansion (2 → 8 Heads)
In Grouped Query Attention:
$$\text{Attention}(Q_i, K_{g(i)}, V_{g(i)})$$
Where $g(i) = \lfloor i / (H_q / H_{kv}) \rfloor$. For $16:2$:
- Query heads $Q_{0..7}$ attend to KV head $0$.
- Query heads $Q_{8..15}$ attend to KV head $1$.

By replicating KV head 0 four times into indices $[0, 1, 2, 3]$ and KV head 1 four times into indices $[4, 5, 6, 7]$:
- $Q_{0,1} \to K_0$, $Q_{2,3} \to K_1$, $Q_{4,5} \to K_2$, etc.
- Because the duplicated heads hold identical weights, the resulting dot-product attention distributions, softmax probabilities, and output projections are **bit-for-bit mathematically identical**.
- The adapted model has $H_q = 16, H_{kv} = 8$ ($16:8 = 2:1$ GQA ratio), natively mapping to FastFlowLM's compiled `_gen_mha_seq_d128_q2` kernel!

Implementation: [`scripts/expand_kv_heads.py`](scripts/expand_kv_heads.py).

### 3. QK-Norm Loader Shim (Not an Identity) for RMSNorm
FastFlowLM's Qwen3 engine (`libqwen3_npu.so`) automatically selects the `_gen_mha_seq_d128_q2` kernel when $d_{head} = 128$ and $d_{ffn} = 6144$. However, `libqwen3_npu.so` expects QK-normalization weights:
`model.layers.{i}.self_attn.q_norm.weight` and `k_norm.weight` ($[128]$ BF16).

Since MiniCPM5-2B does not employ QK-normalization, [`scripts/inject_qk_norm.py`](scripts/inject_qk_norm.py) injects synthetic unit tensors ($\gamma = 1.0$) into `model.q4nx` across all 42 layers:
$$\text{RMSNorm}(x, \gamma = 1.0) = \frac{x}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2 + \epsilon}} \cdot 1.0$$
This satisfies the weight loader but does **not** preserve the original computation: unit scale still normalizes Q and K, so attention scores differ from the un-normalized MiniCPM5 architecture. Output equivalence of this port is **unvalidated**.

---

## 🛠️ End-to-End Conversion Pipeline

If you want to adapt and convert from the original Hugging Face weights from scratch:

```bash
# 1. Clone original weights
git clone https://huggingface.co/openbmb/MiniCPM5-2B hf_raw/

# 2. Expand KV heads from 2 to 8 (16:2 -> 16:8 GQA)
python3 scripts/expand_kv_heads.py --src hf_raw/ --dst hf_adapted/ --target-kv-heads 8

# 3. Convert adapted HF model to GGUF (use Q4_1, NOT Q4_0)
# Per issue #1: Q4_0 -> Q4NX verifies at per-tensor cosine ~-0.4 (garbage output),
# while Q4_1 -> Q4NX verifies at cosine 0.999992 and generates coherent text.
python3 llama.cpp/convert_hf_to_gguf.py hf_adapted/ --outfile minicpm5_2b_gqa8_q4_1.gguf --outtype q4_1

# 4. Convert GGUF to AMD Q4NX block format using FastFlowLM converter
git clone https://github.com/ROCm/FLM_Q4NX_Converter.git
python3 FLM_Q4NX_Converter/convert.py -i minicpm5_2b_gqa8_q4_1.gguf -o output/ -f qwen3

# 5. Closed-engine only: inject synthetic QK-norm loader-shim weights (unit scale; NOT numerically identical)
# Skip this step for the open-kernel route below (llama3 spec, qk_norm=false).
python3 scripts/inject_qk_norm.py output/model.q4nx --layers 42 --head-dim 128
```

---

## 🚀 Serving Pre-built Weights (Quickstart)

### 0. Prerequisites (one time, root)

```bash
# FLM mmaps ~2.4 GB with MAP_LOCKED; the default 8 MB cap kills serve with mmap err=-11.
sudo sh -c 'printf "* soft memlock unlimited\n* hard memlock unlimited\n" >> /etc/security/limits.conf'
# then fully log out/in (or reboot) and verify:
ulimit -l   # must print: unlimited
```

You also need `/dev/accel/accel0` (amdxdna driver) — check with `./scripts/run_flm.sh` or `flm validate`.

### 1. Automated setup (recommended)

```bash
./scripts/setup_flm.sh
```

This downloads the FLM Linux tarball to `~/.local/flm<ver>` (persistent across
reboots, unlike `/tmp`), fetches the prebuilt weights to
`~/.config/flm/models/MiniCPM5-2B-NPU2`, registers `minicpm5:2b` in the FLM
install's own `model_list.json`, and copies the AIE kernels into place.

### 2. Manual setup (if the script doesn't fit your layout)

```bash
mkdir -p ~/.config/flm/models
git clone https://huggingface.co/julianmb/MiniCPM5-2B-NPU2 ~/.config/flm/models/MiniCPM5-2B-NPU2
```

FLM ≥ 1.0.x reads the `model_list.json` next to its own binary (not
`~/.config/flm/model_list.json`), so merge the entry there and wire up the
weights + kernels relative to your install root (`FLM_ROOT`):

```bash
FLM_ROOT="${HOME}/.local/flm106"   # adjust to your install
python3 -c "
import json
base = json.load(open('${FLM_ROOT}/model_list.json'))
entry = json.load(open('configs/model_list_entry.json'))
base.setdefault('models', {}).update(entry['models'])
json.dump(base, open('${FLM_ROOT}/model_list.json', 'w'), indent=2)
"
mkdir -p "${FLM_ROOT}/models" "${FLM_ROOT}/xclbins/MiniCPM5-2B-NPU2"
ln -sfn ~/.config/flm/models/MiniCPM5-2B-NPU2 "${FLM_ROOT}/models/MiniCPM5-2B-NPU2"
cp ~/.config/flm/models/MiniCPM5-2B-NPU2/*.xclbin "${FLM_ROOT}/xclbins/MiniCPM5-2B-NPU2/"
```

### 3. Launch FastFlowLM Server
```bash
./scripts/run_flm.sh minicpm5:2b 8001
```
(`run_flm.sh` auto-detects installs under `~/.local/flm*`; override with `FLM_DIR=...`.)
Verify registration first with `flm list | grep minicpm` — you want `minicpm5:2b ✅`.

Note: this quickstart and `configs/model_list_entry.json` (`quantization_level: Q4_1`,
`family: qwen3`) target the closed FastFlowLM Qwen3 engine, which requires the QK-norm
shim (already baked into the hosted `model.q4nx`) and currently fails at decode with
the 42-layer runlist ERT error below. Prefill succeeds; no tokens are produced.

---

## 🧪 Serving via Open Kernels (Verified Working, Issue #1)

> Reported by [@D-revv](https://github.com/D-revv) in
> [#1](https://github.com/julianmb/minicpm5-xdna2/issues/1) and **independently
> verified on Strix Halo (Ryzen AI Max+ 395, FW 1.1.2.65)**: 2+2→4, 25\*14→350,
> Paris — prefill ~26–32 tok/s, **decode ~27 tok/s**, 42/42 layers resident.
> The closed-engine ERT analysis below does not apply to this route.

Recipe (all steps reproduced here):
1. Start from this repo's `scripts/expand_kv_heads.py` output (16:2 → 16:8 GQA).
2. Export via BF16 GGUF → Q4NX with the Atomic-Germ `FLM_Q4NX_Converter`
   (`-f llama`, Q4_1). Do **not** use Q4_0 (per-tensor cosine ~-0.4, garbage output;
   Q4_1 path is clean). No QK-norm shim — the container must not contain
   `q_norm`/`k_norm` tensors.
3. Build the open stack from source (`Atomic-Germ/OpenFlowLM-Next @ main`):
   `oflm` engine + `minicpm5-2b` kernel set (`-DOFLM_KERNEL_SPECS=minicpm5-2b`),
   using the checked-in `open_kernels/recipes/specs/minicpm5-2b.json`
   (**llama3 spec, `qk_norm=false`**, attention tuple `(128,16,8,128)`,
   `OPEN_KERNELS_UNVALIDATED=1`).
4. Register with `oflm-add <model-dir> --tag minicpm5:2b --family llama3
   --open-kernels <built set>` and serve with `oflm serve minicpm5:2b`.
   No Llama→Qwen3 relabel is needed on this route, so the README's
   output-equivalence caveat for the shim does not apply there.

Known cosmetic quirks: the model wraps answers in `<think>` chatter and repeats
`<|im_end|>` instead of stopping; bound it with `max_tokens` in serve mode.

---

## 🔍 The 42-Layer Runlist ERT Timeout Analysis (Closed Qwen3 Engine Only)

While prefill runs and outputs tokens on the closed FastFlowLM Qwen3 engine (`libqwen3_npu.so`),
sustained autoregressive decode currently hits `ERT_CMD_STATE_TIMEOUT`.
This does not apply to the open-kernel route above (issue #1).

### Disassembly & Technical Root Cause
Disassembly of FastFlowLM's causal LM decode implementation (`libqwen3_npu.so`):
```asm
qwen3_npu::Impl::forward(int):
  ...
  call 24510 <xrt::runlist::execute()@plt>
  mov  %r12, %rsi
  mov  %r14, %rdi
  call 24710 <xrt::runlist::wait(std::chrono::duration<long, std::ratio<1l, 1000l> > const&) const@plt>
```

1. **Prefill (`_prefill_with_mv`) Succeeds:**
   ```text
   [FLM]  Start prefill...
   [FLM]  Prefill chunk 1/1 with 38 tokens
   [FLM]  Creating checkpoint at context length 38
   ```
   Prefill chunks matrix-vector computations layer-by-layer or chunk-by-chunk using `mm.xclbin`.

2. **Decode (`Start generating...`) Stalls:**
   In `libqwen3_npu.so`, decode constructs a single chained `xrt::runlist` encompassing the forward operations for **all layers in the network**.
   - MiniCPM5-2B has **42 layers** and borrows `layer.xclbin` and `attn.xclbin` from `Qwen3-1.7B` (which was compiled for **28 layers**).
   - Dispatching 42 sequential layer executions in a single synchronous runlist batch overruns the hardware command processor buffer depth / timeout threshold of the AIE firmware (`1.1.2.65`) and Linux `amdxdna 0.7` driver.
   - `xrt::runlist::wait()` times out, returning:
     ```json
     {"error":"runlist failed execution (ERT_CMD_STATE_TIMEOUT)"}
     ```

### Correction on the `-noert` Build Flag
Initial speculation was that compiling XRT with `-noert` would bypass the timeout. Community testing by [@Platano78](https://github.com/Platano78) established that:
```text
[-noert]   Do not treat missing ERT FW as a build error
[-npu]     Build for NPU only, implies -noert and disables bundling of Alveo Linux drivers
```
`[-noert]` is strictly a build-time tolerance flag for omitting PCIe Alveo ERT firmware blobs during compilation (and is automatically implied by `[-npu]`). It does not alter the runtime NPU command submission queue or bypass the hardware watchdog.

### Path to Resolution
1. **Multi-Chunk Runlist in `libqwen3_npu.so`:**
   Split the 42-layer forward sequence into two batches of 21 layers (e.g. `runlist_1.execute()` $\to$ `wait()` $\to$ `runlist_2.execute()` $\to$ `wait()`).
2. **Dedicated 42-Layer AIE Kernel Compilation:**
   Compile native `layer.xclbin` and `attn.xclbin` graphs calibrated for 42-layer execution depth.

Until one of these upstream updates lands, 24–28 layer models (`Qwen3-1.7B`, `Qwen3.5-0.8B`, `Llama-3.2-1B`) represent the verified operational ceiling on AMD Strix Halo NPU.

---

## 🔬 Reproduction & Diagnostic Harnesses

### 1. Minimal ERT Timeout Reproducer
Verifies prefill success vs decode timeout against a running FLM instance in under 10 seconds:
```bash
python3 scripts/reproduce_ert_timeout.py --url http://127.0.0.1:8001
```

### 2. Multi-Domain Output Quality Test
```bash
python3 scripts/test_quality.py --url http://127.0.0.1:8001
```

### 3. Speculative Decoding Benchmark (NPU Drafter + iGPU Target)
```bash
python3 scripts/benchmark_speculative.py --npu-url http://127.0.0.1:8001 --gpu-url http://127.0.0.1:8012
```

---

## 🩺 Troubleshooting

All of the below were hit while validating this repo on Strix Halo (FLM v1.0.6, FW 1.1.2.65).

| Symptom | Cause | Fix |
|---|---|---|
| `Model not found: minicpm5:2b` | FLM ≥ 1.0.x reads the `model_list.json` next to its own binary, **not** `~/.config/flm/model_list.json` as older docs said | Run `./scripts/setup_flm.sh`, or merge `configs/model_list_entry.json` into `<FLM_ROOT>/model_list.json` (see quickstart step 2) |
| `mmap(...) failed (err=-11): Resource temporarily unavailable` | memlock capped (default 8 MB); FLM needs ~2.4 GB `MAP_LOCKED` | `limits.conf` memlock unlimited + full re-login/reboot (quickstart step 0); verify `ulimit -l` prints `unlimited` |
| `flm check` → `[json.exception.type_error.302] type must be string, but is null` | Over-strict file verification in `flm check` | Benign as far as we can tell — `flm serve` proceeds past it. Don't chase this; check serve instead |
| `{"error":"runlist failed execution (ERT_CMD_STATE_NEW/TIMEOUT)"}`, prefill OK, no tokens | 42-layer decode chained in one `xrt::runlist` on the closed Qwen3 engine — the known blocker, reproduced on v1.0.6 | No repo-side fix; track the ERT analysis section / issue #1's open-kernel route |
| `{"error":"qds_device::wait() unexpected command state"}` (seen on longer ~236-token prompts) | Same decode path, different failure surface | Same as above — decode is blocked regardless of prompt length |
| `bind: Address already in use` on serve | A previous `flm serve` still holds the port | Kill the old server (`pkill -f 'flm serve'`) or use another `--port` |
| Runtime gone after reboot | Install was under `/tmp` (tmpfs) | Install to `~/.local/flm*` — `setup_flm.sh` does this by default |

---

## 💻 Hardware & Software Profile

- **System:** AMD Ryzen AI Max+ 395 (16 Zen 5 cores, 32 threads)
- **iGPU:** AMD Radeon 8060S (40 CU `gfx1151`, ROCm / Vulkan)
- **NPU:** AMD XDNA 2 (`/dev/accel/accel0`, 48 AIE2p tiles, 50 TOPS)
- **Memory:** 128 GB LPDDR5X-8000 Unified Memory (~273 GB/s)
- **NPU Firmware:** `amdnpu/17f0_11/npu.sbin` (v1.1.2.65)
- **Kernel & Driver:** Linux 7.0.0-31-generic with in-tree `amdxdna` 0.7.0
- **Boot Parameters:** `iommu=pt iommu.passthrough=0`
- **FastFlowLM Version:** v1.0.6 (verified: prefill OK, decode ERT-blocked); v1.0.2 / v1.0.4 per earlier notes

---

## 📄 License & Citations
- Base Model weights and architecture: [OpenBMB Apache 2.0](https://github.com/OpenBMB/MiniCPM)
- Code and configurations in this repository: [Apache 2.0](LICENSE)
- FastFlowLM runtime: [ROCm FastFlowLM](https://github.com/ROCm/FastFlowLM)
- Research and benchmark records: [julianmb/npuhalo](https://github.com/julianmb/npuhalo)
