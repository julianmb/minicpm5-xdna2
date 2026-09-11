# MiniCPM5-2B on AMD XDNA 2 NPU (Strix Halo)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-julianmb%2FMiniCPM5--2B--NPU2-blue)](https://huggingface.co/julianmb/MiniCPM5-2B-NPU2)
[![Hardware](https://img.shields.io/badge/Hardware-AMD_XDNA_2_NPU-red)](https://github.com/julianmb/npuhalo)
[![FastFlowLM](https://img.shields.io/badge/Runtime-FastFlowLM_%E2%89%A5_v0.9.22-green)](https://github.com/ROCm/FastFlowLM)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

This repository contains the end-to-end porting pipeline, architectural adaptations, FastFlowLM configurations, and diagnostic/reproduction harnesses for running **[openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)** on the **AMD XDNA 2 NPU** (`/dev/accel/accel0`, 48 AIE-ML tiles) on **AMD Strix Halo (Ryzen AI Max+ 395)**.

Pre-quantized AMD Q4NX weights, precompiled XCLBIN firmware, and verified tokenizers are hosted on Hugging Face:
👉 **[huggingface.co/julianmb/MiniCPM5-2B-NPU2](https://huggingface.co/julianmb/MiniCPM5-2B-NPU2)**

Part of the **[npuhalo](https://github.com/julianmb/npuhalo)** research initiative on AMD Strix Halo heterogeneous inference.

---

## 📖 Table of Contents
- [The Porting Story: Overcoming Hardware Constraints](#-the-porting-story-overcoming-hardware-constraints)
  - [1. The GQA Ratio Constraint (16:2 vs AIE Kernels)](#1-the-gqa-ratio-constraint-162-vs-aie-kernels)
  - [2. Mathematical KV Head Expansion (2 → 8 Heads)](#2-mathematical-kv-head-expansion-2--8-heads)
  - [3. QK-Norm Identity Injection for RMSNorm](#3-qk-norm-identity-injection-for-rmsnorm)
- [End-to-End Conversion Pipeline](#-end-to-end-conversion-pipeline)
- [Serving Pre-built Weights (Quickstart)](#-serving-pre-built-weights-quickstart)
- [The 42-Layer Runlist ERT Timeout Analysis](#-the-42-layer-runlist-ert-timeout-analysis)
  - [Disassembly & Technical Root Cause](#disassembly--technical-root-cause)
  - [Correction on the `-noert` Build Flag](#correction-on-the--noert-build-flag)
  - [Path to Resolution](#path-to-resolution)
- [Reproduction & Diagnostic Harnesses](#-reproduction--diagnostic-harnesses)
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

### 3. QK-Norm Identity Injection for RMSNorm
FastFlowLM's Qwen3 engine (`libqwen3_npu.so`) automatically selects the `_gen_mha_seq_d128_q2` kernel when $d_{head} = 128$ and $d_{ffn} = 6144$. However, `libqwen3_npu.so` expects QK-normalization weights:
`model.layers.{i}.self_attn.q_norm.weight` and `k_norm.weight` ($[128]$ BF16).

Since MiniCPM5-2B does not employ QK-normalization, [`scripts/inject_qk_norm.py`](scripts/inject_qk_norm.py) injects synthetic unit tensors ($\gamma = 1.0$) into `model.q4nx` across all 42 layers:
$$\text{RMSNorm}(x, \gamma = 1.0) = \frac{x}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2 + \epsilon}} \cdot 1.0$$
This satisfies the engine loader while preserving numerical precision.

---

## 🛠️ End-to-End Conversion Pipeline

If you want to adapt and convert from the original Hugging Face weights from scratch:

```bash
# 1. Clone original weights
git clone https://huggingface.co/openbmb/MiniCPM5-2B hf_raw/

# 2. Expand KV heads from 2 to 8 (16:2 -> 16:8 GQA)
python3 scripts/expand_kv_heads.py --src hf_raw/ --dst hf_adapted/ --target-kv-heads 8

# 3. Convert adapted HF model to GGUF
python3 llama.cpp/convert_hf_to_gguf.py hf_adapted/ --outfile minicpm5_2b_gqa8_q4_0.gguf --outtype q4_0

# 4. Convert GGUF to AMD Q4NX block format using FastFlowLM converter
git clone https://github.com/ROCm/FLM_Q4NX_Converter.git
python3 FLM_Q4NX_Converter/convert.py -i minicpm5_2b_gqa8_q4_0.gguf -o output/ -f qwen3

# 5. Inject synthetic QK-norm identity weights
python3 scripts/inject_qk_norm.py output/model.q4nx --layers 42 --head-dim 128
```

---

## 🚀 Serving Pre-built Weights (Quickstart)

### 1. Download Pre-converted Model & Kernels
```bash
mkdir -p ~/.config/flm/models
git clone https://huggingface.co/julianmb/MiniCPM5-2B-NPU2 ~/.config/flm/models/MiniCPM5-2B-NPU2
```

### 2. Copy AIE Kernels
FastFlowLM requires compiled `.xclbin` files in its `xclbins/` directory:
```bash
FLM_ROOT="${HOME}/.config/flm"
mkdir -p "${FLM_ROOT}/xclbins/MiniCPM5-2B-NPU2"
cp ~/.config/flm/models/MiniCPM5-2B-NPU2/*.xclbin "${FLM_ROOT}/xclbins/MiniCPM5-2B-NPU2/"
```

### 3. Register in `model_list.json`
Add the configuration from [`configs/model_list_entry.json`](configs/model_list_entry.json) to `~/.config/flm/model_list.json`.

### 4. Launch FastFlowLM Server
```bash
./scripts/run_flm.sh minicpm5:2b 8001
```

---

## 🔍 The 42-Layer Runlist ERT Timeout Analysis

While prefill runs and outputs tokens, sustained autoregressive decode currently hits `ERT_CMD_STATE_TIMEOUT`.

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

### Community Reproductions

Independent confirmations tracked in [ROCm/FastFlowLM#712](https://github.com/ROCm/FastFlowLM/issues/712):

**Per-engine depth boundary ([@Platano78](https://github.com/Platano78), same box / firmware / fresh server):**

| Model | Layers | Engine | Decode |
|---|---|---|---|
| `qwen3:1.7b` | 28 | qwen3 | 43.3 t/s |
| `qwen3:4b` | **36** | **qwen3** | **19.8 t/s** |
| `qwen3.6-moe:35b-a3b` | 40 | qwen3.6-moe | 16.8 t/s |
| `gemma4-it:e4b` | **42** | gemma4e | 12.7 t/s |
| `minicpm5:2b` | 42 | **qwen3** | fails |

The `qwen3` engine is fine at 36 layers and fails at 42, while other engines handle 40–42. The boundary is **engine-specific (37–42)**, not a Strix Halo or firmware limit. Substituting the 36-, 40-, and 42-layer `layer.xclbin` binaries all fail identically, so it is not kernel geometry either.

**Containerized / out-of-tree confirmation ([@D-revv](https://github.com/D-revv), Docker in Proxmox LXC, `/dev/accel/accel0` passthrough):** identical timeout with `txn_op_idx = 0xFFFFFFFF`. Ruled out independently:

| Variable | Tested | Result |
|---|---|---|
| Kernel driver | in-tree `amdxdna` 0.7 AND out-of-tree v0.17 (`xdna-driver` main) | timeout |
| NPU firmware | `npu.dev.sbin` AND `npu_7.sbin` (1.1.2.65) | timeout |
| `force_cmdlist` | 0 and 1 | timeout |
| `tdr_timeout_ms` | 60000 (watchdog fires at 60 s, job never completes) | timeout |
| XRT `-noert` | built with it | no effect |

The 60 s TDR result matters: the decode job never retires, so this is a stuck/invalid monolithic submission, not merely a slow one.

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

## 💻 Hardware & Software Profile

- **System:** AMD Ryzen AI Max+ 395 (16 Zen 5 cores, 32 threads)
- **iGPU:** AMD Radeon 8060S (40 CU `gfx1151`, ROCm / Vulkan)
- **NPU:** AMD XDNA 2 (`/dev/accel/accel0`, 48 AIE2p tiles, 50 TOPS)
- **Memory:** 128 GB LPDDR5X-8000 Unified Memory (~273 GB/s)
- **NPU Firmware:** `amdnpu/17f0_11/npu.sbin` (v1.1.2.65)
- **Kernel & Driver:** Linux 7.0.0-31-generic with in-tree `amdxdna` 0.7.0
- **Boot Parameters:** `iommu=pt iommu.passthrough=0`
- **FastFlowLM Version:** v1.0.2 / v1.0.4

---

## 📄 License & Citations
- Base Model weights and architecture: [OpenBMB Apache 2.0](https://github.com/OpenBMB/MiniCPM)
- Code and configurations in this repository: [Apache 2.0](LICENSE)
- FastFlowLM runtime: [ROCm FastFlowLM](https://github.com/ROCm/FastFlowLM)
- Research and benchmark records: [julianmb/npuhalo](https://github.com/julianmb/npuhalo)
