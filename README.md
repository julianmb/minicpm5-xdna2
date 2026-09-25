# MiniCPM5-2B on AMD XDNA 2 NPU (Strix Halo)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-julianmb%2FMiniCPM5--2B--NPU2-blue)](https://huggingface.co/julianmb/MiniCPM5-2B-NPU2)
[![Hardware](https://img.shields.io/badge/Hardware-AMD_XDNA_2_NPU-red)](https://github.com/julianmb/npuhalo)
[![FastFlowLM](https://img.shields.io/badge/Runtime-FastFlowLM_%E2%89%A5_v0.9.22-green)](https://github.com/ROCm/FastFlowLM)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

This repository contains the end-to-end porting pipeline, architectural adaptations, runtime configurations, and diagnostic harnesses for running **[openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)** on the **AMD XDNA 2 NPU** (`/dev/accel/accel0`) on **AMD Strix Halo** (Ryzen AI Max+ 395) and other XDNA 2 parts.

## ✅ Status: it works

Verified on Strix Halo (Ryzen AI Max+ 395, NPU FW 1.1.2.65, amdxdna 0.7, Linux 7.0), running entirely on the NPU:

| | Result |
|---|---|
| Decode | **~7 tok/s sustained** (145–160 ms/token), 42/42 layers resident |
| Prefill | ~26–32 tok/s on an otherwise-idle NPU; ~7–8 tok/s if another NPU process is resident |
| Correctness | `2+2`→`4`, `25*14`→`350`, capital of France→`Paris` |
| Power | NPU-only; iGPU and CPU cores stay free |

This uses the **open-kernel engine** (`Atomic-Germ/OpenFlowLM-Next`). AMD's
closed FastFlowLM engine loads and prefills fine but **cannot decode** this
model (a runlist failure in the closed `libqwen3_npu.so`); that path is kept
here for reproduction only — see [Closed-Engine Decode Failure](#-the-closed-engine-decode-failure-analysis).

## 🚀 Quick start

```bash
git clone https://github.com/julianmb/minicpm5-xdna2
cd minicpm5-xdna2
```

**0. One root step (required).** The NPU runtime needs ~2.4 GB of locked
memory; the default 8 MB cap makes serving fail with `mmap err=-11`:

```bash
sudo sh -c 'printf "* soft memlock unlimited\n* hard memlock unlimited\n" >> /etc/security/limits.conf'
```

Then **log out and back in** (or reboot) — limits only apply to new sessions.
Confirm: `ulimit -l` must print `unlimited`.

**1. Build the weights** (~10 min; downloads ~4.7 GB, converts to Q4_1).
Creates its own Python venv on first run.

```bash
./scripts/build_open_weights.sh
```

**2. Build the engine + NPU kernels** (~15–30 min; compiles `oflm` and the
AIE kernels from source). No `sudo` needed — dependencies are staged into
`~/.local/sysroot`. Safe to re-run; finished stages are skipped.

```bash
./scripts/setup_oflm.sh
```

**3. Serve**

```bash
./scripts/run_oflm.sh minicpm5:2b 8001
```

**4. Ask it something**

```bash
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"minicpm5:2b","messages":[{"role":"user","content":"What is 25*14? Answer with just the number."}],
       "max_tokens":24,"temperature":0.0}'
```

> **No `max_tokens` needed.** Generation stops on `<|im_end|>` on its own
> (`finish_reason: "stop"`), because the build restores the full EOS id set —
> the Q4NX converter alone leaves `eos_token_id: [1]`, which would stream
> `<|im_end|>` until the cap. `build_open_weights.sh` repairs and verifies it.
> The sample above sets `max_tokens: 24` only as belt-and-braces.

**Requirements:** Linux with an XDNA 2 NPU (`/dev/accel/accel0`), plus
`cmake`, `g++`, `ninja`, `git`, `curl`, `python3`, and `apt-get` (used
download-only, never installing). Roughly 28 GB of disk and ~8 GB of
downloads for a full first run.

**Reproducibility:** the engine tree, FastFlowLM release and model revision are
pinned in `scripts/oflm_env.sh` (`OFLM_SRC_REF`, `FLM_VERSION`, `MODEL_REV`).
`setup_oflm.sh` warns if your checkout has drifted from the pinned commit —
that tree is what the verified numbers above were measured on. Override the
variables to move forward deliberately.

<details>
<summary>Stuck? Common failures</summary>

| Symptom | Fix |
|---|---|
| `mmap(...) failed (err=-11)` | memlock step 0 above, then re-login |
| `[ERROR] Memlock limit is too low` | same |
| `No such device with index '0'` | NPU not visible; check `ls /dev/accel/` |
| `Not a git repository` from a helper script | run the scripts from the repo root |
| Decode returns nothing / ERT error | you're on the closed engine — use `run_oflm.sh`, not `run_flm.sh` |

More in [Troubleshooting](#-troubleshooting).
</details>

---

Part of the **[npuhalo](https://github.com/julianmb/npuhalo)** research initiative on AMD Strix Halo heterogeneous inference.

<details>
<summary>Porting details, closed-engine quickstart, analysis and harnesses</summary>

## 📖 Table of Contents
- [The Porting Story: Overcoming Hardware Constraints](#-the-porting-story-overcoming-hardware-constraints)
  - [1. The GQA Ratio Constraint (16:2 vs AIE Kernels)](#1-the-gqa-ratio-constraint-162-vs-aie-kernels)
  - [2. Mathematical KV Head Expansion (2 → 8 Heads)](#2-mathematical-kv-head-expansion-2--8-heads)
  - [3. QK-Norm Identity Injection for RMSNorm](#3-qk-norm-identity-injection-for-rmsnorm)
- [End-to-End Conversion Pipeline](#-end-to-end-conversion-pipeline)
- [Serving Pre-built Weights (Quickstart)](#-serving-pre-built-weights-quickstart)
- [Serving via Open Kernels (Verified Working, Issue #1)](#-serving-via-open-kernels-verified-working-issue-1)
- [The Closed-Engine Decode Failure (Analysis)](#-the-closed-engine-decode-failure-analysis)
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
# Note: if you started from the prebuilt HF weights, the shim is ALREADY baked
# into model.q4nx (verified: 42 layers ship q_norm/k_norm), so step 5 is a
# no-op there. It only applies when converting from scratch as above.
python3 scripts/inject_qk_norm.py output/model.q4nx --layers 42 --head-dim 128
```

If you did not write step 3/4 yourself, just use the prebuilt container instead —
`./scripts/build_open_weights.sh` (open-kernel route) or the HF model repo
(closed-engine route, shim included).

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

> Independently verified on Strix Halo (Ryzen AI Max+ 395, FW 1.1.2.65):
> 2+2→4, 25\*14→350, Paris — prefill ~26–32 tok/s, **decode ~7 tok/s sustained**,
> 42/42 layers resident. The closed-engine ERT analysis below does not apply
> to this route. Original report by [@D-revv](https://github.com/D-revv) in
> [#1](https://github.com/julianmb/minicpm5-xdna2/issues/1).

### Why the closed-engine instructions failed

Anyone who followed the old quickstart hit, in order: (1) `Model not found`
(FLM ≥ 1.0 reads the `model_list.json` next to its binary, not
`~/.config/flm`); (2) `mmap err=-11` (memlock capped at 8 MB, needs
unlimited + re-login); and finally (3) prefill-OK / decode-dead
(`ERT_CMD_STATE_*`) — which no setup step can fix. The route below replaces
the closed engine entirely and scripts every step that was previously manual.

### 0. Prerequisites (one time, root)

Same memlock fix as the closed route (quickstart step 0 above), then re-login:
`ulimit -l` must print `unlimited`. All other dependencies are staged
**without sudo** by the setup script (apt download-only closure into
`~/.local/sysroot`, user-level rustup, source-built `aiebu`).

You need `cmake`, `g++`, `ninja`, `git`, `curl`, `python3` on PATH.

### 1. Build the weights (shim-free Q4NX container)

```bash
./scripts/build_open_weights.sh
# outputs ~/.cache/oflm-weights/MiniCPM5-2B-OFLM/model.q4nx (+tokenizer/config)
```

What it does: `expand_kv_heads.py` (16:2 → 16:8) → BF16 GGUF → Q4NX with
`-f llama` (Q4_1, **no** QK-norm shim — the open llama3 engine with
`qk_norm=false` must not see `q_norm`/`k_norm` tensors; the script verifies
their absence). Override with `SRC_MODEL=` (local dir or HF id),
`OUT_DIR=`, `FORCE=1`.

### 2. Build the engine + kernels (one command, idempotent)

```bash
./scripts/setup_oflm.sh
# uses OUT_DIR above via MODEL_DIR (default: the path from step 1)
```

This clones `Atomic-Germ/OpenFlowLM-Next`, builds `oflm` and the
`minicpm5-2b` kernel set (`dx`, `dx_attn`, `ln`, `lm_head_q4`, …) from the
checked-in `minicpm5-2b.json` spec, builds the XRT runtime mirror, and
registers the model (`oflm-add … --tag minicpm5:2b --family llama3
--open-kernels …`). Re-runs skip completed stages. Expect ~15–30 min total
on first run (engine compile + AIE kernel builds).

### 3. Serve

```bash
./scripts/run_oflm.sh minicpm5:2b 8001
```

Then inference as usual, e.g.:
```bash
curl -s http://127.0.0.1:8001/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"minicpm5:2b","messages":[{"role":"user","content":"What is 25*14? Answer with just the number."}],"max_tokens":30,"temperature":0.0}'
```

Known quirks: the model wraps answers in `<think>` chatter (stripped by the
server's `reasoning_content` split). Generation stops correctly on `<|im_end|>`
— see the EOS note above. `stop` and `stop_token_ids` are accepted by the API
but not implemented by this runtime, so they do not truncate early.

Sustained decode is **~7 tok/s** (145–160 ms/token), degrading slightly as
context grows because attention runs on the host in this kernel set. The first
token is much faster than the sustained rate, so short-answer benchmarks can
badly overestimate throughput — quote the sustained figure. Prefill is
contention-sensitive: a second NPU process (e.g. another `flm` service) drops
it from ~30 tok/s to ~7–8.

`OPEN_KERNELS_UNVALIDATED=1` is required (set by `oflm_env.sh`, sourced by
both scripts) for the `(128,16,8,128)` attention tuple.

---

## 🔍 The Closed-Engine Decode Failure (Analysis)

**If you only want to run the model, skip this section** — use the
[open-kernel route](#-serving-via-open-kernels-verified-working-issue-1) above.

**What is established (verified on Strix Halo, FLM v1.0.1–v1.0.6):**
prefill completes on the NPU; the first decode token fails with
`runlist failed execution (ERT_CMD_STATE_TIMEOUT / ERT_CMD_STATE_NEW)`, or
`qds_device::wait() unexpected command state` on longer prompts. Disassembly
shows decode dispatching through `xrt::runlist::execute()` → `wait()` in
`libqwen3_npu.so` (a closed binary; the public source has only the class
declaration, so the batching logic cannot be patched from outside).

**What is *not* established — earlier claims here were withdrawn:**
- *Not* the `-noert` build flag. It is a build-time tolerance flag for
  omitting Alveo ERT blobs and does not change runtime submission. Rebuilding
  XRT with it changes nothing.
- *Not* a driver/firmware/XRT-version problem. Sweeps over in-tree vs
  out-of-tree `amdxdna`, NPU firmware 1.0 vs 1.1, `force_cmdlist`, TDR
  timeout, and FLM versions all fail identically ([ROCm/FastFlowLM#712]).
- *Not* a proven "37–42 layer boundary." That comparison was uncontrolled
  (differing weights, vocabulary, conversion). The only established fact is
  that **engines differ**: `qwen3:4b` (36 layers) decodes at 19.8 tok/s on the
  same `qwen3` engine, and `gemma4-it:e4b` (42 layers) at 12.7 tok/s on
  `gemma4e` — so 42 layers is not a platform ceiling.
- Kernel geometry is not it either: swapping in 36-, 40- and 42-layer
  `layer.xclbin` binaries fails identically.

**Proposed-but-unproven fixes** (require AMD to ship a new `libqwen3_npu.so`):
splitting the 42-layer forward into two 21-layer sub-runlists, or dedicated
42-layer kernels. Meanwhile the open engine sidesteps the whole path by
running the layer list host-side.

---

## 🔬 Reproduction & Diagnostic Harnesses

These hit any OpenAI-compatible server. Point `--url` at `./scripts/run_oflm.sh`
for the working open-kernel engine, or `./scripts/run_flm.sh` to reproduce the
closed-engine ERT failure.

### 1. Minimal ERT Timeout Reproducer
Verifies prefill success vs decode timeout against a running FLM instance in under 10 seconds:
```bash
python3 scripts/reproduce_ert_timeout.py --url http://127.0.0.1:8001
```
(Closed-engine diagnostic only — on `oflm` it reports generation succeeding.)

### 2. Multi-Domain Output Quality Test
```bash
python3 scripts/test_quality.py --url http://127.0.0.1:8001
```

### 3. Speculative Decoding Benchmark (NPU Drafter + iGPU Target)
```bash
python3 scripts/benchmark_speculative.py --npu-url http://127.0.0.1:8001 --gpu-url http://127.0.0.1:8012
```
Times drafter and target side by side; it does not implement an accept/reject
speculative loop.

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
- Open kernel engine and kernels: [Atomic-Germ/OpenFlowLM-Next](https://github.com/Atomic-Germ/OpenFlowLM-Next)
- Closed FastFlowLM runtime (reproduction path only): [ROCm FastFlowLM](https://github.com/ROCm/FastFlowLM)
- Prebuilt Q4NX weights + AIE kernels: [huggingface.co/julianmb/MiniCPM5-2B-NPU2](https://huggingface.co/julianmb/MiniCPM5-2B-NPU2)
- Research and benchmark records: [julianmb/npuhalo](https://github.com/julianmb/npuhalo)

</details>
