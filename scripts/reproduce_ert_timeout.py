#!/usr/bin/env python3
"""
reproduce_ert_timeout.py — 2-Minute Repro Harness for 42-Layer Runlist Timeout

Demonstrates the exact execution boundary on AMD XDNA 2 (Strix Halo):
1. Prefill succeeds cleanly (Chunked Matrix-Vector GEMM via mm.xclbin).
2. Decode fails on token 0 with ERT_CMD_STATE_TIMEOUT because `libqwen3_npu.so`
   chains 42 layers into a single `xrt::runlist` batch, overflowing the AIE command processor queue.

Usage:
  1. Start FLM server:
       ./scripts/run_flm.sh minicpm5:2b 8001
  2. Run this script:
       python3 scripts/reproduce_ert_timeout.py --url http://127.0.0.1:8001
"""

import sys
import time
import argparse
import requests

def main():
    parser = argparse.ArgumentParser(description="Test MiniCPM5-2B prefill vs decode on FastFlowLM NPU.")
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="FLM server base URL")
    parser.add_argument("--model", default="minicpm5:2b", help="Model tag")
    args = parser.parse_args()

    endpoint = f"{args.url}/v1/chat/completions"
    print("=" * 70)
    print(f"Testing {args.model} against {endpoint}")
    print("=" * 70)

    # Check server health / model list. Per ROCm/FastFlowLM#716, an
    # unresolvable tag can silently serve the resident model, so verify the
    # requested tag is registered before attributing the outcome.
    try:
        models_resp = requests.get(f"{args.url}/v1/models", timeout=5)
        print(f"[1/3] Server reachable (HTTP {models_resp.status_code})")
        try:
            listed = [m.get("id", "") for m in models_resp.json().get("data", [])]
            if args.model not in listed:
                print(f"  [WARN] '{args.model}' not in /v1/models ({listed}). "
                      f"Outcome below cannot be attributed to MiniCPM5.")
            else:
                print(f"  [OK] '{args.model}' registered in /v1/models.")
        except Exception:
            print("  [WARN] Could not parse /v1/models; outcome unattributable.")
    except Exception as e:
        print(f"[ERROR] Cannot connect to FLM at {args.url}: {e}")
        print("Please start the server first using ./scripts/run_flm.sh minicpm5:2b 8001")
        sys.exit(1)

    payload = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ],
        "max_tokens": 10,
        "temperature": 0.0
    }

    print("\n[2/3] Dispatching inference request...")
    print("  Expected behavior:")
    print("    - Prefill: SUCESSS (FLM log shows 'Start prefill... Prefill chunk 1/1 with ... tokens')")
    print("    - Decode:  FAILS with 'runlist failed execution (ERT_CMD_STATE_TIMEOUT)'")
    print("-" * 70)

    t0 = time.time()
    try:
        resp = requests.post(endpoint, json=payload, timeout=30)
        elapsed = time.time() - t0
        print(f"[3/3] Received response in {elapsed:.2f}s (HTTP {resp.status_code}):")
        print(resp.text)
        
        if "ERT_CMD_STATE_TIMEOUT" in resp.text:
            print("\n[REPRODUCED] ERT_CMD_STATE_TIMEOUT returned by the server.")
            print("NOTE: this records an HTTP-level failure signature only; it does not "
                  "by itself identify the failing kernel or submission depth.")
        elif resp.status_code == 200 and "choices" in resp.json():
            served = resp.json().get("model", "")
            if served and served != args.model:
                print(f"\n[UNATTRIBUTABLE] Requested '{args.model}' but response reports "
                      f"'{served}' (see ROCm/FastFlowLM#716). Output skipped.")
            else:
                print("\n[PASSED] Generation succeeded! Output:")
                print(resp.json()["choices"][0]["message"]["content"])
    except requests.exceptions.Timeout:
        print("\n[TIMEOUT] Request timed out on HTTP client side (>30s).")
    except Exception as e:
        print(f"\n[ERROR] Request failed: {e}")

if __name__ == "__main__":
    main()
