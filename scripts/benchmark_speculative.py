#!/usr/bin/env python3
"""
benchmark_speculative.py — Speculative Drafter Benchmark Harness
AMD XDNA 2 NPU (MiniCPM5-2B) as Drafter vs AMD Radeon 8060S / 890M iGPU as Target/Verifier
"""

import time
import json
import asyncio
import aiohttp
import argparse
from typing import Dict, Any

BENCHMARK_PROMPTS = [
    {
        "category": "Factual",
        "prompt": "What is the capital of France? Answer in one short sentence.",
    },
    {
        "category": "Factual",
        "prompt": "Name the three primary colors in one short sentence.",
    },
    {
        "category": "Factual",
        "prompt": "Who wrote Romeo and Juliet? Answer in one short sentence.",
    },
    {
        "category": "Arithmetic",
        "prompt": "What is 15 * 23? Give only the number.",
    },
    {
        "category": "Arithmetic",
        "prompt": "Calculate 48 / 6. Give only the result.",
    },
    {
        "category": "Reasoning/Code",
        "prompt": "Write a one-line Python lambda function to check if a string is a palindrome.",
    },
    {
        "category": "Scientific",
        "prompt": "Explain photosynthesis in one concise sentence.",
    },
    {
        "category": "Creative",
        "prompt": "Write a 5-7-5 syllable haiku about ocean waves.",
    },
]

async def call_npu(session: aiohttp.ClientSession, npu_url: str, model: str, prompt: str, max_tokens: int = 32) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {"Connection": "close"}
    try:
        t0 = time.perf_counter()
        async with session.post(f"{npu_url}/v1/chat/completions", json=payload, headers=headers, timeout=30) as resp:
            data = await resp.json()
        t_elapsed = (time.perf_counter() - t0) * 1000
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return {
            "content": content,
            "tokens": usage.get("completion_tokens", len(content.split())),
            "elapsed_ms": t_elapsed,
            "tps": usage.get("decoding_speed_tps", 0.0),
            "ttft_ms": usage.get("prefill_duration_ttft", 0.0) * 1000,
        }
    except Exception as e:
        return {"content": f"Error: {e}", "tokens": 0, "elapsed_ms": 0.0, "tps": 0.0, "ttft_ms": 0.0}

async def call_gpu(session: aiohttp.ClientSession, gpu_url: str, model: str, prompt: str, max_tokens: int = 32) -> Dict[str, Any]:
    payload = {
        "prompt": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "n_predict": max_tokens,
        "temperature": 0.0,
    }
    try:
        t0 = time.perf_counter()
        async with session.post(f"{gpu_url}/completion", json=payload, timeout=30) as resp:
            data = await resp.json()
        t_elapsed = (time.perf_counter() - t0) * 1000
        content = data.get("content", "")
        timings = data.get("timings", {})
        tokens = timings.get("predicted_n", len(content.split()))
        tps = timings.get("predicted_per_second", 0.0)
        return {
            "content": content,
            "tokens": tokens,
            "elapsed_ms": t_elapsed,
            "predicted_per_second": tps,
            "prompt_ms": timings.get("prompt_ms", 0.0),
        }
    except Exception as e:
        return {"content": f"Error: {e}", "tokens": 0, "elapsed_ms": 0.0, "predicted_per_second": 0.0, "prompt_ms": 0.0}

async def main():
    parser = argparse.ArgumentParser(description="Speculative drafter benchmark.")
    parser.add_argument("--npu-url", default="http://127.0.0.1:8001", help="NPU drafter endpoint")
    parser.add_argument("--npu-model", default="minicpm5:2b", help="NPU model tag")
    parser.add_argument("--gpu-url", default="http://127.0.0.1:8012", help="GPU target endpoint")
    parser.add_argument("--gpu-model", default="target", help="GPU model name")
    args = parser.parse_args()

    print("=" * 80)
    print(" 🚀 SPECULATIVE DECODING BENCHMARK (NPU Drafter vs iGPU Target)")
    print(f"    Drafter: {args.npu_model} @ {args.npu_url}")
    print(f"    Target:  {args.gpu_model} @ {args.gpu_url}")
    print("=" * 80)

    async with aiohttp.ClientSession() as session:
        for idx, item in enumerate(BENCHMARK_PROMPTS, 1):
            prompt = item["prompt"]
            print(f"\n[{idx}/{len(BENCHMARK_PROMPTS)}] Prompt: {prompt}")
            
            npu_res = await call_npu(session, args.npu_url, args.npu_model, prompt)
            gpu_res = await call_gpu(session, args.gpu_url, args.gpu_model, prompt)
            
            print(f"  • NPU Drafter: {npu_res['elapsed_ms']:6.1f} ms | {npu_res['tokens']:2d} tok | '{npu_res['content'].strip()[:50]}'")
            print(f"  • GPU Target : {gpu_res['elapsed_ms']:6.1f} ms | {gpu_res['tokens']:2d} tok | '{gpu_res['content'].strip()[:50]}'")

if __name__ == "__main__":
    asyncio.run(main())
