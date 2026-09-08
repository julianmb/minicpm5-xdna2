#!/usr/bin/env python3
"""
test_quality.py — Multi-Domain Output Quality Evaluation Harness for MiniCPM5-2B

Evaluates generation quality across 5 standard domains:
1. Math & multi-step arithmetic
2. Logical reasoning & deduction
3. Code analysis & bug fixing
4. Structured JSON extraction
5. Instruction following (haiku)
"""

import sys
import time
import argparse
import requests

TEST_CASES = [
    {
        "name": "Math & Multi-step Calculation",
        "prompt": "A farmer has 15 cows and 25 chickens. How many total legs are on the farm? Show your step-by-step calculation.",
        "max_tokens": 200,
        "temperature": 0.0
    },
    {
        "name": "Logical Deduction",
        "prompt": "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have? Explain clearly.",
        "max_tokens": 150,
        "temperature": 0.0
    },
    {
        "name": "Code Analysis & Edge Case Handling",
        "prompt": "Look at this function:\n```python\ndef average(numbers):\n    return sum(numbers) / len(numbers)\n```\nWhat bug happens if `numbers` is empty? Provide the fixed Python function.",
        "max_tokens": 200,
        "temperature": 0.0
    },
    {
        "name": "Structured JSON Extraction",
        "prompt": "Extract the key entities from this sentence: 'On October 14, 2024, Dr. Sarah Lin from DeepMind presented a keynote in Tokyo regarding reinforcement learning.' Return valid JSON with keys: 'person', 'organization', 'date', 'location', 'topic'. Do not output extra prose.",
        "max_tokens": 150,
        "temperature": 0.0
    },
    {
        "name": "Instruction Following / Haiku",
        "prompt": "Write a 5-7-5 syllable haiku about an NPU running AI on silicon.",
        "max_tokens": 100,
        "temperature": 0.3
    }
]

def run_evaluation(url: str, model: str):
    endpoint = f"{url}/v1/chat/completions"
    print("=" * 70)
    print(f"QUALITY EVALUATION ON {model} (AMD XDNA 2 NPU)")
    print("=" * 70)

    for idx, tc in enumerate(TEST_CASES, 1):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": tc["prompt"]}],
            "max_tokens": tc["max_tokens"],
            "temperature": tc["temperature"]
        }

        t0 = time.time()
        try:
            resp = requests.post(endpoint, json=payload, timeout=60)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"[{idx}] {tc['name']} FAILED to connect: {e}")
            continue

        if resp.status_code != 200:
            print(f"[{idx}] {tc['name']} FAILED: HTTP {resp.status_code} - {resp.text}")
            continue

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})

        decode_speed = usage.get("decoding_speed_tps", 0.0)
        prefill_speed = usage.get("prefill_speed_tps", 0.0)
        ttft = usage.get("prefill_duration_ttft", 0.0)

        print(f"\n--- Test {idx}: {tc['name']} ---")
        print(f"Prompt: {tc['prompt']}")
        print(f"Output:\n{content.strip()}")
        print(f"\n[Metrics] TTFT: {ttft:.3f}s | Prefill: {prefill_speed:.1f} tok/s | Decode: {decode_speed:.1f} tok/s | Total: {elapsed:.2f}s")
        print("-" * 70)

    print("\nQuality Evaluation Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate output quality of MiniCPM5-2B.")
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="FLM server base URL")
    parser.add_argument("--model", default="minicpm5:2b", help="Model tag")
    args = parser.parse_args()
    run_evaluation(args.url, args.model)
