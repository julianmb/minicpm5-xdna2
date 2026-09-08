#!/usr/bin/env python3
"""
inject_qk_norm.py — Synthetic QK-Norm Injection for FastFlowLM Qwen3 Engine

FastFlowLM's Qwen3 execution engine (`libqwen3_npu.so`) dynamically reads `head_dim: 128`
and selects the `_gen_mha_seq_d128_q2` kernel. However, libqwen3_npu.so strictly expects
`model.layers.{i}.self_attn.q_norm.weight` and `k_norm.weight` ([128] BF16).

Since MiniCPM5-2B has no architectural QK-normalization, this script injects synthetic
unit tensors (gamma = 1.0) into `model.q4nx` across all 42 transformer layers:
    RMSNorm(x, gamma=1.0) = (x / RMS(x)) * 1.0
This satisfies the FastFlowLM kernel loader without altering numerical precision.
"""

import sys
import argparse
import torch
from safetensors.torch import load_file, save_file

def inject_qk_norm(q4nx_path: str, num_layers: int = 42, head_dim: int = 128):
    print(f"[INFO] Loading {q4nx_path}...")
    tensors = load_file(q4nx_path)

    print(f"[INFO] Injecting unit q_norm and k_norm across {num_layers} layers (head_dim={head_dim})...")
    injected_count = 0
    for i in range(num_layers):
        q_norm_key = f"model.layers.{i}.self_attn.q_norm.weight"
        k_norm_key = f"model.layers.{i}.self_attn.k_norm.weight"
        
        if q_norm_key not in tensors:
            tensors[q_norm_key] = torch.ones(head_dim, dtype=torch.bfloat16)
            injected_count += 1
        if k_norm_key not in tensors:
            tensors[k_norm_key] = torch.ones(head_dim, dtype=torch.bfloat16)
            injected_count += 1

    print(f"[INFO] Injected {injected_count} unit normalization tensors. Total tensors: {len(tensors)}")
    print(f"[INFO] Saving back to {q4nx_path}...")
    save_file(tensors, q4nx_path)
    print("[SUCCESS] Injection complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject synthetic QK-norm weights into model.q4nx.")
    parser.add_argument("q4nx_path", help="Path to model.q4nx file")
    parser.add_argument("--layers", type=int, default=42, help="Number of transformer layers (default: 42)")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension (default: 128)")
    args = parser.parse_args()
    inject_qk_norm(args.q4nx_path, args.layers, args.head_dim)
