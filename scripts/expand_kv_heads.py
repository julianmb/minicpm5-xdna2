#!/usr/bin/env python3
"""
expand_kv_heads.py — GQA Adaptation for AMD XDNA 2 NPU (FastFlowLM)

MiniCPM5-2B is trained with 16 Query heads and 2 Key/Value heads (16:2 = 8:1 GQA ratio).
FastFlowLM's multi-head attention AIE kernel (libmha.so) only supports 2:1, 3:1, and 4:1 ratios
for head_dim=128 (e.g. `_gen_mha_seq_d128_q2`).

This script mathematically expands the 2 KV heads to 8 KV heads (4x replication per head).
Under GQA, replicating head weights across redundant heads results in bit-for-bit mathematically
identical attention distributions, transforming the architecture into a 16:8 (2:1) GQA model
that natively matches the AIE tile hardware kernel.
"""

import os
import json
import shutil
import argparse
import torch
from safetensors.torch import load_file, save_file

def adapt_model(src_dir: str, dst_dir: str, target_kv_heads: int = 8):
    os.makedirs(dst_dir, exist_ok=True)

    # 1. Update config.json
    config_path = os.path.join(src_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {src_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    orig_kv_heads = config.get("num_key_value_heads", 2)
    orig_q_heads = config.get("num_attention_heads", 16)
    head_dim = config.get("head_dim", 128)
    num_layers = config.get("num_hidden_layers", 42)

    print(f"[INFO] Loaded config: H_q={orig_q_heads}, H_kv={orig_kv_heads}, d_head={head_dim}, layers={num_layers}")
    assert target_kv_heads % orig_kv_heads == 0, (
        f"Target {target_kv_heads} must be divisible by original {orig_kv_heads}"
    )
    rep_factor = target_kv_heads // orig_kv_heads
    print(f"[INFO] Expanding KV heads from {orig_kv_heads} to {target_kv_heads} (replication factor {rep_factor}x)")

    config["num_key_value_heads"] = target_kv_heads
    with open(os.path.join(dst_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Saved modified config.json to {dst_dir}")

    # 2. Copy auxiliary files
    aux_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja"
    ]
    for af in aux_files:
        src_file = os.path.join(src_dir, af)
        if os.path.exists(src_file):
            shutil.copy(src_file, os.path.join(dst_dir, af))
            print(f"[INFO] Copied {af}")

    # 3. Load & expand Safetensors
    st_files = [f for f in os.listdir(src_dir) if f.endswith(".safetensors")]
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {src_dir}")

    print(f"[INFO] Found safetensor files: {st_files}")
    for st_name in st_files:
        src_st = os.path.join(src_dir, st_name)
        dst_st = os.path.join(dst_dir, st_name)
        print(f"[INFO] Processing {src_st}...")

        state_dict = load_file(src_st)
        new_state_dict = {}

        for k, v in state_dict.items():
            if "self_attn.k_proj.weight" in k or "self_attn.v_proj.weight" in k:
                hidden_size = v.shape[-1]
                assert v.shape[0] == orig_kv_heads * head_dim, f"Unexpected shape for {k}: {v.shape}"
                v_reshaped = v.view(orig_kv_heads, head_dim, hidden_size)
                v_expanded = v_reshaped.repeat_interleave(rep_factor, dim=0)
                v_out = v_expanded.view(target_kv_heads * head_dim, hidden_size)
                new_state_dict[k] = v_out.contiguous()
            elif "self_attn.k_proj.bias" in k or "self_attn.v_proj.bias" in k:
                assert v.shape[0] == orig_kv_heads * head_dim
                v_reshaped = v.view(orig_kv_heads, head_dim)
                v_expanded = v_reshaped.repeat_interleave(rep_factor, dim=0)
                v_out = v_expanded.view(target_kv_heads * head_dim)
                new_state_dict[k] = v_out.contiguous()
            else:
                new_state_dict[k] = v

        print(f"[INFO] Saving adapted weights to {dst_st}...")
        save_file(new_state_dict, dst_st, metadata={"format": "pt"})
        print(f"[INFO] Successfully saved {dst_st}")

    print("[SUCCESS] Model adaptation complete. Ready for GGUF / Q4NX conversion.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand KV heads in MiniCPM5-2B for FastFlowLM compatibility.")
    parser.add_argument("--src", required=True, help="Path to original HF MiniCPM5-2B directory")
    parser.add_argument("--dst", required=True, help="Output directory for adapted HF weights")
    parser.add_argument("--target-kv-heads", type=int, default=8, help="Target number of KV heads (default: 8)")
    args = parser.parse_args()
    adapt_model(args.src, args.dst, args.target_kv_heads)
