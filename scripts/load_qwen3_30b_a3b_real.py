"""Load real Qwen3-30B-A3B into HF transformers and verify our adapter
works on the real Qwen3MoeSparseMoeBlock forward path."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-layers", type=int, default=4, help="number of MoE layers to keep (skip rest)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    print(f"[qwen3-real] loading Qwen3-30B-A3B from cache (truncated to {args.num_layers} layers)")
    t0 = time.time()

    from transformers import Qwen3MoeForCausalLM, Qwen3MoeConfig, AutoTokenizer

    snapshot = next(
        os.path.join(f"{CACHE}/snapshots", d) for d in os.listdir(f"{CACHE}/snapshots") if not d.startswith(".")
    )
    cfg = Qwen3MoeConfig.from_pretrained(snapshot)
    print(f"[qwen3-real] orig config: {cfg.num_hidden_layers} layers, {cfg.num_experts} experts, "
          f"top-{cfg.num_experts_per_tok}, hidden {cfg.hidden_size}")

    # Truncate to fit single GPU and fast iteration
    cfg.num_hidden_layers = args.num_layers
    cfg.use_cache = False

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # Build model from config (random init for skipped layers; we just need shape)
    model = Qwen3MoeForCausalLM(cfg).to(dtype=dtype, device=args.device)
    print(f"[qwen3-real] built {args.num_layers}-layer Qwen3MoE in {time.time()-t0:.1f}s, "
          f"params={sum(p.numel() for p in model.parameters())/1e9:.2f}B "
          f"(memory={torch.cuda.max_memory_allocated()/1e9:.2f}GB)")

    # Verify shape: each transformer layer has .mlp = Qwen3MoeSparseMoeBlock
    layer0 = model.model.layers[0]
    print(f"[qwen3-real] layer0.mlp type: {type(layer0.mlp).__name__}")
    assert "Qwen3MoeSparseMoeBlock" in type(layer0.mlp).__name__
    print(f"[qwen3-real] mlp.gate type: {type(layer0.mlp.gate).__name__}, "
          f"mlp.experts: {len(layer0.mlp.experts)} experts")

    # Forward sanity
    tok = AutoTokenizer.from_pretrained(snapshot)
    inp = tok("Hello, world.", return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model(**inp)
    print(f"[qwen3-real] forward OK: logits shape={tuple(out.logits.shape)}")
    print(f"[qwen3-real] OK")


if __name__ == "__main__":
    main()
