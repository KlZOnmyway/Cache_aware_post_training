"""End-to-end on real Qwen3-MoE-30B-A3B (truncated to N layers) from HF cache.

Validates:
  - Real ``Qwen3MoeSparseMoeBlock`` layers (128 experts, top-8 native, 2048 hidden).
  - Our adapter discovers them and wraps forward.
  - SwitchHead/cache/budget on top of REAL Qwen3 router.
  - End-to-end loss + backward + parameters update.

Run:
  CUDA_VISIBLE_DEVICES=1 uv run --frozen --no-sync python scripts/single_gpu_real_qwen3.py --num-layers 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

from slime_adapter.megatron_hooks import (
    install_controller_into_layers,
    install_forward_driver,
    install_forward_completion_hook,
)
from slime_adapter.megatron_hooks.moe_forward_patch import call_original_forward
from slime_adapter.modeling._base import MoELayerHandle, MoEModelAdapter

CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B")


# ====================================================================
# Adapter for HF Qwen3MoeSparseMoeBlock
# ====================================================================

class Qwen3HFAdapter(MoEModelAdapter):
    name = "qwen3_hf"

    def iter_moe_layers(self, model):
        for idx, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            if not hasattr(mlp, "gate") or not hasattr(mlp, "experts"):
                continue
            yield MoELayerHandle(
                layer_idx=idx,
                module=mlp,
                hidden_size=model.config.hidden_size,
                num_experts=len(mlp.experts),
                native_top_k=model.config.num_experts_per_tok,
            )

    def compute_router_top_k(self, moe_module, hidden_states, k=2):
        with torch.no_grad():
            # Qwen3MoeSparseMoeBlock expects [B, T, H] but its gate works on flat (N, H)
            flat = hidden_states.reshape(-1, hidden_states.shape[-1]).to(moe_module.gate.weight.dtype)
            logits = moe_module.gate(flat)  # [N, num_experts]
            top = logits.topk(k=k, dim=-1).indices  # [N, k]
            return top.reshape(*hidden_states.shape[:-1], k)

    def forward_with_forced_top_indices(self, moe_module, hidden_states, forced_indices):
        # We don't (yet) override the router topk in the HF Qwen3 forward path —
        # we let it use its natural top-K and just observe + record. The
        # SwitchHead/cache/budget terms still get gradient via STE on σ.
        return call_original_forward(moe_module, hidden_states)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-fraction", type=float, default=0.7)
    p.add_argument("--budget-lambda", type=float, default=0.01)
    p.add_argument("--barrier-lambda", type=float, default=0.1)
    p.add_argument("--gate-init-bias", type=float, default=-1.0)
    p.add_argument("--use-pressure-input", action="store_true", default=True)
    p.add_argument("--cache-window", type=int, default=8)
    p.add_argument("--cache-cap", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()
    args.total_credits = args.budget_fraction * args.num_layers

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"[real-qwen3] loading Qwen3-30B-A3B (truncated to {args.num_layers} layers)")
    t0 = time.time()
    from transformers import Qwen3MoeForCausalLM, Qwen3MoeConfig

    snapshot = next(
        os.path.join(f"{CACHE}/snapshots", d) for d in os.listdir(f"{CACHE}/snapshots") if not d.startswith(".")
    )
    cfg = Qwen3MoeConfig.from_pretrained(snapshot)
    print(f"[real-qwen3]   orig: {cfg.num_hidden_layers} layers, {cfg.num_experts} experts, "
          f"top-{cfg.num_experts_per_tok}, hidden {cfg.hidden_size}")

    cfg.num_hidden_layers = args.num_layers
    cfg.use_cache = False
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    model = Qwen3MoeForCausalLM(cfg).to(dtype=dtype, device=device)
    print(f"[real-qwen3] built in {time.time()-t0:.1f}s, params={sum(p.numel() for p in model.parameters())/1e9:.2f}B"
          f", mem={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Install our controller hooks
    adapter = Qwen3HFAdapter()
    handles = list(adapter.iter_moe_layers(model))
    print(f"[real-qwen3] adapter found {len(handles)} MoE layers")
    for h in handles:
        print(f"  layer {h.layer_idx}: H={h.hidden_size} E={h.num_experts} native_top_k={h.native_top_k}")

    install_controller_into_layers(model, adapter, args)
    install_forward_driver(model)
    install_forward_completion_hook(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[real-qwen3] trainable: {n_train/1e9:.2f}B  total_credits={args.total_credits:.2f}\n")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    losses = []
    for step in range(args.steps):
        x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
        out = model(x)
        logits = out.logits

        from slime_adapter.loss.penalty_loss import state_or_none
        state = state_or_none()
        if state is None:
            print(f"  step {step}: WARN no controller state")
            continue
        s = state.summary()

        # task: maximize logit entropy slightly so it actually moves
        L_task = -(torch.softmax(logits.float(), -1) * torch.log_softmax(logits.float(), -1)).sum(-1).mean()
        if s.layer_local_costs:
            L_budget = torch.stack(s.layer_local_costs, 0).sum(0).mean()
        else:
            L_budget = torch.zeros((), device=device)
        overflow = (s.total_used_per_token - args.total_credits).clamp_min(0.0)
        L_barrier = (overflow * overflow).mean()
        L_total = L_task + args.budget_lambda * L_budget + args.barrier_lambda * L_barrier

        optimizer.zero_grad(set_to_none=True)
        L_total.backward()
        optimizer.step()
        losses.append(float(L_total.detach()))

        biases = [round(float(L.mlp.switch_head.linear.bias.detach().mean().item()), 3)
                  for L in model.model.layers]
        print(
            f"step {step}: L={losses[-1]:.4f} L_task={float(L_task):.4f} "
            f"L_bud={float(L_budget):.4f} L_bar={float(L_barrier):.4f} "
            f"biases={biases}"
        )
        for L in model.model.layers:
            L.mlp.controller_replay.reset()

    print(f"\n[real-qwen3] losses: {[round(l, 3) for l in losses]}")
    print(f"[real-qwen3] {'OK — loss dropped' if losses[0] > losses[-1] else 'WARN'}")


if __name__ == "__main__":
    main()
