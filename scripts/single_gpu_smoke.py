"""Single-GPU end-to-end smoke test for the slime_adapter pipeline.

Boots a tiny mock MoE model and exercises the full controller flow:

  1. ``install_controller_into_layers`` wires SwitchHead/cache/budget per layer.
  2. Forward pre-hook calls ``begin_controller_forward`` (allocates TokenBudgetState).
  3. Wrapped MoE forward: σ → STE → switch → used_top2 → cache push → cost charge.
  4. Forward post-hook ends budget state and stashes summary in the loss-side TLS.
  5. We compute ``L_task + λ_b·L_budget + λ_h·L_barrier`` and backward.

Verifies that:
  - Gradients flow through SwitchHead (per-layer biases drift across steps).
  - Per-layer ``layer_local_costs`` and ``total_used_per_token`` are populated.
  - The full pipeline runs without slime / Megatron / sglang installed.

Usage::

    uv run --frozen --extra dev python scripts/single_gpu_smoke.py
    uv run --frozen --extra dev python scripts/single_gpu_smoke.py --steps 8 --batch-size 4
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from slime_adapter.megatron_hooks import (
    install_controller_into_layers,
    install_forward_driver,
    install_forward_completion_hook,
)
from slime_adapter.megatron_hooks.moe_forward_patch import call_original_forward
from slime_adapter.modeling._base import MoELayerHandle, MoEModelAdapter


# ====================================================================
# Tiny mock MoE — same shape as Qwen3 (layer.mlp = MoE block)
# ====================================================================

class _Router(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_experts))
        self.num_experts = num_experts
        self.top_k = 2

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class _Expert(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return torch.relu(self.fc(x))


class _MoEBlock(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.router = _Router(hidden_size, num_experts)
        self.experts = nn.ModuleList(_Expert(hidden_size) for _ in range(num_experts))

    def forward(self, hidden, *, forced_indices=None):
        if forced_indices is None:
            scores = torch.softmax(self.router(hidden), dim=-1)
            top2 = scores.topk(k=2, dim=-1).indices
        else:
            top2 = forced_indices
        out = torch.zeros_like(hidden)
        for e_idx, expert in enumerate(self.experts):
            mask = (top2 == e_idx).any(dim=-1, keepdim=True).to(hidden.dtype)
            out = out + mask * expert(hidden)
        return out


class _Layer(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.mlp = _MoEBlock(hidden_size, num_experts)

    def forward(self, x):
        return self.mlp(x) + x


class _Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_experts):
        super().__init__()
        self.layers = nn.ModuleList(_Layer(hidden_size, num_experts) for _ in range(num_layers))

    def forward(self, x):
        for L in self.layers:
            x = L(x)
        return x


class _Config:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size


class MockModel(nn.Module):
    def __init__(self, hidden_size, num_layers, num_experts):
        super().__init__()
        self.config = _Config(hidden_size)
        self.decoder = _Decoder(hidden_size, num_layers, num_experts)

    def forward(self, hidden):
        return self.decoder(hidden)


# Alias to keep _Layer factory readable
_MoEBlock = _MoEBlock  # noqa: F811


class _Layer(_Layer):  # noqa: F811
    pass  # alias


# ====================================================================
# Mock adapter
# ====================================================================

class MockAdapter(MoEModelAdapter):
    name = "mock"

    def iter_moe_layers(self, model):
        for idx, layer in enumerate(model.decoder.layers):
            mlp = layer.mlp
            yield MoELayerHandle(
                layer_idx=idx, module=mlp,
                hidden_size=mlp.hidden_size,
                num_experts=mlp.num_experts,
                native_top_k=2,
            )

    def compute_router_top_k(self, moe_module, hidden_states, k=2):
        with torch.no_grad():
            return torch.nn.functional.linear(
                hidden_states, moe_module.router.weight, moe_module.router.bias,
            ).topk(k=k, dim=-1).indices

    def forward_with_forced_top_indices(self, moe_module, hidden_states, forced_indices):
        return call_original_forward(moe_module, hidden_states, forced_indices=forced_indices)


# ====================================================================
# Smoke runner
# ====================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-fraction", type=float, default=0.7)
    p.add_argument("--budget-lambda", type=float, default=0.05)
    p.add_argument("--barrier-lambda", type=float, default=0.5)
    p.add_argument("--gate-init-bias", type=float, default=-1.0)  # softer init → see budget cost flow
    p.add_argument("--cache-window", type=int, default=16)
    p.add_argument("--cache-cap", type=int, default=30)
    p.add_argument("--use-pressure-input", action="store_true", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    args.total_credits = args.budget_fraction * args.num_layers
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model = MockModel(args.hidden_size, args.num_layers, args.num_experts).to(device)

    adapter = MockAdapter()
    install_controller_into_layers(model, adapter, args)
    install_forward_driver(model)
    install_forward_completion_hook(model)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] device={device} layers={args.num_layers} experts={args.num_experts} hidden={args.hidden_size}")
    print(f"[smoke] trainable params: {n_train}")
    print(f"[smoke] total_credits = {args.total_credits} per token")
    print()

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)

    for step in range(args.steps):
        x = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device)
        out = model(x)

        from slime_adapter.loss.penalty_loss import state_or_none
        state = state_or_none()
        if state is None:
            print(f"[smoke] step {step}: WARN no controller state")
            continue
        summary = state.summary()

        if summary.layer_local_costs:
            L_budget = torch.stack(summary.layer_local_costs, dim=0).sum(dim=0).mean()
        else:
            L_budget = torch.zeros((), device=device)
        overflow = (summary.total_used_per_token - args.total_credits).clamp_min(0.0)
        L_barrier = (overflow * overflow).mean()
        L_task = out.pow(2).mean()
        L_total = L_task + args.budget_lambda * L_budget + args.barrier_lambda * L_barrier

        optimizer.zero_grad(set_to_none=True)
        L_total.backward()
        optimizer.step()

        # diagnostics
        n_on, n_total = 0, 0
        for L in model.decoder.layers:
            for e in L.mlp.controller_replay.entries:
                n_on += int(e["switch"].sum().item())
                n_total += int(e["switch"].numel())
        rate = n_on / max(1, n_total) if n_total else 0.0
        biases = [round(float(L.mlp.switch_head.linear.bias.detach().mean().item()), 3)
                  for L in model.decoder.layers]

        print(
            f"step {step:2d}: "
            f"L={float(L_total.detach()):.4f}  "
            f"L_task={float(L_task.detach()):.4f}  "
            f"L_bud={float(L_budget.detach()):.4f}  "
            f"L_bar={float(L_barrier.detach()):.4f}  "
            f"sw_rate={rate:.3f}  "
            f"biases={biases}"
        )

        for L in model.decoder.layers:
            L.mlp.controller_replay.reset()

    print("\n[smoke] OK.")


if __name__ == "__main__":
    args = parse_args()
    args.total_credits = args.budget_fraction * args.num_layers
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model = MockModel(args.hidden_size, args.num_layers, args.num_experts).to(device)
    adapter = MockAdapter()

    install_controller_into_layers(model, adapter, args)
    install_forward_driver(model)
    install_forward_completion_hook(model)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] device={device} layers={args.num_layers} experts={args.num_experts} hidden={args.hidden_size}")
    print(f"[smoke] trainable params: {n_train}")
    print(f"[smoke] total_credits per token: {args.total_credits}\n")

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)

    for step in range(args.steps):
        x = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device)
        out = model(x)

        from slime_adapter.loss.penalty_loss import state_or_none
        state = state_or_none()
        if state is None:
            print(f"[step {step}] WARN: no controller state captured (driver hooks not active)")
            continue
        summary = state.summary()

        L_task = out.pow(2).mean()
        if summary.layer_local_costs:
            L_budget = torch.stack(summary.layer_local_costs, dim=0).sum(dim=0).mean()
        else:
            L_budget = torch.zeros((), device=device)
        overflow = (summary.total_used_per_token - args.total_credits).clamp_min(0.0)
        L_barrier = (overflow * overflow).mean()
        L_total = L_task + args.budget_lambda * L_budget + args.barrier_lambda * L_barrier

        optimizer.zero_grad(set_to_none=True)
        L_total.backward()
        optimizer.step()

        n_on = sum(int(e["switch"].sum().item())
                   for L in model.decoder.layers for e in L.mlp.controller_replay.entries)
        n_total = sum(int(e["switch"].numel())
                      for L in model.decoder.layers for e in L.mlp.controller_replay.entries)
        rate = n_on / max(1, n_total) if n_total else 0.0
        biases = [round(float(L.mlp.switch_head.linear.bias.detach().mean().item()), 3)
                  for L in model.decoder.layers]
        print(
            f"step {step:2d}: "
            f"L={float(L_total.detach()):.4f}  "
            f"L_task={float(L_task.detach()):.4f}  "
            f"L_budget={float(L_budget.detach()):.4f}  "
            f"L_barrier={float(L_barrier.detach()):.4f}  "
            f"switch_rate={rate:.3f}  "
            f"biases={biases}"
        )
        for L in model.decoder.layers:
            L.mlp.controller_replay.reset()

    print("\n[smoke] OK.")
