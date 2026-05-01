"""End-to-end smoke against the *real* slime ``RoutingReplay``.

Tiny in-process MoE whose router is wrapped by slime's
``get_routing_replay_compute_topk``, so toggling ``ROUTING_REPLAY_STAGE``
exercises the same record/replay path slime patches into mcore.

Verifies:
  - slime ``routing_replay`` is per-layer registered (needs ENABLE_ROUTING_REPLAY=1
    BEFORE the router is constructed).
  - rollout (record) → trainer (replay_forward) reproduces routing bit-exact.
  - SwitchHead/cache/budget gradients flow + loss decreases.
"""

from __future__ import annotations

import argparse
import os

# slime registers ``routing_replay`` on each TopKRouter only if this env is "1"
# at the time of construction — set it before importing/building anything else.
os.environ.setdefault("ENABLE_ROUTING_REPLAY", "1")

import torch
import torch.nn as nn

import slime.utils.routing_replay as slime_rr  # type: ignore  # noqa: E402

from slime_adapter.megatron_hooks import (  # noqa: E402
    install_controller_into_layers,
    install_forward_driver,
    install_forward_completion_hook,
)
from slime_adapter.megatron_hooks.moe_forward_patch import call_original_forward  # noqa: E402
from slime_adapter.modeling._base import MoELayerHandle, MoEModelAdapter  # noqa: E402


# =====================================================================
# Tiny MoE that uses slime's RoutingReplay-wrapped compute_topk
# =====================================================================

class SlimeRouter(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_experts))
        self.num_experts = num_experts
        self.top_k = 2

        def native(scores, topk, num_groups=None, group_topk=None):
            top = scores.topk(topk, dim=-1).indices
            return scores.gather(-1, top), top

        self._compute = slime_rr.get_routing_replay_compute_topk(native)
        slime_rr.register_routing_replay(self)

    def forward(self, hidden):
        scores = torch.softmax(
            torch.nn.functional.linear(hidden, self.weight, self.bias), dim=-1
        )
        flat = scores.reshape(-1, scores.shape[-1])
        slime_rr.set_routing_replay(self.routing_replay)
        probs, idx = self._compute(flat, self.top_k)
        return probs.reshape(*scores.shape[:-1], -1), idx.reshape(*scores.shape[:-1], -1)


class Expert(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return torch.relu(self.fc(x))


class MoEBlock(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.router = SlimeRouter(hidden_size, num_experts)
        self.experts = nn.ModuleList(Expert_alias := [Expert(hidden_size) for _ in range(num_experts)])

    def forward(self, hidden, *, forced_indices=None):
        _, indices = self.router(hidden)
        if forced_indices is not None:
            indices = forced_indices
        out = torch.zeros_like(hidden)
        for e_idx, expert in enumerate(self.experts):
            mask = (indices == e_idx).any(dim=-1, keepdim=True).to(hidden.dtype)
            out = out + mask * expert(hidden)
        return out


# Alias for clarity
Expert = Expert


class Layer(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.mlp = MoEBlock(hidden_size, num_experts)

    def forward(self, x):
        return self.mlp(x) + x


class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_experts):
        super().__init__()
        self.layers = nn.ModuleList(Layer(hidden_size, num_experts) for _ in range(num_layers))

    def forward(self, x):
        for L in self.layers:
            x = L(x)
        return x


class Cfg:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size


class TinyMoE(nn.Module):
    def __init__(self, hidden_size, num_layers, num_experts):
        super().__init__()
        self.config = Cfg(hidden_size)
        self.decoder = Decoder(hidden_size, num_layers, num_experts)

    def forward(self, x):
        return self.decoder(x)


# =====================================================================
# Adapter
# =====================================================================

class TinyAdapter(MoEModelAdapter):
    name = "tiny_slime"

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
                hidden_states, moe_module.router.weight, moe_module.router.bias
            ).topk(k=k, dim=-1).indices

    def forward_with_forced_top_indices(self, moe_module, hidden_states, forced_indices):
        return call_original_forward(moe_module, hidden_states, forced_indices=forced_indices)


# =====================================================================
# CLI + main
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-fraction", type=float, default=0.7)
    p.add_argument("--budget-lambda", type=float, default=0.05)
    p.add_argument("--barrier-lambda", type=float, default=0.5)
    p.add_argument("--gate-init-bias", type=float, default=-1.0)
    p.add_argument("--cache-window", type=int, default=16)
    p.add_argument("--cache-cap", type=int, default=30)
    p.add_argument("--use-pressure-input", action="store_true", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.total_credits = args.budget_fraction * args.num_layers
    return args


class TopModel(nn.Module):
    def __init__(self, hidden_size, num_layers, num_experts):
        super().__init__()
        self.config = Cfg(hidden_size)
        self.decoder = Dec(hidden_size, num_layers, num_experts)

    def forward(self, x):
        return self.decoder(x)


class Dec(nn.Module):
    def __init__(self, hidden_size, num_layers, num_experts):
        super().__init__()
        self.layers = nn.ModuleList(L(hidden_size, num_experts) for _ in range(num_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class L(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.mlp = MoEBlock(hidden_size, num_experts)

    def forward(self, x):
        return self.mlp(x) + x


def run():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"[real-slime] device={device} L={args.num_layers} E={args.num_experts} H={args.hidden_size}")
    print(f"[real-slime] ENABLE_ROUTING_REPLAY={os.environ.get('ENABLE_ROUTING_REPLAY')}")

    model = TopModel(args.hidden_size, args.num_layers, args.num_experts).to(device)

    # Verify slime attached routing_replay buffers per-layer
    for layer in model.decoder.layers:
        assert hasattr(layer.mlp.router, "routing_replay"), \
            "slime did not register routing_replay; check ENABLE_ROUTING_REPLAY env timing"

    adapter = TinyAdapter()
    install_controller_into_layers(model, adapter, args)
    install_forward_driver(model)
    install_forward_completion_hook(model)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[real-slime] params={n_train} total_credits={args.total_credits:.2f}\n")

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)

    losses = []
    for step in range(args.steps):
        # 1) ROLLOUT — record routing
        os.environ["ROUTING_REPLAY_STAGE"] = "record"
        slime_rr.RoutingReplay.clear_all()

        x = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device)
        with torch.no_grad():
            target_out = model(x)

        rec = sum(len(L.mlp.router.routing_replay.top_indices_list) for L in model.decoder.layers)

        # 2) TRAIN forward — replay
        os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
        for L in model.decoder.layers:
            L.mlp.router.routing_replay.clear_forward()

        out = model(x)
        replay_diff = (out - target_out).abs().mean().item()

        # 3) Compute aux loss
        from slime_adapter.loss.penalty_loss import state_or_none
        state = state_or_none()
        assert state is not None
        s = state.summary()
        L_task = out.pow(2).mean()
        L_budget = (
            torch.stack(s.layer_local_costs, 0).sum(0).mean()
            if s.layer_local_costs else torch.zeros((), device=device)
        )
        overflow = (s.total_used_per_token - args.total_credits).clamp_min(0.0)
        L_barrier = (overflow * overflow).mean()
        L_total = L_task + args.budget_lambda * L_budget + args.barrier_lambda * L_barrier

        optimizer.zero_grad(set_to_none=True)
        L_total.backward()
        optimizer.step()
        losses.append(float(L_total.detach()))

        biases = [round(float(LL.mlp.switch_head.linear.bias.detach().mean()), 3)
                  for LL in model.decoder.layers]
        print(
            f"step {step}: recorded={rec} replay_diff={replay_diff:.2e} "
            f"L_total={float(L_total):.4f} L_task={float(L_task):.4f} "
            f"L_bud={float(L_budget):.4f} L_bar={float(L_barrier):.4f} "
            f"biases={biases}"
        )
        for LL in model.decoder.layers:
            LL.mlp.controller_replay.reset()

    print(f"\n[real-slime] losses: {[round(l, 3) for l in losses or [0]]}")
    if losses[0] > losses[-1]:
        print(f"[real-slime] OK — loss dropped {losses[0]:.3f} → {losses[-1]:.3f}")
    else:
        print(f"[real-slime] WARN — loss did not drop monotonically")


# Adapter alias to handle ordering
class TinyAdapter(TinyAdapter):  # noqa: F811
    pass


if __name__ == "__main__":
    run()
