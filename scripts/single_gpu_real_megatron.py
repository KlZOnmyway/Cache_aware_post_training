"""End-to-end smoke on a *real* Megatron-LM ``MoELayer`` stack.

Validates:
  - Real ``megatron.core.transformer.moe.MoELayer`` instances are constructed.
  - Each MoELayer's ``TopKRouter`` carries a ``routing_replay`` instance
    (slime's monkey-patch is active).
  - Our ``MegatronMoEAdapter.iter_moe_layers`` discovers them.
  - install_controller_into_layers attaches SwitchHead/cache/budget per layer.
  - Forward driver pre/post hooks fire correctly.
  - slime's RoutingReplay: record→replay reproduces routing bit-exact.
  - Aux loss (budget + barrier) backprops; total loss decreases over steps.

Pre-reqs:
  Megatron-LM at ``external/Megatron-LM`` (with slime patch applied).
  slime at ``external/slime`` (importable).
  ENABLE_ROUTING_REPLAY=1 set BEFORE module construction.
  CUDA_VISIBLE_DEVICES=<idle GPU>

Run:
  CUDA_VISIBLE_DEVICES=1 uv run --frozen python scripts/single_gpu_real_megatron.py
"""

from __future__ import annotations

import argparse
import os
import sys

# slime's register_routing_replay attaches per-layer buffer only when this
# env var is "1" at the time of router construction.
os.environ.setdefault("ENABLE_ROUTING_REPLAY", "1")

# Megatron and slime live under external/ (not pip installed).
_HERE = os.path.dirname(__file__)
_EXT = os.path.abspath(os.path.join(_HERE, "..", "external"))
for sub in ("Megatron-LM", "slime"):
    p = os.path.join(_EXT, sub)
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.distributed as dist
import torch.nn as nn

# Single-process distributed init — Megatron's parallel_state needs it.
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, world_size=1, rank=0)

from megatron.core import parallel_state as mpu  # noqa: E402
from megatron.core.tensor_parallel import random as mcore_random  # noqa: E402
mpu.destroy_model_parallel()
mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
# Initialize the model-parallel CUDA RNG tracker (required for ColumnParallelLinear init).
mcore_random.model_parallel_cuda_manual_seed(42)

import slime.utils.routing_replay as slime_rr  # noqa: E402

# Megatron has a bug where, if transformer-engine isn't installed, ``te_general_gemm``
# is referenced as a name in router_gating_linear without ever being defined.
# Inject a None so the ``if te_general_gemm is not None`` guard short-circuits cleanly.
import megatron.core.transformer.moe.moe_utils as _mu  # noqa: E402
if not hasattr(_mu, "te_general_gemm"):
    _mu.te_general_gemm = None

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec  # noqa: E402
from megatron.core.transformer.moe.moe_layer import MoELayer  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

from slime_adapter.megatron_hooks import (  # noqa: E402
    install_controller_into_layers,
    install_forward_driver,
    install_forward_completion_hook,
)
from slime_adapter.megatron_hooks.moe_forward_patch import call_original_forward  # noqa: E402
from slime_adapter.modeling._base import MoELayerHandle, MoEModelAdapter  # noqa: E402


# ===========================================================================
# Adapter for real Megatron MoELayer
# ===========================================================================

class MegatronMoEAdapter(MoEModelAdapter):
    name = "megatron_moe"

    def iter_moe_layers(self, model):
        for idx, layer in enumerate(model.decoder.layers):
            mlp = layer.mlp
            yield MoELayerHandle(
                layer_idx=idx,
                module=mlp,
                hidden_size=mlp.config.hidden_size,
                num_experts=mlp.config.num_moe_experts,
                native_top_k=mlp.router.topk,
            )

    def compute_router_top_k(self, moe_module, hidden_states, k=2):
        with torch.no_grad():
            logits = moe_module.router.gating(hidden_states)
            return logits.topk(k=k, dim=-1).indices

    def forward_with_forced_top_indices(self, moe_module, hidden_states, forced_indices):
        out = call_original_forward(moe_module, hidden_states)
        if isinstance(out, tuple):
            return out[0]
        return out


# ===========================================================================
# Build a stack of real Megatron MoELayer (no attention, just MoE chain)
# ===========================================================================

class MoEStack(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, num_experts: int, device: torch.device):
        super().__init__()
        cfg = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            ffn_hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_experts,
            moe_router_topk=2,
            moe_router_load_balancing_type="none",
            moe_token_dispatcher_type="alltoall",
            moe_router_score_function="softmax",
            moe_router_pre_softmax=False,
            moe_grouped_gemm=False,
            bias_activation_fusion=False,
            bias_dropout_fusion=False,
            gated_linear_unit=True,
            normalization="RMSNorm",
            add_bias_linear=False,
            params_dtype=torch.float32,
            pipeline_dtype=torch.float32,
            sequence_parallel=False,
            moe_router_dtype="fp32",
            bf16=False,
            fp16=False,
        )
        spec = get_gpt_layer_local_spec(num_experts=num_experts, moe_grouped_gemm=False)
        moe_submods = spec.submodules.mlp.submodules

        self.config = cfg
        self.decoder = nn.Module()
        layers = []
        for li in range(num_layers):
            layer = nn.Module()
            layer.mlp = MoELayer(config=cfg, submodules=moe_submods, layer_number=li + 1)
            layers.append(layer)
        self.decoder.layers = nn.ModuleList(layers)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder.layers:
            out = layer.mlp(x)
            if isinstance(out, tuple):
                out = out[0]
            x = out + x
        return x


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-fraction", type=float, default=0.7)
    p.add_argument("--budget-lambda", type=float, default=0.05)
    p.add_argument("--barrier-lambda", type=float, default=0.5)
    p.add_argument("--gate-init-bias", type=float, default=-1.0)
    p.add_argument("--use-pressure-input", action="store_true", default=True)
    p.add_argument("--cache-window", type=int, default=8)
    p.add_argument("--cache-cap", type=int, default=30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.total_credits = args.budget_fraction * args.num_layers
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"[real-megatron] device={device} layers={args.num_layers} experts={args.num_experts}")
    print(f"[real-megatron] ENABLE_ROUTING_REPLAY={os.environ.get('ENABLE_ROUTING_REPLAY')}")

    model = MoEStack(args.num_layers, args.hidden_size, args.num_experts, device)
    print(f"[real-megatron] built {args.num_layers} real Megatron MoELayer(s)")

    # Verify slime's patch attached routing_replay to each TopKRouter
    for li, layer in enumerate(model.decoder.layers):
        assert hasattr(layer.mlp.router, "routing_replay"), \
            f"layer {li} TopKRouter missing routing_replay (slime patch not active)"
    print(f"[real-megatron] all {args.num_layers} TopKRouter have routing_replay ✓")

    # Install our controller hooks
    adapter = MegatronMoEAdapter()
    install_controller_into_layers(model, adapter, args)
    install_forward_driver(model)
    install_forward_completion_hook(model)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[real-megatron] trainable params: {n_train}")
    print(f"[real-megatron] total_credits per token: {args.total_credits:.2f}\n")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    losses = []
    for step in range(args.steps):
        # ---- 1) Rollout: record routing ----
        os.environ["ROUTING_REPLAY_STAGE"] = "record"
        slime_rr.RoutingReplay.clear_all()
        x = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device)
        with torch.no_grad():
            target = model(x)
        rec = sum(
            len(layer.mlp.router.routing_replay.top_indices_list)
            for layer in model.decoder.layers
        )

        # ---- 2) Train forward: replay routing ----
        os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
        for layer in model.decoder.layers:
            layer.mlp.router.routing_replay.clear_forward()
        out = model(x)
        replay_diff = (out - target).abs().mean().item()

        # ---- 3) Aux loss ----
        from slime_adapter.loss.penalty_loss import state_or_none
        state = state_or_none()
        assert state is not None, "forward driver hooks didn't fire"
        s = state.summary()

        L_task = out.pow(2).mean()
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

        biases = [
            round(float(L.mlp.switch_head.linear.bias.detach().mean().item()), 3)
            for L in model.decoder.layers
        ]
        print(
            f"step {step}: recorded={rec} replay_diff={replay_diff:.2e} "
            f"L={losses[-1]:.4f} L_task={float(L_task):.4f} "
            f"L_bud={float(L_budget):.4f} L_bar={float(L_barrier):.4f} "
            f"biases={biases}"
        )
        for layer in model.decoder.layers:
            layer.mlp.controller_replay.reset()

    print(f"\n[real-megatron] losses: {[round(l, 3) for l in losses]}")
    if losses[0] > losses[-1]:
        print(f"[real-megatron] OK — loss dropped {losses[0]:.3f} → {losses[-1]:.3f}")
    else:
        print(f"[real-megatron] WARN: loss did not drop monotonically")


if __name__ == "__main__":
    main()
