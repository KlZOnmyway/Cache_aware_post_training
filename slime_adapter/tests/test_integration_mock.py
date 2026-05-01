"""End-to-end integration test on a tiny mock MoE model.

Exercises the full controller forward pipeline without slime / Megatron / sglang:

  install_controller_into_layers → begin_controller_forward → model(...) →
  end_controller_forward → check per-layer costs and switch decisions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from slime_adapter.modeling._base import MoELayerHandle, MoEModelAdapter


# ----- mock model -----------------------------------------------------------

class _Router(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_experts))
        self.num_experts = num_experts
        self.top_k = 2

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class _Expert(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.linear(x)


class _MoEBlock(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int):
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
        self.layers = nn.ModuleList([
            _Layer(hidden_size, num_experts) for _ in range(num_layers)
        ])

    def forward(self, x):
        for L in self.layers:
            x = L(x)
        return x


class _Layer(_Layer):  # noqa: F811 — re-using cleanly above.
    pass


class _Config:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size


class MockMoEModel(nn.Module):
    """Looks like Megatron's ``model.decoder.layers``."""

    def __init__(self, hidden_size: int = 8, num_layers: int = 3, num_experts: int = 8):
        super().__init__()
        self.config = _Config(hidden_size)
        self.decoder = _Decoder(hidden_size, num_layers, num_experts)

    def forward(self, x):
        return self.decoder(x)


# =============================================================================
# Mock adapter
# =============================================================================

class _MockAdapter(MoEModelAdapter):
    name = "mock"

    def iter_moe_layers(self, model):
        for idx, layer in enumerate(model.decoder.layers):
            mlp = layer.mlp
            yield MoELayerHandle(
                layer_idx=idx,
                module=mlp,
                hidden_size=mlp.hidden_size,
                num_experts=mlp.num_experts,
                native_top_k=2,
            )

    def compute_router_top_k(self, moe_module, hidden_states, k=2):
        logits = torch.nn.functional.linear(
            hidden_states, weight=moe_module.router.weight, bias=moe_module.router.bias,
        )
        return logits.topk(k=k, dim=-1).indices

    def forward_with_forced_top_indices(self, moe_module, hidden_states, forced_indices):
        # Avoid recursing through our wrapped forward — call the saved original.
        from slime_adapter.megatron_hooks.moe_forward_patch import call_original_forward
        return call_original_forward(moe_module, hidden_states, forced_indices=forced_indices)


# ============================================================================
# Args + helper
# ============================================================================

class _Args:
    gate_init_bias = -2.0
    use_pressure_input = True
    cache_window = 4
    cache_cap = 30
    budget_fraction = 0.7


def _make_model(num_layers=3, hidden_size=8, num_experts=8):
    return MockMoEModel(hidden_size=hidden_size, num_layers=num_layers, num_experts=num_experts)


# ============================================================================
# Tests
# ============================================================================

def test_install_controller_into_layers_attaches_state():
    from slime_adapter.megatron_hooks.moe_forward_patch import install_controller_into_layers

    torch.manual_seed(0)
    model = _make_model(num_layers=3)
    install_controller_into_layers(model, _MockAdapter(), _Args())

    for layer in model.decoder.layers:
        mlp = layer.mlp
        assert hasattr(mlp, "switch_head")
        assert hasattr(mlp, "cache_state")
        assert hasattr(mlp, "controller_replay")
        assert mlp._slime_adapter_wrapped is True


def test_full_forward_produces_layer_costs():
    from slime_adapter.megatron_hooks.moe_forward_patch import (
        install_controller_into_layers, begin_controller_forward, end_controller_forward,
    )

    torch.manual_seed(0)
    model = _make_model(num_layers=3)
    install_controller_into_layers(model, _MockAdapter(), _Args())

    B, T, H = 2, 5, 8
    hidden = torch.randn(B, T, H)
    begin_controller_forward(model, hidden_states_proxy=hidden)
    out = model(hidden)
    state = end_controller_forward(model)

    assert out.shape == hidden.shape
    assert state is not None
    summary = state.summary()
    assert len(summary.layer_local_costs) == 3
    for cost in summary.layer_local_costs:
        assert cost.shape == (B, T)
        assert (cost >= 0).all()

    # total_used_per_token = sum of per-layer costs
    total = summary.total_used_per_token
    expected = sum(summary.layer_local_costs)
    assert torch.allclose(total.detach(), expected.detach())


def test_init_bias_yields_low_switch_rate():
    """With init_bias=-2.0 (σ≈0.12), most STE outputs should be 0."""
    from slime_adapter.megatron_hooks.moe_forward_patch import (
        install_controller_into_layers, begin_controller_forward, end_controller_forward,
    )

    torch.manual_seed(0)
    model = _make_model(num_layers=3)
    install_controller_into_layers(model, _MockAdapter(), _Args())

    hidden = torch.randn(2, 8, 8)
    begin_controller_forward(model, hidden_states_proxy=hidden)
    _ = model(hidden)
    end_controller_forward(model)

    n_on, n_total = 0, 0
    for layer in model.decoder.layers:
        for entry in layer.mlp.controller_replay.entries:
            sw = entry["switch"]
            n_on += int(sw.sum().item())
            n_total += int(sw.numel())
    assert n_total > 0
    rate = n_on / n_total
    assert rate < 0.5, f"initial switch rate too high: {rate}"


# ============================================================================
# Mock adapter (re-declared for clarity at end)
# ============================================================================

class _MockAdapter(_MockAdapter):  # noqa: F811
    pass
