"""Tests for LoRA on expert layers."""

import torch
import torch.nn as nn
import pytest


def test_lora_linear_output_unchanged_at_init():
    """LoRA B=0 at init → output equals base output."""
    from slime_adapter.modeling.lora import LoRALinear

    base = nn.Linear(16, 32)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(2, 16)

    base_out = base(x)
    lora_out = lora(x)
    assert torch.allclose(base_out, lora_out, atol=1e-6)


def test_lora_linear_base_frozen():
    """Base parameters should be frozen after wrapping."""
    from slime_adapter.modeling.lora import LoRALinear

    base = nn.Linear(16, 32)
    lora = LoRALinear(base, r=4, alpha=8)

    for p in lora.base.parameters():
        assert not p.requires_grad
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad


def test_lora_linear_gradient_flows():
    """Gradient should flow through LoRA A and B."""
    from slime_adapter.modeling.lora import LoRALinear

    base = nn.Linear(16, 32)
    lora = LoRALinear(base, r=4, alpha=8)
    lora.lora_B.data.normal_(std=0.1)

    x = torch.randn(2, 16)
    out = lora(x)
    out.sum().backward()

    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None
    assert lora.lora_A.grad.abs().sum() > 0


def test_lora_linear_tuple_output():
    """Handles mcore-style (output, bias) tuple returns."""
    from slime_adapter.modeling.lora import LoRALinear

    class FakeParallelLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(32, 16))

        def forward(self, x):
            return x @ self.weight.T, torch.zeros(32)

    base = FakeParallelLinear()
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(2, 16)
    out = lora(x)

    assert isinstance(out, tuple)
    assert out[0].shape == (2, 32)
    assert out[1].shape == (32,)


def test_freeze_base_params():
    """freeze_base_params freezes everything except lora/controller/router."""
    from slime_adapter.modeling.lora import freeze_base_params

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(16, 16)
            self.switch_head = nn.Linear(16, 1)
            self.router = nn.Linear(16, 8)
            self.lora_A = nn.Parameter(torch.zeros(4, 16))
            self.lora_B = nn.Parameter(torch.zeros(64, 4))

    model = FakeModel()
    stats = freeze_base_params(model)

    assert not model.attention.weight.requires_grad
    assert model.switch_head.weight.requires_grad
    assert model.router.weight.requires_grad
    assert model.lora_A.requires_grad


def test_collect_param_groups():
    """collect_param_groups builds groups with correct LRs."""
    from slime_adapter.modeling.lora import collect_param_groups

    model = nn.Module()
    model.switch_head_linear = nn.Linear(16, 1)
    model.router_weight = nn.Parameter(torch.randn(8, 16))
    model.lora_A_param = nn.Parameter(torch.randn(4, 16))

    groups = collect_param_groups(
        model, lora_lr=1e-5, router_lr=2e-5, controller_lr=1e-4
    )

    names = {g["name"] for g in groups}
    assert "lora" in names
    assert "router" in names
    assert "controller" in names

    for g in groups:
        if g["name"] == "lora":
            assert g["lr"] == 1e-5
        elif g["name"] == "router":
            assert g["lr"] == 2e-5
        elif g["name"] == "controller":
            assert g["lr"] == 1e-4


def test_apply_expert_lora_on_sequential_mlp():
    """apply_expert_lora wraps linear_fc1/fc2 in SequentialMLP-like structure."""
    from slime_adapter.modeling.lora import LoRALinear, apply_expert_lora

    class FakeExpertMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = nn.Linear(16, 64)
            self.linear_fc2 = nn.Linear(64, 16)

    class FakeExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.local_experts = nn.ModuleList([FakeExpertMLP() for _ in range(4)])

    class FakeMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = nn.Linear(16, 4)
            self.experts = FakeExperts()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe = FakeMoE()

    class FakeAdapter:
        def iter_moe_layers(self, model):
            from slime_adapter.modeling._base import MoELayerHandle
            yield MoELayerHandle(
                layer_idx=0, module=model.moe,
                hidden_size=16, num_experts=4, native_top_k=2,
            )

    model = FakeModel()
    count = apply_expert_lora(model, FakeAdapter(), r=4, alpha=8)

    assert count == 8  # 4 experts × 2 linears
    for expert in model.moe.experts.local_experts:
        assert isinstance(expert.linear_fc1, LoRALinear)
        assert isinstance(expert.linear_fc2, LoRALinear)
        assert not expert.linear_fc1.base.weight.requires_grad
        assert expert.linear_fc1.lora_A.requires_grad
