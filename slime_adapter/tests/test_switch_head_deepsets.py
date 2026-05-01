"""Unit tests for ExpertSetEncoder + SwitchHead-with-DeepSets-context."""

from __future__ import annotations

import torch

from slime_adapter.controller.expert_set_encoder import ExpertSetEncoder
from slime_adapter.controller.switch_head import SwitchHead


# =====================================================================
# ExpertSetEncoder
# =====================================================================

def test_encoder_indices_permutation_invariant():
    enc = ExpertSetEncoder(num_experts=64, embed_dim=32, set_dim=16)
    a = enc.encode_indices(torch.tensor([[0, 1, 2]]))
    b = enc.encode_indices(torch.tensor([[2, 0, 1]]))
    assert torch.allclose(a, b, atol=1e-6)


def test_encoder_indices_vs_mask_agreement():
    enc = ExpertSetEncoder(num_experts=32, embed_dim=8, set_dim=8)
    idx = torch.tensor([[3, 5, 7]])
    mask = torch.zeros(1, 32, dtype=torch.bool)
    mask[0, [3, 5, 7]] = True
    rep_idx = enc.encode_indices(idx)
    rep_mask = enc.encode_mask(mask)
    assert torch.allclose(rep_idx, rep_mask, atol=1e-6)


def test_encoder_handles_minus_one_padding():
    enc = ExpertSetEncoder(num_experts=64, embed_dim=16, set_dim=16)
    full = enc.encode_indices(torch.tensor([[0, 1, 2]]))
    padded = enc.encode_indices(torch.tensor([[0, 1, 2, -1]]))
    assert torch.allclose(full, padded, atol=1e-6)


# =====================================================================
# SwitchHead with DeepSets context
# =====================================================================

def test_switch_head_legacy_compat():
    sh = SwitchHead(hidden_size=8, init_bias=-2.0)
    out = sh(torch.zeros(2, 4, 8), pressure_scalar=torch.zeros(2, 4))
    assert out.shape == (2, 4)


def test_switch_head_with_set_inputs():
    sh = SwitchHead(hidden_size=8, init_bias=-2.0,
                    cache_set_dim=16, topk_set_dim=16)
    h = torch.randn(3, 5, 8)
    p = torch.zeros(3, 5)
    cache = torch.zeros(3, 5, 16)
    top2 = torch.zeros(3, 5, 16)
    out = sh(h, p, cache_set_repr=cache, top_k_set_repr=top2)
    assert out.shape == (3, 5)


def test_switch_head_broadcasts_per_batch_set_repr_over_T():
    sh = SwitchHead(hidden_size=8, cache_set_dim=8, topk_set_dim=8)
    h = torch.randn(2, 4, 8)
    p = torch.zeros(2, 4)
    cache_b = torch.randn(2, 8)        # [B, set_dim]
    top2_b = torch.randn(2, 8)
    out = sh(h, pressure_scalar=p, cache_set_repr=cache_b, top_k_set_repr=top2_b)
    assert out.shape == (2, 4)


def test_gradient_flows_through_encoder_and_switch_head():
    enc = ExpertSetEncoder(num_experts=32, embed_dim=8, set_dim=16)
    sh = SwitchHead(hidden_size=8, init_bias=-2.0,
                    cache_set_dim=16, topk_set_dim=16)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    pressure = torch.zeros(2, 4)
    cache_idx = torch.tensor([[0, 1, 2], [3, 4, 5]])
    top2_idx = torch.tensor([[6, 7], [8, 9]])

    cache_rep = enc.encode_indices(cache_idx)
    top2_rep = enc.encode_indices(top2_idx)
    logits = sh(hidden, pressure, cache_set_repr=cache_rep, top_k_set_repr=top2_rep)
    logits.mean().backward()

    grads = [p.grad for p in list(enc.parameters()) + list(sh.parameters())
             if p.grad is not None]
    assert grads, "no gradients flowed back"
    assert any(g.abs().max().item() > 0 for g in grads)
