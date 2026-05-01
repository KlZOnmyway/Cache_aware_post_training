"""Smoke tests for the model-agnostic controller core.

These tests do NOT require slime / Megatron / SGLang — they exercise the
``slime_adapter.controller`` modules directly.
"""

from __future__ import annotations

import math

import pytest
import torch

from slime_adapter import LayerCache, CreditsTracker, SwitchHead, ste_binary
from slime_adapter.controller.ste import ste_binary_with_noise, ste_binary_with_temperature


# -------------------------------------------------------------------
# LayerCache
# -------------------------------------------------------------------

def test_layer_cache_basic_push_evict():
    c = LayerCache(window=4, cap=30)
    c.push((0, 1))
    c.push((2, 3))
    c.push((4, 5))
    c.push((6, 7))           # window full now
    assert c.size == 8
    c.push((0, 8))            # evicts (0, 1) → (0,) goes to count=1, drops; (1,) drops
    # (0, 8): 0 is re-added so it's still in cache; 1 is evicted; 8 is new
    assert 0 in c
    assert 1 not in c
    assert 8 in c


def test_layer_cache_n_new_basic():
    c = LayerCache(window=4, cap=30)
    c.push((1, 2))
    c.push((3, 4))
    assert c.n_new((1, 2)) == 0     # both already in cache
    assert c.n_new((1, 5)) == 1     # only 5 is new
    assert c.n_new((6, 7)) == 2     # both new


def test_layer_cache_cap_enforced():
    c = LayerCache(window=4, cap=4)
    c.push((1, 2))
    c.push((3, 4))                  # 4 distinct, exactly cap
    assert c.size == 4
    c.push((5, 6))                  # 6 distinct, over cap → evict oldest entry (1, 2)
    assert c.size == 4
    assert 1 not in c
    assert 5 in c


# ------------------------------------------------------------------
# CreditsTracker
# ------------------------------------------------------------------

def test_credits_tracker_basic():
    t = CreditsTracker.from_config(num_moe_layers=10, fraction=0.7)
    assert t.total == pytest.approx(7.0)
    assert t.used == 0.0
    assert t.pressure == 0.0
    t.charge(switch_signal=1, n_new=2)
    assert t.used == 2.0
    assert t.pressure == pytest.approx(2.0 / 7.0)
    t.reset_for_new_token()
    assert t.used == 0.0


def test_credits_tracker_overflow():
    t = CreditsTracker(total_credits=2.0)
    t.charge(switch_signal=1, n_new=3)
    assert t.used == 3.0
    assert t.overflow == 1.0


# -------------------------------------------------------------------
# SwitchHead
# -------------------------------------------------------------------

def test_switch_head_init_bias_drives_initial_sigma():
    h = SwitchHead(hidden_size=8, init_bias=-2.0)
    x = torch.zeros(3, 5, 8)
    p = torch.zeros(3, 5)
    sigma = torch.sigmoid(h(x, p))
    # zero weight + init_bias=-2 → all logits = -2 → σ = sigmoid(-2) ≈ 0.119
    assert torch.allclose(sigma, torch.full_like(sigma, math.exp(-2) / (1 + math.exp(-2))), atol=1e-5)


def test_switch_head_pressure_input_required():
    h = SwitchHead(hidden_size=8, use_pressure_input=True)
    with pytest.raises(ValueError):
        h(torch.zeros(2, 3, 8))   # missing pressure


def test_switch_head_no_pressure_mode():
    h = SwitchHead(hidden_size=8, use_pressure_input=False)
    out = h(torch.zeros(2, 3, 8))
    assert out.shape == (2, 3)


# -------------------------------------------------------------------
# STE
# -------------------------------------------------------------------

def test_ste_binary_forward_is_threshold():
    sigma = torch.tensor([0.1, 0.4, 0.6, 0.9])
    out = (lambda s: ste_binary(s).detach())(sigma)
    assert torch.allclose(out, torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_ste_binary_backward_is_identity():
    sigma = torch.tensor([0.4], requires_grad=True)
    y = ste_binary(sigma)
    y.backward(torch.ones_like(y))
    assert torch.allclose(sigma.grad, torch.tensor([1.0]))


def test_ste_with_noise_clamps_to_unit():
    torch.manual_seed(0)
    sigma = torch.tensor([0.1, 0.5, 0.9], requires_grad=True)
    out = ste_binary_with_noise(sigma, noise_std=0.5)
    # Forward must be in {0, 1} regardless of how the noise lands.
    assert torch.all((out == 0) | (out == 1)), f"got {out!r}"


# Allow running directly without pytest
if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"OK   {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(0 if failed == 0 else 1)


from slime_adapter.controller.ste import ste_binary_with_noise  # noqa: E402  (re-import for tests above)
