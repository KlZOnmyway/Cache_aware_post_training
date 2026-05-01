"""Unit tests for BatchedLayerCache.

Checks per-batch independence, window+cap eviction, and equivalence with the
single-trajectory ``LayerCache`` when batch size = 1.
"""

from __future__ import annotations

import pytest
import torch

from slime_adapter.controller.cache_state import BatchedLayerCache, LayerCache


# =====================================================================
# Basic semantics
# =====================================================================

def test_initial_n_new_full():
    """Empty cache → every expert in top_k counts as new."""
    cache = BatchedLayerCache(num_experts=16, window=4, cap=4)
    cache.begin_batch(batch_size=2, device="cpu")
    top = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    assert torch.equal(cache.n_new(top), torch.tensor([2, 2]))


def test_push_then_repeat_gives_zero():
    cache = BatchedLayerCache(num_experts=16, window=4, cap=4)
    cache.begin_batch(batch_size=2, device="cpu")
    top = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    cache.push(top)
    assert torch.equal(cache.n_new(top), torch.tensor([0, 0]))


def test_per_batch_independence():
    """Two batch rows have independent cache histories."""
    cache = BatchedLayerCache(num_experts=16, window=4, cap=4)
    cache.begin_batch(batch_size=2, device="cpu")
    cache.push(torch.tensor([[0, 1], [2, 3]]))
    # row 0 sees (0,1) cached; row 1 sees (2,3) cached.
    probe = torch.tensor([[2, 3], [0, 1]], dtype=torch.long)
    assert torch.equal(cache.n_new(probe), torch.tensor([2, 2]))


def test_window_eviction():
    """Pushing > window entries causes oldest to fall out of the union."""
    cache = BatchedLayerCache(num_experts=16, window=2, cap=16)
    cache.begin_batch(batch_size=1, device="cpu")
    cache.push(torch.tensor([[0, 1]]))
    cache.push(torch.tensor([[2, 3]]))
    # window full
    cache.push(torch.tensor([[4, 5]]))
    # 0,1 should now be evicted
    probe = torch.tensor([[0, 1]])
    assert int(cache.n_new(probe)[0]) == 2


def test_window_only_eviction():
    """v4: cap dropped — pure rolling window. Distinct count is bounded
    implicitly by window × k."""
    cache = BatchedLayerCache(num_experts=64, window=3, cap=999)
    cache.begin_batch(batch_size=1, device="cpu")
    for ids in ([0, 1], [2, 3], [4, 5]):
        cache.push(torch.tensor([list(ids)]))
    assert int((cache.count > 0).sum().item()) == 6      # all in window
    cache.push(torch.tensor([[6, 7]]))                   # evicts (0,1)
    assert int((cache.count > 0).sum().item()) == 6
    assert cache.count[0, 0] == 0 and cache.count[0, 6] == 1


# =============================================================================
# Equivalence with the single-trajectory LayerCache (B=1)
# =============================================================================

def test_b1_matches_python_layer_cache():
    """For B=1 the tensor cache should track the Python deque exactly."""
    # use cap large enough that window-eviction is the only mechanism in play
    py = LayerCache(window=4, cap=999)
    bt = BatchedLayerCache(num_experts=16, window=4, cap=999)
    bt.begin_batch(batch_size=1, device="cpu")

    sequence = [(0, 1), (2, 3), (0, 4), (5, 6), (7, 8), (0, 9)]
    for top in sequence:
        py_n = py.n_new(top)
        bt_n = int(bt.n_new(torch.tensor([list(top)])).item())
        assert py_n == bt_n, f"mismatch on {top}: py={py_n}, bt={bt_n}"
        py.push(top)
        bt.push(torch.tensor([list(top)]))


# =====================================================================
# Per-(b, t) accuracy under the wrapped forward
# =====================================================================

def test_sequential_step_advances_per_batch():
    """Verify each batch row's cache accumulates independently across steps."""
    cache = BatchedLayerCache(num_experts=8, window=4, cap=8)
    cache.begin_batch(batch_size=3, device="cpu")

    # Step 0: each row picks different experts
    cache.push(torch.tensor([[0, 1], [2, 3], [4, 5]]))
    # Step 1: row 0 reuses (0, 1) → n_new=0; row 1 picks (4, 5) → n_new=2; row 2 picks (2, 3) → n_new=2
    n = cache.n_new(torch.tensor([[0, 1], [4, 5], [2, 3]]))
    assert n.tolist() == [0, 2, 2]
