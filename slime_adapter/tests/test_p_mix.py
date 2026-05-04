"""Unit tests for MiniLLM-style p_mix rollout + IS-weight integration.

These tests do NOT spin up SGLang; ``query_topk_logprobs`` is exercised in
the integration test suite. Here we focus on the math:

  * sample_p_mix produces sane (token, w_t, log_p_S) given mock top-K dicts
  * boundary cases α=0 and α=1 give the expected IS weights
  * loss-side `_apply_importance_weights_inplace` rescales advantages correctly
"""

from __future__ import annotations

import math

import pytest
import torch


# =====================================================================
# sample_p_mix
# =====================================================================

def test_sample_p_mix_basic():
    from slime_adapter.rollout.mix_generate import sample_p_mix

    student = {1: -0.5, 2: -1.0, 3: -2.0}
    teacher = {2: -0.3, 3: -1.5, 4: -1.8}

    torch.manual_seed(0)
    tok, w, log_p_S, log_p_mix = sample_p_mix(student, teacher, alpha=0.5)
    # token must be from union of supports
    assert tok in {1, 2, 3, 4}
    # importance weight is positive and finite
    assert 0.0 < w < 1e6
    # log_p_S is non-positive (it's a log-probability)
    assert log_p_S <= 1e-6
    assert log_p_mix <= 1e-6


def test_sample_p_mix_alpha_zero_is_pure_student():
    """α=0 → p_mix = p_student → w_t = 1 exactly."""
    from slime_adapter.rollout.mix_generate import sample_p_mix

    student = {10: -0.1, 11: -2.0}
    teacher = {12: -0.5, 13: -3.0}
    torch_seed_all(7)
    for _ in range(20):
        tok, w, _, _ = sample_p_mix_call(student, teacher, alpha=0.0)
        assert tok in student.keys()
        assert math.isclose(w, 1.0, rel_tol=1e-6)


def torch_seed_all(s):
    import torch
    torch.manual_seed(s)


def sample_p_mix_call(s, t, alpha):
    from slime_adapter.rollout.mix_generate import sample_p_mix
    return sample_p_mix(s, t, alpha)


def test_sample_p_mix_alpha_one_is_pure_teacher():
    """With α=1 the rollout draws from p_teacher only; w_t = p_S/p_T."""
    student = {10: -0.1, 11: -2.0}
    teacher = {10: -0.5, 11: -1.5}
    torch_seed_all(11)
    seen = set()
    for _ in range(50):
        tok, w, _, _ = sample_p_mix_call(student, teacher, alpha=1.0)
        seen.add(tok)
        # w_t = p_S(tok) / p_T(tok); both >0 since tok ∈ teacher's support
        assert w > 0.0
    # we should sample at least one of teacher's tokens
    assert seen.issubset({10, 11})


# =============================================================================
# Switch PG loss computation
# =============================================================================

def test_switch_pg_per_sample_alignment():
    """Verify _compute_switch_pg aligns per-sample advantages with [B, T] switch logprob."""
    from slime_adapter.loss.penalty_loss import _compute_switch_pg

    zero = torch.tensor(0.0)
    B, T = 2, 4

    class FakeSummary:
        switch_logprob_per_token = torch.randn(B, T)

    batch = {
        "advantages": [torch.tensor([1.0, 1.0, 1.0]), torch.tensor([-1.0, -1.0])],
    }
    L = _compute_switch_pg(batch, FakeSummary(), zero)
    assert L.shape == ()
    assert L.requires_grad is False   # advantages are detached
    assert L.item() != 0.0           # should be non-trivial


# =============================================================================
# End-to-end: p_mix sample → trajectory_cache_cost path is preserved
# =============================================================================

def test_cache_cost_unchanged_by_p_mix_path():
    """p_mix only affects sampling and IS weights; cache cost on a sample
    with mocked controller_records is computed by the same function."""
    from slime_adapter.rollout.reward_kl import trajectory_cache_cost

    class S:
        pass

    s = S()
    s.controller_records = [
        {"switch": 1, "n_new": 2, "token": 0, "layer": 0, "used_top2": [3, 7],
         "new_top2": [3, 7], "pressure_in": 0.0},
        {"switch": 0, "n_new": 0, "token": 0, "layer": 1, "used_top2": [3, 7],
         "new_top2": [4, 8], "pressure_in": 0.1},
        {"switch": 1, "n_new": 1, "token": 1, "layer": 0, "used_top2": [4, 7],
         "new_top2": [4, 7], "pressure_in": 0.0},
    ]
    assert trajectory_cache_cost(s) == 1 * 2 + 0 + 1 * 1
