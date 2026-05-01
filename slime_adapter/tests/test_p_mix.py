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
# Importance-weight rescaling on the loss side
# =============================================================================

def test_apply_importance_weights_direct_list():
    from slime_adapter.loss.penalty_loss import _apply_importance_weights_inplace
    import torch

    batch = {
        "advantages": [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0]),
        ],
        "importance_weights": [
            torch.tensor([0.5, 1.0, 2.0]),
            torch.tensor([1.0, 0.5]),
        ],
    }
    _apply_importance_weights_inplace(None, batch)
    assert torch.allclose(batch["advantages"][0], torch.tensor([0.5, 2.0, 6.0]))
    assert torch.allclose(batch["advantages"][1], torch.tensor([4.0, 2.5]))


def test_apply_importance_weights_via_sample_metadata():
    from slime_adapter.loss.penalty_loss import _apply_importance_weights_inplace
    import torch

    class FakeSample:
        def __init__(self, w):
            self.metadata = {"importance_weights": w}

    batch = {
        "advantages": [torch.tensor([10.0, 20.0])],
        "samples": [FakeSample([0.1, 0.5])],
    }
    _apply_importance_weights_inplace(None, batch)
    assert torch.allclose(batch["advantages"][0], torch.tensor([1.0, 10.0]))


def test_apply_importance_weights_no_op_when_missing():
    from slime_adapter.loss.penalty_loss import _apply_importance_weights_inplace
    import torch

    a0 = torch.tensor([1.0, 2.0])
    batch = {"advantages": [a0]}     # no IS info
    _apply_importance_weights_inplace(None, batch)
    assert torch.allclose(batch["advantages"][0], torch.tensor([1.0, 2.0]))


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
