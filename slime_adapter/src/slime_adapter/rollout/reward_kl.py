"""Reward function for slime — distillation + cache cost in the trajectory reward.

Two callbacks for slime's rollout pipeline:

  - ``reward_func(args, sample, **kwargs)`` (async)
        Calls a frozen teacher SGLang server and stashes per-token teacher
        logprobs onto each sample. The teacher-KL contribution to training
        flows through slime's reference-model machinery (``--kl-coef``); this
        function only fetches.

  - ``post_process_rewards(args, samples, **kwargs)``
        Builds the **scalar trajectory reward** that GRPO turns into the
        group-relative advantage A_t. Composition::

            r_traj = α_q · task(traj)
                     − α_c · Σ_t Σ_l  switch_{t,l} · n_new_{t,l}

        That cache-cost term is what makes this real RL — switch_t mutates
        ω_{t+1}, which mutates r_{t+k}, and PG with discounted return
        attributes the credit. See PORT_TO_SLIME.md §3.

Hyperparameters on ``args``:
    correctness_reward_alpha (α_q)  weight of task-correctness reward.   default 0.0
    cache_cost_lambda        (α_c)  weight of trajectory cache cost.     default ``budget_lambda``
"""

from __future__ import annotations

from typing import Any, List

import aiohttp


# =============================================================================
# Async reward fetch (teacher SGLang)
# =============================================================================

async def reward_func(args, sample, **kwargs):
    """Fetch teacher logprobs from a frozen SGLang teacher.

    Result is later parsed by ``post_process_rewards`` into
    ``sample.teacher_log_probs`` for slime's KL-to-reference machinery.
    """
    payload = {
        "input_ids": list(sample.tokens),
        "sampling_params": {"max_new_tokens": 0, "temperature": 1.0},
        "return_logprob": True,
    }
    url = str(args.rm_url).rstrip("/")
    if not url.endswith("/generate"):
        url = url + "/generate"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
    return data


# =====================================================================
# Per-sample reward post-processing
# =====================================================================

def post_process_rewards(args, samples: List[Any], **kwargs):
    """Build the scalar trajectory reward GRPO sees.

    Returns ``(rewards, rewards)`` — slime expects two parallel lists.
    """
    α_q = float(getattr(args, "correctness_reward_alpha", 0.0))
    α_c = float(
        getattr(args, "cache_cost_lambda",
                getattr(args, "budget_lambda", 0.0))
    )
    cold_start = int(getattr(args, "cache_cost_cold_start_skip", 0))

    rewards: list[float] = []
    for sample in samples:
        _stash_teacher_logprobs(sample)

        r_task = α_q * float(getattr(sample, "is_correct", 0.0))
        cache_cost = trajectory_cache_cost(sample, skip_first=cold_start)
        r_cache = -α_c * cache_cost

        sample.reward_task = r_task
        sample.reward_cache_cost = r_cache
        sample.cache_cost_raw = cache_cost

        rewards.append(r_task + r_cache)
    return rewards, list(rewards)


# =============================================================================
# Helpers
# =============================================================================

def trajectory_cache_cost(sample, *, skip_first: int = 0) -> float:
    """``Σ_t Σ_l  switch_{t,l} · n_new_{t,l}`` from sample.controller_records.

    Args:
        sample: a slime ``Sample`` (or anything with ``controller_records`` /
            ``metadata['controller_records']``).
        skip_first: number of leading response tokens for which we *do not*
            charge cache cost. The cache is empty at t=0 so n_new=k for every
            layer in the first ~window steps; charging the policy for that
            cold-start would burn the whole budget on the trivially-required
            initial loads. Default 0 (no masking); recommended ≈ window=16
            once you're tuning end-to-end.
    """
    recs = getattr(sample, "controller_records", None)
    if recs is None:
        md = getattr(sample, "metadata", None)
        if isinstance(md, dict):
            recs = md.get("controller_records")
    if not recs:
        return 0.0
    total = 0.0
    for r in recs:
        try:
            tok = int(r.get("token", 0))
            if tok < int(skip_first):
                continue
            total += float(int(r["switch"]) * int(r["n_new"]))
        except (KeyError, TypeError, ValueError):
            continue
    return total


def attach_teacher_logprobs(sample) -> None:
    """Public alias for tests."""
    _stash_teacher_logprobs(sample)


def _stash_teacher_logprobs(sample) -> None:
    raw = getattr(sample, "reward", None)
    if not raw or not isinstance(raw, dict):
        return
    try:
        # SGLang OPD-style payload: meta_info.input_token_logprobs is
        # [(logprob, token_id, ...), ...].
        meta = raw.get("meta_info") or {}
        token_lp = meta.get("input_token_logprobs") or raw.get("input_token_logprobs")
        if not token_lp:
            return
        import torch
        # First entry is BOS / no-target — drop it.
        vals = []
        for item in token_lp[1:]:
            vals.append(
                float(item[0]) if isinstance(item, (list, tuple)) else float(item)
            )
        sample.teacher_log_probs = torch.tensor(vals, dtype=torch.float32)
    except Exception:
        return


__all__ = [
    "reward_func",
    "post_process_rewards",
    "trajectory_cache_cost",
    "attach_teacher_logprobs",
]
