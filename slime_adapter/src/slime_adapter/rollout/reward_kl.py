"""KL distillation reward — adapted from slime/rollout/on_policy_distillation.py.

Two callbacks for slime's rollout pipeline:

  - ``reward_func(args, sample, **kw)`` — async; queries a teacher SGLang
    server for token-level logprobs and returns the raw response. Wired up
    via ``--custom-rm-path slime_adapter.rollout.reward_kl:reward_func`` and
    ``--rm-url http://teacher-host:port``.

  - ``post_process_rewards(args, samples, **kw)`` — pulls the teacher
    logprobs out of each sample's reward payload, stores them in
    ``sample.teacher_log_probs``, and returns scalar rewards (= 0.0 because
    the actual KL signal is consumed by the loss patch, not the GRPO advantage).

Together with the loss patch (``slime_adapter.loss.penalty_loss``), this gives
us a pure on-policy distillation training signal.
"""

from __future__ import annotations

from typing import Any, List

import aiohttp
import torch


async def reward_func(args, sample, **kwargs) -> dict:
    """Async reward fetch: ask the frozen-teacher SGLang for input_token_logprobs.

    Args expected on ``args``:
        rm_url: teacher SGLang base url, e.g. http://localhost:30001
    """
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "max_new_tokens": 0,
            "temperature": 1.0,
        },
        "return_logprob": True,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, samples: List[Any], **kwargs):
    """Stash teacher logprobs onto each sample; return scalar 0 rewards.

    GRPO/PPO compute advantages from these scalars; we want the advantage to
    be uniform (or near-zero) since the real signal goes through the loss
    patch's KL term. If ``args.correctness_reward_alpha > 0`` is set, you can
    add a per-trajectory bonus here.
    """
    raw_rewards = [getattr(s, "_raw_reward_response", None) for s in samples]
    response_lengths = [getattr(s, "response_length", len(s.tokens)) for s in samples]

    for sample, raw, rl in zip(samples, raw_rewards, response_lengths):
        if raw is None:
            sample.teacher_log_probs = None
            continue
        # SGLang OPD logprob format
        try:
            tlp = [item[0] for item in raw["meta_info"]["input_token_logprobs"][1:]]
        except KeyError:
            sample.teacher_log_probs = None
            continue
        import torch as _torch
        sample.teacher_log_probs = _torch.tensor(tlp, dtype=_torch.float32)[-rl:]

    # Scalar reward = 0 (KL signal lives in the loss patch).
    rewards = [0.0] * len(samples)
    # Optional: add correctness bonus when args.correctness_reward_alpha > 0
    alpha = float(getattr(args, "correctness_reward_alpha", 0.0))
    if alpha > 0:
        rewards = [alpha * float(getattr(s, "is_correct", False)) for s in samples]
    return rewards, rewards


__all__ = ["reward_func", "post_process_rewards"]
