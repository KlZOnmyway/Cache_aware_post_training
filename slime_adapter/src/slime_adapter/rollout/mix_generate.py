"""Teacher-mixed rollout (MiniLLM-style p_mix) for slime + SGLang.

Per-token sampling from p_mix = (1 - alpha) * p_student + alpha * p_teacher.
Importance weight w_t = p_student(token) / p_mix(token) recorded on each
sample under ``Sample.metadata['importance_weights']``. The trainer
(``slime_adapter.loss.penalty_loss``) multiplies w_t into the GRPO advantage,
so the policy gradient is unbiased w.r.t. the student even though we sample
from the mixture.

Why p_mix beats KL anchor at preventing reward hacking
    The KL anchor only pulls the policy toward the teacher in logit space.
    The student can still find token-space "blind spots" (regions of low
    teacher mass) that the reward (e.g. cache cost) trivially exploits.
    Sampling from p_mix forces every emitted token to lie in the teacher's
    effective support, so the trajectory never enters those regions.

Reference: Gu et al., MiniLLM (https://arxiv.org/abs/2306.08543), Section 2.2.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import aiohttp
import torch

logger = logging.getLogger(__name__)

_NEG_INF = -1e30
_PROB_FLOOR = 1e-30


# =====================================================================
# SGLang HTTP helper
# =====================================================================

async def query_topk_logprobs(
    session: aiohttp.ClientSession,
    url: str,
    input_ids: Sequence[int],
    top_k: int,
    temperature: float = 1.0,
) -> dict:
    """Ask one SGLang server for top-K next-token logprobs.

    Returns ``{token_id: logprob}``. Uses ``max_new_tokens=1`` and reads
    ``meta_info.output_top_logprobs[0]``.
    """
    payload = {
        "input_ids": [int(t) for t in input_ids],
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": float(temperature),
        },
        "return_logprob": True,
        "top_logprobs_num": int(top_k),
    }
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()

    meta = data.get("meta_info") or {}
    top = meta.get("output_top_logprobs") or []
    if top and top[0]:
        return {int(item[1]): float(item[0]) for item in top[0]}

    # Fallback: just the sampled token's logprob.
    out = meta.get("output_token_logprobs") or []
    if out:
        return {int(out[0][1]): float(out[0][0])}
    return {}


# =====================================================================
# Mix-and-sample
# =====================================================================

def sample_p_mix(
    student_top: dict,
    teacher_top: dict,
    alpha: float,
) -> Tuple[int, float, float, float]:
    """Form p_mix over the union of top-K supports and draw one token.

    Returns ``(token_id, w_t, log_p_S, log_p_mix)`` where:
        w_t       = p_S(token) / p_mix(token)
        log_p_S   = log p_S(token)   (monitoring / diagnostics)
        log_p_mix = log p_mix(token) (stored in rollout_log_probs for TIS)
    """
    if not student_top and not teacher_top:
        raise RuntimeError("p_mix: both top-K dicts empty; SGLang returned no logprobs")

    vocab = sorted(set(student_top.keys()) | set(teacher_top.keys()))
    log_s_list = [float(student_top.get(t, _NEG_INF)) for t in vocab]
    log_t_list = [float(teacher_top.get(t, _NEG_INF)) for t in vocab]

    log_S = torch.tensor(log_s_list, dtype=torch.float64)
    log_T = torch.tensor(log_t_list, dtype=torch.float64)
    p_S = torch.softmax(log_S, dim=0)
    p_T = torch.softmax(log_T, dim=0)
    p_mix = (1.0 - float(alpha)) * p_S + float(alpha) * p_T

    idx = int(torch.multinomial(p_mix, num_samples=1).item())
    token = int(vocab[idx])

    p_S_val = float(p_S[idx].clamp_min(_PROB_FLOOR))
    p_mix_val = float(p_mix[idx].clamp_min(_PROB_FLOOR))
    w_t = p_S_val / p_mix_val
    # Returns (token, log_p_mix, log_p_S):
    #   log_p_mix → goes into Sample.rollout_log_probs (slime's TIS uses
    #               ratio = exp(logπ_θ − rollout_log_probs), which becomes
    #               π_θ / p_mix — the IS correction we want for free).
    #   log_p_S   → kept on metadata for diagnostics / monitoring (sanity
    #               check that w_t = p_S/p_mix lies in a sane range).
    log_p_mix = float(torch.log(p_mix[idx].clamp_min(_PROB_FLOOR)))
    log_p_S = float(torch.log(p_S[idx].clamp_min(_PROB_FLOOR)))
    return token, w_t, log_p_S, log_p_mix


# =====================================================================
# One trajectory: token-by-token p_mix loop
# =====================================================================

async def generate_one_p_mix(
    session: aiohttp.ClientSession,
    *,
    student_url: str,
    teacher_url: str,
    prompt_ids: Sequence[int],
    alpha: float,
    top_k: int,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
) -> dict:
    """Run one trajectory under p_mix sampling.

    Returns a dict::

        tokens              : full sequence (prompt + response)
        response_length     : number of generated tokens
        response_tokens     : just the generated tokens
        rollout_log_probs   : per-step log p_student(chosen_token)
        importance_weights  : per-step w_t = p_S/p_mix
    """
    ids = [int(t) for t in prompt_ids] if prompt_ids is not None else []
    response_tokens: List[int] = []
    rollout_log_probs: List[float] = []         # log p_mix(token) — slime TIS reads this
    student_log_probs: List[float] = []         # log p_S(token)  — monitoring only
    importance_weights: List[float] = []        # w_t = p_S/p_mix — monitoring only

    for _ in range(int(max_new_tokens)):
        student_top, teacher_top = await asyncio.gather(
            query_topk_logprobs(session, student_url, ids, top_k, temperature),
            query_topk_logprobs(session, teacher_url, ids, top_k, temperature),
        )
        token, w_t, log_p_S, log_p_mix = sample_p_mix(student_top, teacher_top, alpha)
        ids.append(token)
        response_tokens.append(token)
        # Slime PG computes ratio = exp(log_probs(θ) − rollout_log_probs);
        # by storing log p_mix here we make ratio = π_θ / p_mix — exactly the
        # IS correction p_mix sampling requires. No metadata transfer needed.
        rollout_log_probs.append(log_p_mix)
        student_log_probs.append(log_p_S)
        importance_weights.append(w_t)
        if eos_token_id is not None and token == int(eos_token_id):
            break

    return {
        "tokens": ids,
        "response_length": len(response_tokens),
        "response_tokens": response_tokens,
        "rollout_log_probs": rollout_log_probs,
        "student_log_probs": student_log_probs,
        "importance_weights": importance_weights,
    }


# =====================================================================
# Slime entry point
# =====================================================================



async def generate_rollout(args, rollout_id, data_source, evaluation: bool = False):
    """slime-compatible drop-in for ``--rollout-function-path``.

    The ``data_source`` argument is a slime ``RolloutDataSource[WithBuffer]``
    (see ``slime/rollout/data_source.py``). We call ``get_samples(num_samples)``
    which returns ``list[list[Sample]]`` — one inner list per prompt, with
    ``n_samples_per_prompt`` already pre-replicated into Sample objects whose
    ``.tokens / .prompt / .label`` are populated by slime's tokenizer.

    Each Sample is then run through ``_one_trajectory`` to do per-token p_mix
    sampling and post-rollout teacher-logprob fetch, mutating the Sample in
    place. Returns the flat list of completed Samples.
    """
    from slime.utils.types import Sample  # type: ignore (lazy import; slime may not be present in tests)

    alpha = float(getattr(args, "teacher_mix_alpha", 0.5))
    top_k = int(getattr(args, "mix_top_k", 64))
    max_new = int(getattr(args, "rollout_max_response_len", 1024))
    temperature = float(getattr(args, "rollout_temperature", 1.0))
    eos = int(getattr(args, "eos_token_id", 151643))  # Qwen3 <|endoftext|>

    student_url = "http://{}:{}/generate".format(
        getattr(args, "sglang_router_ip", "127.0.0.1"),
        getattr(args, "sglang_router_port", 30000),
    )
    teacher_url = str(getattr(args, "rm_url", "")).rstrip("/")
    if not teacher_url:
        raise ValueError("--rm-url must point at the teacher SGLang server for p_mix rollout")
    if not teacher_url.endswith("/generate"):
        teacher_url = teacher_url + "/generate"

    # slime's data source API: get_samples(num_samples) → list[list[Sample]]
    # The size we ask for is rollout_batch_size (default 1 group at a time).
    num_groups = int(getattr(args, "rollout_batch_size", 1))
    prompt_groups = data_source.get_samples(num_groups)

    timeout = aiohttp.ClientTimeout(total=max_new * 2.0 + 60.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for group in prompt_groups:
            for sample in group:
                tasks.append(_one_trajectory(
                    session=session,
                    sample=sample,
                    student_url=student_url,
                    teacher_url=teacher_url,
                    alpha=alpha,
                    top_k=top_k,
                    max_new_tokens=max_new,
                    eos_token_id=eos,
                    temperature=temperature,
                ))
        completed = await asyncio.gather(*tasks)
    return [s for s in completed if s is not None]


async def _one_trajectory(
    *,
    session: aiohttp.ClientSession,
    sample,                                  # slime Sample (in/out)
    student_url: str,
    teacher_url: str,
    alpha: float,
    top_k: int,
    max_new_tokens: int,
    eos_token_id: int,
    temperature: float,
):
    """Mutate a slime ``Sample`` in place with p_mix rollout results.

    The Sample arrives from ``data_source.get_samples`` already populated
    with ``prompt`` (string) and ``tokens`` (the tokenised prompt — slime's
    Dataset has run the chat template). We extend it by appending generated
    tokens to ``sample.tokens`` and filling ``rollout_log_probs``,
    ``teacher_log_probs``, and ``metadata['importance_weights']``.
    """
    prompt_ids = list(sample.tokens) if sample.tokens else []
    if not prompt_ids:
        logger.warning("p_mix: empty prompt; skipping sample group=%s idx=%s",
                       getattr(sample, "group_index", None),
                       getattr(sample, "index", None))
        return None

    out = await generate_one_p_mix(
        session,
        student_url=student_url,
        teacher_url=teacher_url,
        prompt_ids=prompt_ids,
        alpha=alpha,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
    )

    teacher_lp = await _fetch_teacher_logprobs_full(
        session=session,
        teacher_url=teacher_url,
        tokens=out["tokens"],
        response_length=out["response_length"],
    )

    # Mutate the Sample in place — slime expects this object back
    sample.tokens = out["tokens"]
    sample.response_length = out["response_length"]
    sample.rollout_log_probs = out["rollout_log_probs"]            # log p_mix per step
    if teacher_lp is not None:
        sample.teacher_log_probs = teacher_lp
    sample.metadata = dict(getattr(sample, "metadata", None) or {})
    sample.metadata.update({
        "importance_weights": out["importance_weights"],
        "student_log_probs":  out["student_log_probs"],
        "p_mix_alpha":        float(alpha),
    })
    # slime's Sample.Status enum
    try:
        from slime.utils.types import Sample as _Sample  # type: ignore
        sample.status = _Sample.Status.COMPLETED
    except Exception:
        pass
    return sample


async def _fetch_teacher_logprobs_full(
    *,
    session: aiohttp.ClientSession,
    teacher_url: str,
    tokens: Sequence[int],
    response_length: int,
) -> Optional[list[float]]:
    """One-shot teacher-logprob fetch over the full completed trajectory.

    SGLang returns ``meta_info.input_token_logprobs`` for every position when
    ``max_new_tokens=0`` and ``return_logprob=True``. We drop the leading BOS
    entry and slice the trailing ``response_length`` values so the returned
    list aligns with the response tokens (not the prompt).
    """
    payload = {
        "input_ids": [int(t) for t in tokens],
        "sampling_params": {"max_new_tokens": 0, "temperature": 1.0},
        "return_logprob": True,
    }
    try:
        async with session.post(teacher_url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as e:
        logger.warning("p_mix: teacher-logprob fetch failed: %s", e)
        return None
    meta = (data or {}).get("meta_info") or {}
    raw = meta.get("input_token_logprobs") or []
    if not raw:
        return None
    vals = [
        float(item[0]) if isinstance(item, (list, tuple)) else float(item)
        for item in raw[1:]
    ]
    if response_length > 0 and len(vals) >= response_length:
        vals = vals[-response_length:]
    return vals




__all__ = [
    "generate_rollout",
    "generate_one_p_mix",
    "query_topk_logprobs",
    "sample_p_mix",
]
