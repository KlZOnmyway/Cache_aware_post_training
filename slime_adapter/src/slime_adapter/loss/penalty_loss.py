"""Wrap slime's ``policy_loss_function`` to add cache-aware aux terms and
to apply the per-token p_mix importance weights.

Pipeline at each training step:

    1. IS-weight rescale: ``advantages[i] *= w_t[i]``  per token
       (w_t = p_student / p_mix from rollout/mix_generate.py)
    2. slime's stock policy loss runs unchanged on the rescaled advantages
       — its PG term becomes ``−E[ratio·clip(A_t · w_t)]``, which is the
       unbiased PG estimator under p_mix sampling.
    3. We add three local terms on top:

           L_aux = λ_pg_s · L_switch_pg
                 + λ_h    · mean(max(0, used_t − budget)²)        (barrier)
                 + λ_chunk · L_chunk_consistency                   (smoothness)

       where  L_switch_pg = − E[A_t · Σ_l logπ(switch_{t,l})]      (joint actor PG)

The ``advantages`` tensors mutated in step 1 are read again in step 3, so
the SwitchHead PG also gets the IS correction automatically.
"""

from __future__ import annotations

import threading
from typing import Optional

import torch


_orig_policy_loss = None
_applied = False
_LAST_IS_MEAN = 1.0   # for monitoring; updated each step


# =============================================================================
# Apply / remove
# =============================================================================

def apply_patch() -> None:
    """Install the wrapper. Idempotent."""
    global _orig_policy_loss, _applied
    if _applied:
        return
    try:
        from slime.backends.megatron_utils import loss as _slime_loss  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "slime is not importable; install via scripts/install_externals.sh"
        ) from e
    _orig_policy_loss = _slime_loss.policy_loss_function
    _slime_loss.policy_loss_function = _wrapped_policy_loss
    _applied = True


apply_patches = apply_patch


def restore_patch() -> None:
    global _applied
    if not _applied:
        return
    from slime.backends.megatron_utils import loss as _slime_loss  # type: ignore
    if _orig_policy_loss is not None:
        _slime_loss.policy_loss_function = _orig_policy_loss
    _applied = False


# =============================================================================
# Wrapped loss
# =============================================================================

def _wrapped_policy_loss(args, batch, logits, sum_of_sample_mean):
    """Run slime's PG, add cache-aware aux terms.

    p_mix IS-correction is no longer applied here.  ``rollout/mix_generate.py``
    records ``log p_mix(token)`` (not ``log p_student``) into Sample.rollout_log_probs,
    so slime's stock TIS computes ratio = exp(log π_θ - log p_mix) = π_θ / p_mix,
    which IS the IS correction for p_mix sampling.  Similarly, teacher-KL is
    applied via slime's ``--use-opd --opd-type sglang`` (not via this wrapper).
    """
    # slime's stock PG + KL anchor + entropy + TIS handles p_mix natively
    base_loss, base_metrics = _orig_policy_loss(args, batch, logits, sum_of_sample_mean)

    state = state_or_none()
    if state is None:
        return base_loss, base_metrics

    summary = state.summary()
    device = base_loss.device if hasattr(base_loss, "device") else None
    zero = torch.zeros((), device=device) if device is not None else torch.tensor(0.0)

    # Step 3a: hinge² barrier on credits overflow
    total_credits = _resolve_total_credits(args, summary)
    overflow = (summary.total_used_per_token - total_credits).clamp_min(0.0)
    L_barrier = (overflow * overflow).mean()

    # Step 3b: chunk consistency (set by adapter)
    L_chunk_obj = getattr(state, "chunk_consistency_loss", None)
    L_chunk = L_chunk_obj if isinstance(L_chunk_obj, torch.Tensor) else zero

    # Step 3c: joint-actor PG term  − E[A_t · Σ_l logπ(switch_{t,l})]
    L_switch_pg = zero
    adv = _extract_advantages(batch)
    slp = summary.switch_logprob_per_token
    if adv is not None and slp is not None and slp.numel() > 0:
        a = adv.to(slp.device).to(slp.dtype).detach()
        if a.shape != slp.shape:
            try:
                a = a.view_as(slp)
            except RuntimeError:
                a = a.mean().expand_as(slp)
        L_switch_pg = -(a * slp).mean()

    λ_h = float(getattr(args, "barrier_lambda",     0.5))
    λ_c = float(getattr(args, "consistency_lambda", 0.05))
    λ_s = float(getattr(args, "switch_pg_lambda",   1.0))

    aux = λ_s * L_switch_pg + λ_h * L_barrier_safely(state, summary, device) + λ_c * L_chunk
    # (We compute L_barrier above already; reuse it instead of re-computing.)
    aux = λ_s * L_switch_pg + λ_h * L_barrier + λ_c * L_chunk
    total_loss = base_loss + aux

    metrics = dict(base_metrics) if base_metrics is not None else {}
    metrics.update({
        "loss/aux_total":           _scalar(aux),
        "loss/switch_pg":           _scalar(L_switch_pg),
        "loss/barrier":             _scalar(L_barrier),
        "loss/chunk_consistency":   _scalar(L_chunk),
        "rollout/cache_used_mean":      _scalar(summary.total_used_per_token.mean()),
        "rollout/cache_overflow_mean":  _scalar(overflow_safe(summary, total_credits)),
        "rollout/is_weight_mean":   float(_LAST_IS_MEAN),
    })
    return total_loss, metrics


def L_barrier_safely(state, summary, device):
    """Compatibility shim — recompute the hinge² barrier from the summary."""
    overflow = (summary.total_used_per_token - float(summary.total_credits)).clamp_min(0.0)
    return (overflow * overflow).mean()


def overflow_safe(summary, total_credits):
    """Mean overflow per token (for metric reporting only)."""
    overflow = (summary.total_used_per_token - float(total_credits)).clamp_min(0.0)
    return overflow.mean()


# =============================================================================
# IS-weight injection
# =============================================================================

_LAST_IS_STATS = {"mean": 1.0, "min": 1.0, "max": 1.0}


def _last_is_mean() -> float:
    return float(_LAST_IS_STATS["mean"])


def _apply_importance_weights_inplace(args, batch) -> None:
    """Multiply each advantage tensor by its sample's per-token w_t.

    No-op if the batch carries no metadata (e.g. when the rollout is not
    p_mix). When applied, also records ``_LAST_IS_STATS`` for telemetry.
    """
    global _LAST_IS_STATS
    advs = _extract_advantages_list(batch)
    weights = _extract_importance_weights_list(batch)
    if advs is None or weights is None:
        return
    if len(advs) != len(weights):
        return
    means = []
    for i, (a, w) in enumerate(zip(advs := advs, weights)):
        if w is None:
            continue
        wt = torch.as_tensor(w, dtype=a.dtype, device=a.device)
        if wt.shape != a.shape:
            # Best-effort: trim or pad to match
            n = min(wt.numel(), a.numel())
            a_view = a.view(-1)
            w_view = wt.view(-1)[:n]
            a_view[:n].mul_(w_view)
            means.append(float(w_view.mean().item()) if w_view.numel() else 1.0)
        else:
            a.mul_(wt)
            means.append(float(wt.mean().item()))
    if means:
        _LAST_IS_MEAN_LOCAL = sum(means) / len(means)
        global _LAST_IS_MEAN
        _LAST_IS_MEAN = _LAST_IS_MEAN_LOCAL


def _extract_advantages(batch):
    """Return per-token advantages as a single concatenated tensor (or None)."""
    advs = _extract_advantages_list(batch)
    if advs is None:
        return None
    if len(advs) == 0:
        return None
    try:
        return torch.cat(advs, dim=0)
    except RuntimeError:
        return None


def _extract_advantages_list(batch):
    if isinstance(batch, dict):
        adv = batch.get("advantages")
    else:
        adv = getattr(batch, "advantages", None)
    if adv is None:
        return None
    if isinstance(adv, list):
        return adv
    return [adv]


def _extract_importance_weights_list(batch):
    """Pull list[Tensor] of per-token IS weights from sample metadata.

    Two locations are supported:
      * ``batch["importance_weights"]`` — set by a custom slime hook
      * ``batch["samples"][i].metadata["importance_weights"]`` — set by
        ``slime_adapter.rollout.mix_generate``
    """
    if isinstance(batch, dict):
        direct = batch.get("importance_weights")
        if direct is not None:
            return direct if isinstance(direct, list) else [direct]
        samples = batch.get("samples")
    else:
        direct = getattr(batch, "importance_weights", None)
        if direct is not None:
            return direct if isinstance(direct, list) else [direct]
        samples = getattr(batch, "samples", None)

    if samples is None:
        return None
    weights = []
    for s in samples:
        md = getattr(s, "metadata", None) or {}
        w = md.get("importance_weights")
        if w is None:
            return None
        weights.append(torch.as_tensor(w))
    return weights


def _apply_importance_weights_inplace(args, batch) -> None:  # noqa: F811 - canonical
    """Multiply each advantage tensor by its per-token w_t in place.

    No-op when the batch carries no IS weights; safe to call always.
    """
    global _LAST_IS_STATS, _LAST_IS_MEAN
    advs = _extract_advantages_list(batch)
    ws = _extract_importance_weights_list(batch)
    if advs is None or ws is None or len(advs) != len(ws):
        _LAST_IS_MEAN = 1.0
        return

    all_w = []
    for a, w in zip(advs, ws):
        if w is None:
            continue
        wt = torch.as_tensor(w, dtype=a.dtype, device=a.device)
        if wt.numel() != a.numel():
            n = min(wt.numel(), a.numel())
            wt = wt.view(-1)[:n]
            a_flat = a.view(-1)
            a_flat[:n].mul_(wt)
            all_w.append(wt)
        else:
            wt = wt.view_as(a)
            a.mul_(wt)
            all_w.append(wt)
    if all_w:
        cat = torch.cat([w.view(-1) for w in all_w], dim=0)
        _LAST_IS_MEAN = float(cat.mean().item())
        _LAST_IS_STATS = {
            "mean": float(cat.mean().item()),
            "min":  float(cat.min().item()),
            "max":  float(cat.max().item()),
        }


# =============================================================================
# TLS handoff (forward driver writes; loss patch reads)
# =============================================================================

_TLS = threading.local()


def set_last_controller_state(state) -> None:
    _TLS.state = state


def state_or_none():
    return getattr(_TLS, "state", None)


def clear_last_controller_state() -> None:
    if hasattr(_TLS, "state"):
        del _TLS.state


# =============================================================================
# Helpers
# =============================================================================

def _scalar(t) -> float:
    if hasattr(t, "detach"):
        return float(t.detach().item())
    return float(t)


def _resolve_total_credits(args, summary) -> float:
    val = getattr(args, "total_credits", None)
    if val is not None:
        return float(val)
    if getattr(summary, "total_credits", None):
        return float(summary.total_credits)
    fraction = float(getattr(args, "budget_fraction", 0.7))
    n_layers = max(1, len(getattr(summary, "layer_local_costs", []))
                   or int(getattr(args, "num_moe_layers", 1)))
    return fraction * n_layers


def _extract_advantages_list(batch):
    """Return ``batch["advantages"]`` as a list of tensors (or None)."""
    if isinstance(batch, dict):
        adv = batch.get("advantages")
    else:
        adv = getattr(batch, "advantages", None)
    if adv is None:
        return None
    if isinstance(adv, list):
        return adv
    return [adv]


__all__ = [
    "apply_patch",
    "apply_patches",
    "restore_patch",
    "set_last_controller_state",
    "state_or_none",
    "clear_last_controller_state",
]


# =============================================================================
# Helpers (small)
# =============================================================================

def _scalar(t) -> float:
    if hasattr(t, "detach"):
        try:
            return float(t.detach().item())
        except Exception:
            return float(t.detach().mean().item())
    return float(t)


def _resolve_total_credits(args, summary) -> float:  # noqa: F811
    val = getattr(args, "total_credits", None)
    if val is not None:
        return float(val)
    if getattr(summary, "total_credits", None):
        return float(summary.total_credits)
    fraction = float(getattr(args, "budget_fraction", 0.7))
    n_layers = max(1, len(getattr(summary, "layer_local_costs", []) or [])
                   or int(getattr(args, "num_moe_layers", 1)))
    return fraction * n_layers
