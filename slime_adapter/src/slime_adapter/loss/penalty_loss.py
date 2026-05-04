"""Wrap slime's ``policy_loss_function`` to add cache-aware aux terms.

Pipeline at each training step:

    1. slime's stock PG loss runs unchanged — its TIS mechanism computes
       ratio = π_θ / p_mix from rollout_log_probs, providing the IS correction
       for p_mix sampling automatically.
    2. We add three local terms on top:

           L_aux = λ_s · L_switch_pg
                 + λ_h · mean(max(0, used_t − budget)²)        (barrier)
                 + λ_c · L_chunk_consistency                    (smoothness)

       where  L_switch_pg = − E[A_i · mean_t(Σ_l logπ(switch_{t,l}))]  (joint actor PG)
"""

from __future__ import annotations

import logging
import threading

import torch

logger = logging.getLogger(__name__)

_orig_policy_loss = None
_applied = False
_LAST_IS_MEAN = 1.0


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
    """Run slime's PG, add cache-aware aux terms."""
    base_loss, base_metrics = _orig_policy_loss(args, batch, logits, sum_of_sample_mean)

    state = state_or_none()
    if state is None:
        return base_loss, base_metrics

    summary = state.summary()
    device = base_loss.device if hasattr(base_loss, "device") else None
    zero = torch.zeros((), device=device) if device is not None else torch.tensor(0.0)

    # Hinge² barrier on credits overflow
    total_credits = _resolve_total_credits(args, summary)
    overflow = (summary.total_used_per_token - total_credits).clamp_min(0.0)
    L_barrier = (overflow * overflow).mean()

    # Chunk consistency (set by adapter forward)
    L_chunk_obj = getattr(state, "chunk_consistency_loss", None)
    L_chunk = L_chunk_obj if isinstance(L_chunk_obj, torch.Tensor) else zero

    # Joint-actor PG term: − E_i[A_i · mean_t(Σ_l logπ(switch_{t,l}))]
    L_switch_pg = _compute_switch_pg(batch, summary, zero)

    λ_h = float(getattr(args, "barrier_lambda",     0.5))
    λ_c = float(getattr(args, "consistency_lambda", 0.05))
    λ_s = float(getattr(args, "switch_pg_lambda",   1.0))

    aux = λ_s * L_switch_pg + λ_h * L_barrier + λ_c * L_chunk
    total_loss = base_loss + aux

    metrics = dict(base_metrics) if base_metrics is not None else {}
    metrics.update({
        "loss/aux_total":           _scalar(aux),
        "loss/switch_pg":           _scalar(L_switch_pg),
        "loss/barrier":             _scalar(L_barrier),
        "loss/chunk_consistency":   _scalar(L_chunk),
        "rollout/cache_used_mean":  _scalar(summary.total_used_per_token.mean()),
        "rollout/cache_overflow_mean": _scalar(
            (summary.total_used_per_token - float(total_credits)).clamp_min(0.0).mean()
        ),
        "rollout/is_weight_mean":   float(_LAST_IS_MEAN),
    })
    return total_loss, metrics


def _compute_switch_pg(batch, summary, zero) -> torch.Tensor:
    """Per-sample joint-actor PG: −E_i[A_i · mean_t(switch_logprob_i)].

    slime's advantages are per-sample 1D tensors (response-only, variable
    length). switch_logprob_per_token is [B, T] covering the full padded
    sequence. We compute per-sample means to align them correctly.
    """
    slp = summary.switch_logprob_per_token
    if slp is None or slp.numel() == 0:
        return zero

    advs = _extract_advantages_list(batch)
    if advs is None or len(advs) == 0:
        return zero

    B = slp.shape[0]

    # Per-sample mean advantage (GRPO advantages are per-trajectory constant,
    # so .mean() just extracts the scalar).
    sample_advs = []
    for a in advs:
        sample_advs.append(a.to(slp.device).to(slp.dtype).detach().mean())

    if len(sample_advs) != B:
        # Batch size mismatch — can happen with gradient accumulation.
        # Fall back to global mean.
        if len(sample_advs) == 0:
            return zero
        a_mean = torch.stack(sample_advs).mean()
        return -(a_mean * slp.mean())

    a_per_sample = torch.stack(sample_advs)                    # [B]
    slp_per_sample = slp.mean(dim=-1)                          # [B]
    return -(a_per_sample * slp_per_sample).mean()


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
        try:
            return float(t.detach().item())
        except Exception:
            return float(t.detach().mean().item())
    return float(t)


def _resolve_total_credits(args, summary) -> float:
    val = getattr(args, "total_credits", None)
    if val is not None:
        return float(val)
    if getattr(summary, "total_credits", None):
        return float(summary.total_credits)
    fraction = float(getattr(args, "budget_fraction", 0.7))
    n_layers = max(1, len(getattr(summary, "layer_local_costs", []) or [])
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
