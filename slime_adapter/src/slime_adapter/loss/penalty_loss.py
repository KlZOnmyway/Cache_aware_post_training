"""Monkey-patch slime's ``policy_loss_function`` to add controller aux terms.

slime's policy loss (slime ≥ 0.2.4) sits at
``slime.backends.megatron_utils.loss.policy_loss_function`` with signature::

    policy_loss_function(args, batch, logits, sum_of_sample_mean) -> (loss, metrics)

We wrap it: after slime computes its own PG loss, we add::

    L_total = L_slime
            + λ_b · mean( Σ_l switch(t,l) · n_new(t,l) )           # uniform per-switch cost
            + λ_h · mean( max(0, total_used(t) - total_credits)² )  # token barrier
            + λ_c · L_chunk_consistency                              # router smoothness

The aux terms come from the most recent forward's ``TokenBudgetState``,
written into a thread-local by ``slime_adapter.megatron_hooks.driver``.
The hyperparameters (``budget_lambda``, ``barrier_lambda``, ``consistency_lambda``,
``total_credits``) are read from slime's ``args`` namespace.

Apply once at trainer startup::

    import slime_adapter.loss.penalty_loss as _pl
    _pl.apply_patch()
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import torch


_orig_policy_loss: Optional[Callable] = None
_applied: bool = False


# =============================================================================
# Apply / restore
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
            "slime is not importable. Install via "
            "`bash scripts/install_externals.sh slime` first."
        ) from e
    _orig_policy_loss = _slime_loss.policy_loss_function
    _slime_loss.policy_loss_function = _patched_policy_loss
    _applied = True


def restore_patch() -> None:
    global _orig_policy_loss, _applied
    if not _applied:
        return
    from slime.backends.megatron_utils import loss as _slime_loss  # type: ignore
    if _orig_policy_loss is not None:
        _slime_loss.policy_loss_function = _orig_policy_loss
    _orig_policy_loss = None
    _applied = False


apply_patch = apply_patches = apply_patch  # canonical name + alias


# =============================================================================
# The wrapper
# =============================================================================

def _patched_policy_loss(args, batch, logits, sum_of_sample_mean):
    """Same signature as slime's ``policy_loss_function``."""
    base_loss, base_metrics = _orig_policy_loss(args, batch, logits, sum_of_sample_mean)

    state = _get_last_state()
    if state is None:
        return base_loss, base_metrics

    summary = state.summary()
    device = base_loss.device if torch.is_tensor(base_loss) else None

    # term 1: uniform per-switch cost
    if summary.layer_local_costs:
        L_budget = torch.stack(summary.layer_local_costs, dim=0).sum(dim=0).mean()
    else:
        L_budget = torch.zeros((), device=device)

    # term 2: token-level hinge² barrier
    total_credits = _resolve_total_credits(args, summary)
    overflow = (summary.total_used_per_token - total_credits).clamp_min(0.0)
    L_barrier = (overflow * overflow).mean()

    # term 3: chunk routing consistency
    L_chunk = getattr(state_at_loss(), "chunk_consistency_loss", None)
    if L_chunk is None:
        L_chunk = torch.zeros((), device=device)

    λ_b = float(getattr(args, "budget_lambda", 0.05))
    λ_h = float(getattr(args, "barrier_lambda", 0.5))
    λ_c = float(getattr(args, "consistency_lambda", 0.05))

    aux = λ_b * L_budget + λ_h * L_barrier + λ_c * L_chunk
    total_loss = base_loss + aux

    metrics = dict(base_metrics) if (base_metrics := _last_base_metrics()) is not None else {}
    metrics.update({
        "loss/aux": _scalar(aux),
        "loss/budget_cost": _scalar(L_budget),
        "loss/barrier": _scalar(L_barrier),
        "loss/chunk_consist": _scalar(L_chunk),
    })
    return total_loss, metrics


def _patched_policy_loss(args, batch, logits, sum_of_sample_mean):  # alias
    return _wrapped_policy_loss(args, batch, logits, sum_of_sample_mean)


def _wrapped_policy_loss(args, batch, logits, sum_of_sample_mean):
    base_loss, base_metrics = _orig_policy_loss(args, batch, logits, sum_of_sample_mean)
    state = state_or_none()
    if state is None:
        return base_loss, base_metrics

    summary = state.summary()
    device = base_loss.device if torch.is_tensor(base_loss) else None

    if summary.layer_local_costs:
        L_budget = torch.stack(summary.layer_local_costs, dim=0).sum(dim=0).mean()
    else:
        L_budget = torch.zeros((), device=device)

    total_credits = _resolve_total_credits(args, summary)
    overflow = (summary.total_used_per_token - total_credits).clamp_min(0.0)
    L_barrier = (overflow * overflow).mean()

    L_chunk = getattr(state, "chunk_consistency_loss", None)
    if L_chunk is None:
        L_chunk = torch.zeros((), device=device)

    λ_b = float(getattr(args, "budget_lambda", 0.05))
    λ_h = float(getattr(args, "barrier_lambda", 0.5))
    λ_c = float(getattr(args, "consistency_lambda", 0.05))

    aux = λ_b * L_budget + λ_h * L_barrier + λ_c * L_chunk
    total_loss = base_loss + aux

    metrics = dict(base_metrics) if base_metrics is not None else {}
    metrics.update({
        "loss/aux_total": _scalar(aux),
        "loss/budget_cost": _scalar(L_budget),
        "loss/barrier": _scalar(L_barrier),
        "loss/chunk_consist": _scalar(L_chunk),
    })
    return total_loss, metrics


# Final canonical wrapper — replaces the placeholder above.
_patched_policy_loss = _wrapped_policy_loss


# =============================================================================
# Cross-cutting state passing
# =============================================================================

_LAST_STATE_TLS = threading.local()


def set_last_controller_state(state) -> None:
    """Called by the forward driver after each forward pass."""
    _LAST_STATE_TLS.state = state


def state_or_none():
    return getattr(_LAST_STATE_TLS, "state", None)


def clear_last_controller_state() -> None:
    if hasattr(_LAST_STATE_TLS, "state"):
        del _LAST_STATE_TLS.state


def _get_last_state():
    return state_or_none()


def state_at_loss():
    return state_or_none()


# =============================================================================
# Helpers
# =============================================================================

def _scalar(t) -> float:
    if torch.is_tensor(t):
        return float(t.detach().item())
    return float(t)


def _resolve_total_credits(args, summary) -> float:
    val = getattr(args, "total_credits", None)
    if val is not None and float(val) > 0:
        return float(val)
    if summary.total_credits > 0:
        return float(summary.total_credits)
    fraction = float(getattr(args, "budget_fraction", 0.7))
    L = max(1, len(summary.layer_local_costs) or int(getattr(args, "num_moe_layers", 1)))
    return fraction * L


def _last_base_metrics():
    """Placeholder kept for compatibility with older imports; not used."""
    return None
