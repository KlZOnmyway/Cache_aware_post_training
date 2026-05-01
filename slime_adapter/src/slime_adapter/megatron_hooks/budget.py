"""Autograd-aware credit accountant + per-token switch log-probability tracker.

Lives on the model object during one training forward; allocated by
``LayerBudgetTracker.begin(...)`` from a forward pre-hook.

What it accumulates per token (shape ``[B, T]``):

  • ``used_so_far``        — running Σ_l (switch_l · n_new_l) cost
  • ``layer_costs``        — list of per-layer ``[B, T]`` costs
  • ``switch_logprob_total``  — running Σ_l Bernoulli log-prob of the switch
                              decision under σ_l (autograd-attached via σ),
                              consumed by the joint-actor PG term in the loss.

Why the log-prob lives here: the SwitchHead is a per-layer Bernoulli policy.
For GRPO with the joint action (token, switch_1..L), the per-token policy
log-probability is::

    log π_θ(joint_t | s_t) = log π_token(token_t)  +  Σ_l Bernoulli(switch_{t,l}; σ_{t,l})

Slime's standard PG already handles the token logπ. We add the Σ_l Bernoulli
piece via this state, and the loss patch multiplies it by the per-token
advantage.

The pressure feature passed to ``SwitchHead`` is ``.detach()``ed — early
layers never see late-layer gradients via the budget bookkeeping. Temporal
credit assignment goes through the discounted return (in the reward), not
through this tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class BudgetReadout:
    """Read-only snapshot consumed by the loss patch."""

    layer_local_costs: List[torch.Tensor]      # list of [B, T]
    total_used_per_token: torch.Tensor         # [B, T]
    total_credits: float
    switch_logprob_per_token: torch.Tensor     # [B, T] — Σ_l Bernoulli logπ

    @property
    def overflow_per_token(self) -> torch.Tensor:
        """``max(0, total_used - total_credits)`` — the hinge² barrier input."""
        return (self.total_used_per_token - self.total_credits).clamp_min(0.0)


@dataclass
class TokenBudgetState:
    """Per-forward, per-(B,T) accumulator. Built fresh by ``LayerBudgetTracker``."""

    total_credits: float
    used_so_far: torch.Tensor                              # [B, T]
    layer_costs: List[torch.Tensor] = field(default_factory=list)
    switch_logprob_total: torch.Tensor | None = None       # [B, T]
    chunk_consistency_loss: torch.Tensor | None = None     # scalar (set by adapter)

    # ----- forward-time API used by each MoE layer's wrapped forward -----

    def pressure_at_entry(self) -> torch.Tensor:
        """Forward-only feature → SwitchHead. Detached."""
        return (self.used_so_far / self.total_credits).detach()

    def charge_layer(
        self,
        switch_signal: torch.Tensor,
        n_new_int: torch.Tensor,
    ) -> torch.Tensor:
        """Legacy entry: only updates the cost. No log-prob accumulated.

        Use ``charge_layer_with_logp`` instead for the joint-actor PG term.
        """
        cost = switch_signal * n_new_int.to(switch_signal.dtype)
        self.used_so_far = self.used_so_far + cost
        self.layer_costs.append(cost)
        return cost

    def charge_layer_with_logp(
        self,
        switch_signal: torch.Tensor,                        # [B, T] hard 0/1 (post-STE)
        n_new_int: torch.Tensor,                            # [B, T] long
        sigma: torch.Tensor,                                # [B, T] in (0, 1) pre-STE
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Charge cost AND accumulate Bernoulli log-prob of the switch action.

        Returns the per-layer cost ``[B, T]``; mutates ``self.used_so_far``,
        ``self.layer_costs``, and ``self.switch_logprob_total``.

        The log-prob path lets ``L_PG_switch = -E[A_t · Σ_l logπ(switch_{t,l})]``
        flow gradient back to ``SwitchHead`` through ``σ`` directly (no STE
        in this term — STE is only used for the forward decision).
        """
        cost = switch_signal * n_new_int.to(switch_signal.dtype)
        self.used_so_far = self.used_so_far + cost
        self.layer_costs.append(cost)

        sig = sigma.clamp(eps, 1.0 - eps)
        logp_l = switch_signal * torch.log(sig) + (1.0 - switch_signal) * torch.log1p(-sig)
        if self.switch_logprob_total is None:
            self.switch_logprob_total = logp_l
        else:
            self.switch_logprob_total = self.switch_logprob_total + logp_l
        return cost

    # ----- read-only summary for the loss patch -----
    def summary(self) -> BudgetReadout:
        slp = (self.switch_logprob_total
               if self.switch_logprob_total is not None
               else torch.zeros_like(self.used_so_far))
        return BudgetReadout(
            layer_local_costs=list(self.layer_costs),
            total_used_per_token=self.used_so_far,
            total_credits=self.total_credits,
            switch_logprob_per_token=slp,
        )


class LayerBudgetTracker:
    """Forward-pass-scoped factory for ``TokenBudgetState``."""

    def __init__(self, total_credits: float):
        if total_credits <= 0:
            raise ValueError(f"total_credits must be > 0, got {total_credits}")
        self._total_credits = float(total_credits)

    @property
    def total_credits(self) -> float:
        return self._total_credits

    @total_credits.setter
    def total_credits(self, v: float) -> None:
        self._total_credits = float(v)

    def begin(self, hidden_states: torch.Tensor) -> TokenBudgetState:
        B, T = hidden_states.shape[:2]
        zeros = torch.zeros(B, T, dtype=hidden_states.dtype, device=hidden_states.device)
        return TokenBudgetState(total_credits=self.total_credits, used_so_far=zeros)

    @classmethod
    def from_args(cls, args, num_moe_layers: int) -> "LayerBudgetTracker":
        fraction = float(getattr(args, "budget_fraction", 0.7))
        return cls(total_credits=fraction * num_moe_layers)


__all__ = [
    "BudgetReadout",
    "TokenBudgetState",
    "LayerBudgetTracker",
]
