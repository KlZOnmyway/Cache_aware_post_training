"""Autograd-aware per-token credit accountant for the training-side forward.

Lives next to the forward path: one instance per forward pass. As we walk
the model's MoE layers in order, each layer:

  1. reads ``state.pressure_at_entry()`` (forward-only, detached) to feed into
     its ``SwitchHead``;
  2. computes ``switch_signal`` (via STE);
  3. asks the cache for ``n_new``;
  4. calls ``state.charge_layer(switch_signal, n_new)`` to update the running
     ``used_so_far`` *with autograd attached*; this returns the per-layer
     cost ``[B, T]`` that the loss aggregates.

When all layers are done, the loss reads ``state.summary()`` to get:

  - ``layer_local_costs`` → ``Σ_l cost_l`` becomes the uniform per-switch cost
  - ``total_used_per_token`` → used by the hinge² barrier
  - ``overflow_per_token``    → ``clamp(total_used - total_credits, min=0)``

The pressure tensor exposed to ``SwitchHead`` is **detached**, so gradients
flow into σ_l only via the local ``cost`` term (and via the barrier when the
token overflows). Earlier layers' switch decisions don't gradient-leak into
later layers' σ through pressure — exactly the per-layer cleanliness we want.

Cross-layer credit assignment goes through the **KL gradient through the
hidden chain**, not through the budget. See PORT_TO_SLIME.md §1 / §3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class BudgetReadout:
    """What the loss patch consumes after the forward pass finishes."""

    layer_local_costs: List[torch.Tensor]   # list of [B, T], one per MoE layer
    total_used_per_token: torch.Tensor       # [B, T]
    total_credits: float

    @property
    def overflow_per_token(self) -> torch.Tensor:
        """``max(0, total_used - total_credits)`` — for the hinge² barrier."""
        return (self.total_used_per_token - self.total_credits).clamp_min(0.0)


@dataclass
class TokenBudgetState:
    """Per-forward, per-(B,T) running budget. Created fresh by ``LayerBudgetTracker``."""

    total_credits: float
    used_so_far: torch.Tensor                              # [B, T]
    layer_costs: List[torch.Tensor] = field(default_factory=list)

    def pressure_at_entry(self) -> torch.Tensor:
        """Forward-only feature passed into next ``SwitchHead`` (detached)."""
        return (self.used_so_far / self.total_credits).detach()

    def charge_layer(self, switch_signal: torch.Tensor, n_new_int: torch.Tensor) -> torch.Tensor:
        """Charge one layer's cost; returns ``[B, T]`` cost (autograd-attached via STE)."""
        cost = switch_signal * n_new_int.to(switch_signal.dtype)
        # use += would mutate the autograd graph; use functional add
        self.used_so_far = self.used_so_far + cost
        self.layer_costs.append(cost)
        return cost

    def summary(self) -> BudgetReadout:
        return BudgetReadout(
            layer_local_costs=list(self.layer_costs),
            total_used_per_token=self.used_so_far,
            total_credits=self.total_credits,
        )


class LayerBudgetTracker:
    """Forward-pass-scoped factory for ``TokenBudgetState``.

    Lives on the model object (or in the forward closure) for the duration of
    a single forward pass.
    """

    def __init__(self, total_credits: float):
        if total_credits <= 0:
            raise ValueError(f"total_credits must be > 0, got {total_credits}")
        self.total_credits = float(total_credits)

    def begin(self, hidden_states: torch.Tensor) -> TokenBudgetState:
        """Allocate a fresh per-(B,T) running state with the given dtype/device."""
        B, T = hidden_states.shape[:2]
        zeros = torch.zeros(B, T, dtype=hidden_states.dtype, device=hidden_states.device)
        return TokenBudgetState(total_credits=self.total_credits, used_so_far=zeros)

    @classmethod
    def from_args(cls, args, num_moe_layers: int) -> "LayerBudgetTracker":
        """Build from CLI args: ``total_credits = budget_fraction × num_moe_layers``."""
        fraction = float(getattr(args, "budget_fraction", 0.7))
        return cls(total_credits=fraction * num_moe_layers)  # type: ignore[arg-type]

    @property
    def total_credits(self) -> float:  # noqa: F811 (override property accessor)
        return self._total_credits

    @total_credits.setter
    def total_credits(self, v: float) -> None:
        self._total_credits = float(v)
