"""Per-token credit budget tracker.

Budget semantics
----------------

At each new generation token, we reset ``used`` to 0. As the model goes through
its MoE layers in order (l=0..L-1), each layer that decides to switch consumes
``n_new`` credits (the number of new experts that need loading vs the layer's
cache). Total credits per token = ``0.7 × num_moe_layers`` by default.

This tracker is the **scalar** (rollout-side) version: simple Python floats,
no autograd. The Megatron training-side version that keeps the running
``used`` as a tensor (so STE-backed gradient can flow into ``switch_l``)
lives in ``slime_adapter.megatron_hooks.budget``.

Typical usage on the rollout path::

    tracker = CreditsTracker.from_config(num_moe_layers=24, fraction=0.7)
    for token in generate_loop:
        tracker.reset_for_new_token()
        for layer in moe_layers:
            pressure = tracker.pressure  # entry-time feature for SwitchHead
            switch_decision = ...
            n_new = cache.n_new(new_top2)
            tracker.charge(switch_signal=switch_decision, n_new=n_new)
"""

from __future__ import annotations


class CreditsTracker:
    """Per-token, per-rollout credit budget tracker (scalar, no autograd)."""

    __slots__ = ("total", "used")

    def __init__(self, total_credits: float, *, used: float = 0.0) -> None:
        if total_credits <= 0:
            raise ValueError(f"total_credits must be > 0, got {total_credits}")
        self.total = float(total_credits)
        self.used = float(used)

    # ----- token-level lifecycle -----
    def reset_for_new_token(self) -> None:
        """Call once at each new generated token (before its layer-0 forward)."""
        self.used = 0.0

    # ----- per-layer ops -----
    @property
    def pressure(self) -> float:
        """Fraction of budget already consumed (entering this layer)."""
        return self.used / self.total

    @property
    def overflow(self) -> float:
        """How much we've overshot the budget so far this token (>= 0)."""
        u = self.used - self.total
        return u if u > 0 else 0.0

    def charge(self, switch_signal: float, n_new: int) -> None:
        """Record this layer's contribution to the running total.

        ``switch_signal`` is 0 or 1 (the layer's binary switch decision after STE).
        ``n_new`` is the number of cache misses caused by the load.
        """
        self.used += float(switch_signal) * int(n_new)

    # ----- factories -----
    @classmethod
    def from_config(cls, num_moe_layers: int, fraction: float = 0.7) -> "CreditsTracker":
        """Convenience: total = fraction × num_moe_layers (default 0.7)."""
        if num_moe_layers < 1:
            raise ValueError(f"num_moe_layers must be >= 1, got {num_moe_layers}")
        return cls(total_credits=float(num_moe_layers) * float(fraction))

    def reset_full(self, total_credits: float | None = None) -> None:
        if total_credits is not None:
            self.total = float(total_credits)
        self.used = 0.0

    def __repr__(self) -> str:
        return f"CreditsTracker(total={self.total:.3f}, used={self.used:.3f}, pressure={self.pressure:.3f})"


__all__ = ["CreditsTracker"]
