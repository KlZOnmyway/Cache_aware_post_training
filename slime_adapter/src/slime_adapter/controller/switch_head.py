"""Per-layer sigmoid gate that decides whether to refresh the cache for this token.

Inputs at each (token, layer):
  - hidden state  (pre-MoE)        [..., hidden_size]
  - pressure      (forward-only)   [...]   = credits_used_before_l / total_credits

Output: a scalar logit per (..., ) location. Caller does sigmoid + STE binarization.

Design notes
------------

- One ``SwitchHead`` instance per MoE layer. Total trainable params ≈ L × (H + 2),
  which is well under 1 MB for typical 24-layer / 4096-hidden MoEs.

- ``pressure`` is given as a *forward-only feature* (caller ``.detach()``-es it).
  This lets the model condition its switch decision on remaining budget without
  introducing a backward path through the credits accumulator (which would
  re-couple early layers to late layers, undoing the per-layer credit assignment).

- The Linear's bias doubles as a learnable per-layer prior: layers that should
  switch often will train it high, layers that should usually conserve budget
  will train it low. We initialize at ``init_bias = -2.0`` so the initial switch
  probability is sigmoid(-2) ≈ 0.12 (sparse → don't burn the budget on cold start).

- ``zero_init_weight=True`` makes the head a near-constant predictor at init,
  so the initial behaviour is fully determined by ``init_bias``. This avoids
  any random "lucky early gradient" effects.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SwitchHead(nn.Module):
    """Single-output per-MoE-layer gate.

    Args:
        hidden_size: dim of the pre-MoE hidden state.
        init_bias: bias initialization; sigmoid(init_bias) is the starting switch prob.
        use_pressure_input: if True (default), concat a 1-d ``pressure`` scalar to the
            input. Set False to ablate the budget-aware feature.
        zero_init_weight: zero the weight matrix at init so output ≈ init_bias initially.
    """

    def __init__(
        self,
        hidden_size: int,
        init_bias: float = -2.0,
        use_pressure_input: bool = True,
        zero_init_weight: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.use_pressure_input = bool(use_pressure_input)
        in_dim = self.hidden_size + (1 if self.use_pressure_input else 0)
        self.linear = nn.Linear(in_dim, 1)
        if zero_init_weight:
            nn.init.zeros_(self.linear.weight)
        else:
            nn.init.normal_(self.linear.weight, std=1.0 / math.sqrt(in_dim))
        nn.init.constant_(self.linear.bias, float(init_bias))

    def forward(
        self,
        hidden: torch.Tensor,
        pressure_scalar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns logits with shape == hidden.shape[:-1]."""
        if self.use_pressure_input:
            if pressure_scalar is None:
                raise ValueError(
                    "SwitchHead built with use_pressure_input=True; pass `pressure_scalar=...`."
                )
            # Broadcast pressure to (...,) matching hidden's leading dims, then to (...,1).
            p = pressure_scalar.to(dtype=hidden.dtype)
            if p.dim() == 0:
                # scalar pressure → broadcast to (...,1)
                p = p.expand(*hidden.shape[:-1])
            elif p.shape != hidden.shape[:-1]:
                # caller may have passed [B] for [B, T, H] — broadcast over T
                try:
                    p = p.expand(*hidden.shape[:-1])
                except RuntimeError as e:
                    raise ValueError(
                        f"pressure_scalar shape {tuple(p.shape)} cannot be broadcast to "
                        f"hidden's leading dims {tuple(hidden.shape[:-1])}"
                    ) from e
            x = torch.cat([hidden, p.unsqueeze(-1)], dim=-1)
        else:
            x = hidden
        return self.linear(x).squeeze(-1)

    # ----- conveniences -----
    @property
    def init_switch_prob(self) -> float:
        return float(torch.sigmoid(self.linear.bias.detach()).mean().item())
