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
        *,
        cache_set_dim: int = 0,
        topk_set_dim: int = 0,
    ) -> None:
        """SwitchHead with optional DeepSets context.

        Args:
            hidden_size: dim of pre-MoE hidden state.
            init_bias: cold-start σ ≈ sigmoid(init_bias).
            use_pressure_input: append 1-d pressure scalar.
            zero_init_weight: zero the weight matrix so cold-start σ depends only on bias.
            cache_set_dim: dim of the cache set representation (output of
                ``ExpertSetEncoder``). 0 = disabled (legacy behavior).
            topk_set_dim: dim of the top-K set representation. 0 = disabled.
        """
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.use_pressure_input = bool(use_pressure_input)
        self.cache_set_dim = int(cache_set_dim)
        self.topk_set_dim = int(topk_set_dim)
        in_dim = (
            self.hidden_size
            + (1 if self.use_pressure_input else 0)
            + self.cache_set_dim_safe()
            + self.topk_set_dim_safe()
        )
        self.linear = nn.Linear(in_dim, 1)
        if zero_init_weight:
            nn.init.zeros_(self.linear.weight)
        else:
            nn.init.normal_(self.linear.weight, std=1.0 / max(1, in_dim) ** 0.5)
        nn.init.constant_(self.linear.bias, float(init_bias))

    def cache_set_dim_safe(self) -> int:
        return max(0, int(self.cache_set_dim))

    def topk_set_dim_safe(self) -> int:
        return max(0, int(self.topk_set_dim))

    def forward(
        self,
        hidden: torch.Tensor,
        pressure_scalar: torch.Tensor | None = None,
        cache_set_repr: torch.Tensor | None = None,
        top_k_set_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns logits with shape == hidden.shape[:-1].

        Optional inputs (set ``cache_set_dim``/``topk_set_dim`` > 0 to enable):
            cache_set_repr  — DeepSets pool of currently-cached experts; shape
                              ``[..., cache_set_dim]``, broadcastable over leading dims.
            top_k_set_repr  — DeepSets pool of router top-k indices; shape
                              ``[..., topk_set_dim]``.
        """
        feats = [hidden]

        if self.use_pressure_input:
            if pressure_scalar is None:
                raise ValueError(
                    "SwitchHead built with use_pressure_input=True; pass `pressure_scalar=...`."
                )
            p = pressure_scalar.to(hidden.dtype)
            if p.dim() == 0:
                p = p.expand(*hidden.shape[:-1])
            elif p.shape != hidden.shape[:-1]:
                try:
                    p = p.expand(*hidden.shape[:-1])
                except RuntimeError as e:
                    raise ValueError(
                        f"pressure_scalar shape {tuple(p.shape)} cannot be broadcast to "
                        f"hidden's leading dims {tuple(hidden.shape[:-1])}"
                    ) from e
            feats.append(p.unsqueeze(-1))

        if self.cache_set_dim > 0:
            if cache_set_repr is None:
                raise ValueError(
                    f"SwitchHead built with cache_set_dim={self.cache_set_dim}; "
                    "pass cache_set_repr=..."
                )
            feats.append(_broadcast_to_leading(cache_set_repr, hidden))

        if self.topk_set_dim > 0:
            if top_k_set_repr is None:
                raise ValueError(
                    f"SwitchHead built with topk_set_dim={self.topk_set_dim}; "
                    "pass top_k_set_repr=..."
                )
            feats.append(_broadcast_to_leading(top_k_set_repr, hidden))

        x = feats[0] if len(feats) == 1 else torch.cat(feats, dim=-1)
        return self.linear(x).squeeze(-1)


def _broadcast_to_leading(rep: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Broadcast ``rep`` to match ``hidden``'s leading dims.

    If ``rep.shape[-1] == D`` and ``rep.shape[:-1] != hidden.shape[:-1]``,
    we add a singleton dim before the last and expand. Most common case:
    ``hidden=[B, T, H]``, ``rep=[B, D]`` → returns ``[B, T, D]``.
    """
    if rep.shape[:-1] == hidden.shape[:-1]:
        return rep.to(dtype=hidden.dtype)
    cur = rep.to(dtype=hidden.dtype)
    while cur.dim() < hidden.dim():
        cur = cur.unsqueeze(-2)
    target = list(hidden.shape[:-1]) + [cur.shape[-1]]
    return cur.expand(*target)


def _switch_head_init_switch_prob(self) -> float:
    """Cold-start σ from the bias alone (zero-init weights → output ≈ bias)."""
    return float(torch.sigmoid(self.linear.bias.detach()).mean().item())


# Re-attach the convenience property to SwitchHead (defined out of body to keep
# the function block above readable).
SwitchHead.init_switch_prob = property(_switch_head_init_switch_prob)
