"""DeepSets encoder for expert sets — ported from rl_moe's controller.

Used by ``SwitchHead`` to encode two sets at each (b, t) position:

  * ``cache_set_repr`` — experts currently in the rolling cache (state ω_t).
    Variable-size; passed as a ``[B, num_experts]`` bool mask.
  * ``top_k_set_repr`` — router's top-K candidates (what would have to be
    loaded if the layer fires). Fixed-size K=2; passed as ``[B, k] long``.

DeepSets gives us permutation invariance for both — we pool with mean over
the set elements after a per-element MLP φ. The set-rep is concatenated to
``(hidden_state, pressure)`` before the gating linear, giving the gate full
visibility into "what would I have to pay if I switched right now".

Reference: rl_moe ``transformers_patches/models/gpt_oss/modeling_gpt_oss.py``
lines 773–797 (DeepSets φ over expert embedding, Q-head input).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ExpertSetEncoder(nn.Module):
    """Per-layer DeepSets encoder for sets of expert ids.

    forward(...) supports two input modes:
      * ``mask=[..., num_experts]``  bool / 0-1   → variable-size set
      * ``indices=[..., k]`` long                  → fixed-size set; missing
                                                     positions can be sentinel
                                                     ``-1`` and are masked out.

    Output shape is ``[..., set_dim]``.
    """

    def __init__(
        self,
        num_experts: int,
        embed_dim: int = 64,
        set_dim: Optional[int] = None,
        pool: str = "mean",
    ):
        super().__init__()
        if pool not in {"mean", "sum"}:
            raise ValueError(f"pool must be 'mean' or 'sum', got {pool!r}")
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        self.num_experts = int(num_experts)
        self.embed_dim = int(embed_dim)
        self.set_dim = int(set_dim) if set_dim is not None else int(embed_dim)
        self.pool_mode = str(pool)

        self.expert_embedding = nn.Embedding(self.num_experts, self.embed_dim)
        # φ: per-element transformation (DeepSets)
        self.phi = nn.Sequential(
            nn.Linear(self.embed_dim, self.set_dim),
            nn.GELU(),
            nn.Linear(self.set_dim, self.set_dim),
        )
        # Small init so initial set_repr ≈ 0 — keeps the gate's cold-start
        # decision close to the (h, pressure)-only behavior of v4.
        nn.init.normal_(self.expert_embedding.weight, std=0.02)
        for m in self.phi.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ----------------------------------------------------------------- #
    # Encoding paths
    # ----------------------------------------------------------------- #

    def encode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        """``idx: [..., k]`` long → ``[..., set_dim]`` set representation.

        ``-1`` entries are treated as "missing" and contribute nothing.
        """
        valid = idx >= 0                                       # [..., k] bool
        safe = idx.clamp(min=0, max=self.expert_embedding.num_embeddings - 1)
        emb = self.expert_embedding(safe)                      # [..., k, embed_dim]
        emb = emb * valid.unsqueeze(-1).to(emb.dtype)          # zero invalid
        phi_emb = self.phi(emb)                                # [..., k, set_dim]
        if self.pool_mode == "sum":
            return phi_emb.sum(dim=-2)
        denom = valid.sum(dim=-1, keepdim=True).clamp_min(1).to(phi_emb.dtype)
        return phi_emb.sum(dim=-2) / denom

    def encode_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """``mask: [..., num_experts]`` bool/0-1 → ``[..., set_dim]`` set rep.

        Computes ``Σ_e mask[..., e] · φ(E[e])`` then mean-pools by the count
        in ``mask``. Permutation-invariant by construction.
        """
        all_phi = self._all_phi()                              # [num_experts, set_dim]
        m = mask.to(dtype=all_phi.dtype)                       # [..., num_experts]
        summed = torch.einsum("...e,eh->...h", m, all_phi)     # [..., set_dim]
        if self.pool_mode == "sum":
            return summed
        n = m.sum(dim=-1, keepdim=True).clamp_min(1.0)         # [..., 1]
        return summed / n

    # ----- helpers -----

    def _all_phi(self) -> torch.Tensor:
        """Cached per-call φ(E[e]) for all e — small, recomputed each forward
        so embedding gradients flow normally."""
        idx = torch.arange(
            self.num_experts,
            device=self.expert_embedding.weight.device,
        )
        return self.phi(self.expert_embedding(idx))            # [E, set_dim]


__all__ = ["ExpertSetEncoder"]
