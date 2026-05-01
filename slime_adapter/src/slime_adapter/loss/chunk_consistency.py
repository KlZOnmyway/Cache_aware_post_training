"""Chunk-wise routing-consistency loss.

For each (layer, chunk-of-K-tokens), compute the within-chunk KL of the
softmax-router-distribution to the chunk-mean distribution. This regularizer
encourages routing decisions to be temporally smooth, which:

  - makes the cache more useful (within-chunk reuse);
  - lets ``switch=0`` be the right call for many tokens (saving budget);
  - generally aligns student routing with itself across nearby tokens.

Note: we do **not** compare student routing to teacher routing (would conflict
with LoRA-specialized experts).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_chunk_consistency(
    *,
    router_logits: torch.Tensor | None = None,
    chunk_size: int = 8,
    enabled: bool = False,
    eps: float = 1e-12,
    device=None,
) -> torch.Tensor:
    """Public hook the wrapped MoE-layer forward calls to obtain the chunk
    routing-consistency loss term.

    Stage-1 default: returns a 0-d tensor (placeholder). When ``enabled=True``
    AND ``router_logits`` is provided, forwards to
    :func:`chunk_routing_consistency_loss` below.

    Rationale for the gate
        Extracting ``router_logits`` is adapter-specific. We ship the wiring
        with a no-op default so the loss pipeline never depends on per-adapter
        plumbing. Flip ``enabled=True`` (and pass real logits) once an adapter
        routes them through.

    Args:
        router_logits: ``[B, T, L, E]`` or ``[B, T, E]`` (single layer).
        chunk_size: chunk width K (tokens per chunk).
        enabled: when False, always returns zero (default).
        device: optional device hint when ``router_logits`` is None.
        eps: log-domain floor.

    Returns:
        Scalar ``torch.Tensor`` (0-d). Plug into
        ``state.chunk_consistency_loss = compute_chunk_consistency(...)``.
    """
    if not enabled or router_logits is None:
        target = device or (router_logits.device if router_logits is not None else None)
        return (torch.zeros((), device=target)
                if target is not None else torch.tensor(0.0))
    if router_logits.dim() == 3:
        router_logits = router_logits.unsqueeze(2)
    return chunk_routing_consistency_loss(
        router_logits, chunk_size=chunk_size, eps=eps,
    )



def chunk_routing_consistency_loss(
    router_logits: torch.Tensor,
    chunk_size: int = 8,
    *,
    per_layer_dim: int = 2,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mean within-chunk forward-KL between each token's routing dist and the chunk's mean.

    Args:
        router_logits: shape ``[B, T, L, E]`` — per-(batch, token, layer) router logits.
            If your training pipe yields a different layout, reshape first.
        chunk_size: K — number of consecutive tokens per chunk.
        per_layer_dim: which axis is the "layer" axis (default 2). The reduction
            is over (B, n_chunks, K, layer); we average across all of them.
        eps: numerical floor for log-domain math.

    Returns:
        Scalar tensor (mean over all (b, chunk, layer, t-in-chunk) of KL terms).
    """
    if router_logits.dim() != 4:
        raise ValueError(
            f"router_logits must be [B, T, L, E]; got shape {tuple(router_logits.shape)}"
        )
    B, T, L, E = router_logits.shape
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if T == 0:
        return torch.zeros((), device=router_logits.device, dtype=router_logits.dtype)
    # If sequence is shorter than chunk_size, just use one big chunk
    if T < chunk_size:
        chunk_size = T

    n_chunks = T // chunk_size
    if n_chunks == 0:
        n_chunks = 1
        chunk_size = T

    truncated = router_logits[:, : n_chunks * chunk_size]                 # [B, n*K, L, E]
    chunked = truncated.reshape(B, n_chunks, chunk_size, L, E)            # [B, n, K, L, E]

    p = torch.softmax(chunked, dim=-1)                                    # softmax over experts
    p_mean = p.mean(dim=2, keepdim=True)                                  # [B, n, 1, L, E]
    log_p = torch.log(p.clamp_min(eps))
    log_pm = torch.log(p_mean.clamp_min(eps)).detach()                    # stop-grad on chunk mean
    kl = (p * (log_p - log_pm)).sum(dim=-1)                               # [B, n, K, L]

    return kl.mean()
