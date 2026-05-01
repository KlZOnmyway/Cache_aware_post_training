"""Straight-Through Estimator (STE) helpers for binarizing sigmoid switch outputs.

Forward: ``hard = (sigma > threshold).float()``. Backward: gradient flows through ``sigma`` as
identity (the standard STE).

We expose two flavors:

- ``ste_binary(sigma, threshold=0.5)`` — plain STE.
- ``ste_binary_with_noise(sigma, noise_std, threshold=0.5)`` — adds zero-mean Gaussian noise to
  ``sigma`` before thresholding (still STE-backward); useful as exploration during early
  training to keep the policy from collapsing into σ ∈ {0, 1}.

The "no Bernoulli sampling" property is preserved: the forward decision is deterministic given
``sigma``, only the backward pass is approximated.
"""

from __future__ import annotations

import torch


def ste_binary(sigma: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Straight-Through Estimator with hard threshold.

    forward:  out = (sigma > threshold).float()
    backward: out = sigma                  (identity)

    Args:
        sigma: tensor in [0, 1], typically sigmoid(logit).
        threshold: forward threshold; default 0.5.

    Returns:
        Tensor with the same shape as ``sigma``, values in {0., 1.} numerically,
        but backward gradient flows through ``sigma`` unchanged.
    """
    hard = (sigma > threshold).to(sigma.dtype)
    # standard STE: numerically equals `hard`, but backward = grad_out * 1 (through sigma)
    return hard.detach() + sigma - sigma.detach()


def ste_binary_with_noise(
    sigma: torch.Tensor,
    noise_std: float = 0.0,
    threshold: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """STE with optional additive Gaussian noise on ``sigma`` (forward only).

    Useful for early-training exploration to avoid switch_head saturating at 0 or 1
    before it has seen enough gradient.

    Args:
        sigma: tensor in [0, 1].
        noise_std: stddev of added noise; 0.0 disables.
        threshold: hard threshold for binarization.
        generator: optional torch.Generator for reproducibility.

    Returns:
        STE-binarized tensor.
    """
    if noise_std > 0.0:
        noise = torch.empty_like(sigma).normal_(generator=generator) * noise_std
        sigma = (sigma + noise).clamp(0.0, 1.0)
    return ste_binary(sigma, threshold=threshold)


def ste_binary_with_temperature(
    sigma: torch.Tensor,
    temperature: float = 1.0,
    threshold: float = 0.5,
) -> torch.Tensor:
    """STE binarization with optional sharpening / softening of σ.

    Higher temperature → softer (closer to 0.5 across the input range).
    Lower temperature → sharper (steeper sigmoid).

    Implemented by remapping σ ∈ [0,1] through a logit-temperature transform:
        logit' = (logit(σ) / temperature)
        σ'     = sigmoid(logit')
    Then we apply standard STE on σ'. Default ``temperature=1.0`` is a no-op.
    """
    if temperature == 1.0:  # no-op fast path
        return ste_binary(sigma, threshold=threshold)
    eps = 1e-6
    sigma_clamped = sigma.clamp(eps, 1.0 - eps)
    logit = torch.log(sigma_clamped) - torch.log1p(-sigma_clamped)
    sigma_t = torch.sigmoid(logit / float(temperature))
    return ste_binary(sigma_t, threshold=threshold)
