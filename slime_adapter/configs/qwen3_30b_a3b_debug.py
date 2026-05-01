"""Tiny debug config — for sanity check the controller core without full training.

Useful for unit-testing the SwitchHead / cache / budget tracker without booting
slime / Megatron / SGLang.
"""

from __future__ import annotations

import torch

from slime_adapter import SwitchHead, LayerCache, CreditsTracker, ste_binary


def main() -> None:
    H, T, B, num_layers = 64, 8, 2, 4

    # 1. SwitchHead per layer
    heads = [SwitchHead(hidden_size=H, init_bias=-2.0) for _ in range(num_layers)]

    # 2. CreditsTracker (single token)
    credits = CreditsTracker.from_config(num_moe_layers=num_layers, fraction=0.7)
    credits.reset_for_new_token()

    # 3. Cache per layer
    caches = [LayerCache(window=16, cap=30) for _ in range(num_layers)]

    # 4. Walk through layers
    hidden = torch.randn(B, T, H)
    for l, head in enumerate(heads):
        pressure = torch.full((B, T), credits.pressure)
        sigma = torch.sigmoid(head(hidden, pressure))
        switch = ste_binary(sigma)
        n_new = torch.full((B, T), 1, dtype=torch.long)  # pretend 1 new expert per step
        credits.charge(switch_signal=float(switch.mean()), n_new=1)
        print(f"layer {l}: σ_mean={sigma.mean():.3f}, switch_mean={switch.mean():.3f}, "
              f"pressure={credits.pressure:.3f}, used={credits.used:.3f}/{credits.total}")
        caches[l].push((0, 1))


if __name__ == "__main__":
    main()
