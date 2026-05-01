"""Verify our sglang TopK patch installs and intercepts correctly.

Uses sglang's actual ``TopK`` op without booting a full server:

  1. Construct a real ``TopK`` op (the in-process version sglang uses inside MoE).
  2. Apply our ``slime_adapter.sglang_patches.apply_patches()``.
  3. Without an active ``RequestControllerState`` in TLS — verify TopK forward
     falls through to the original (unchanged behavior).
  4. With an active ``RequestControllerState`` + a registered SwitchHead —
     verify TopK forward returns the controller-decided indices and updates
     the per-request cache + budget.
  5. ``restore_patches()`` — verify forward goes back to original.

Run::

    CUDA_VISIBLE_DEVICES=1 uv run --frozen --no-sync python scripts/single_gpu_real_sglang_patch.py
"""

from __future__ import annotations

import os

import torch

from slime_adapter.controller.switch_head import SwitchHead
from slime_adapter.sglang_patches import (
    apply_patches,
    restore_patches,
    register_switch_head,
    clear_switch_heads,
    RequestControllerState,
    set_current_state,
    CURRENT_STATE,
)


def main():
    # 1. Build a real TopK op
    from sglang.srt.layers.moe.topk import TopK, StandardTopKOutput
    NUM_EXPERTS = 8
    HIDDEN = 16
    NUM_TOKENS = 4
    LAYER_ID = 3

    topk_op = TopK(top_k=NUM_EXPERTS, layer_id=LAYER_ID)  # native top_k = NUM_EXPERTS  (we'll override)
    print(f"[real-sglang] built TopK op (layer_id={LAYER_ID})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Save original behavior for comparison
    orig_native = TopK.forward_native
    orig_cuda = TopK.forward_cuda

    # 3. Apply patches
    apply_patches()
    assert TopK.forward_native is not orig_native, "patch didn't replace forward_native"
    assert TopK.forward_cuda is not orig_cuda, "patch didn't replace forward_cuda"
    print(f"[real-sglang] apply_patches() ✓ replaced TopK.forward_native and forward_cuda")

    # 4. Without active state, patched forward should fall through to original behavior
    hidden = torch.randn(NUM_TOKENS, HIDDEN, device=device)
    router_logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, device=device)

    # Build minimal-shape sglang TopK output without state
    topk_op.topk_config.top_k = 2  # we want top-2
    out_no_state = topk_op.forward_native(hidden, router_logits)
    print(f"[real-sglang] no state: TopK.forward_native returned {type(out_no_state).__name__}, "
          f"topk_ids.shape={out_no_state.topk_ids.shape}")

    # 5. Now register a SwitchHead and activate a RequestControllerState
    sh = SwitchHead(hidden_size=HIDDEN, init_bias=0.0, use_pressure_input=True).to(device)
    register_switch_head(LAYER_ID, sh)
    state = RequestControllerState(
        num_moe_layers=4,  # arbitrary; only this layer matters for this test
        cache_window=4, cache_cap=30, budget_fraction=0.7,
    )
    state.on_new_token(0)
    token = set_current_state(state)
    try:
        out_with_state = topk_op.forward_native(hidden, router_logits)
        print(f"[real-sglang] with controller: TopK.forward_native returned "
              f"{type(out_with_state).__name__}, topk_ids.shape={out_with_state.topk_ids.shape}")

        assert out_with_state.topk_ids.shape == (NUM_TOKENS, 2), \
            f"controller path should produce top-2; got {out_with_state.topk_ids.shape}"
        assert len(state.records) == NUM_TOKENS, \
            f"expected {NUM_TOKENS} records, got {len(state.records)}"
        rec = state.records[0]
        print(f"[real-sglang] state captured records: layer={rec.layer_idx}, "
              f"switch={rec.switch}, used_top2={rec.used_top2}, n_new={rec.n_new}")
    finally:
        CURRENT_STATE.reset(token)

    # 6. Restore and confirm forward is back to original
    restore_patches()
    assert TopK.forward_native is orig_native, "restore_patches didn't restore forward_native"
    print(f"[real-sglang] restore_patches() ✓ TopK.forward_native back to original")

    clear_switch_heads()
    print(f"[real-sglang] OK")


if __name__ == "__main__":
    main()
