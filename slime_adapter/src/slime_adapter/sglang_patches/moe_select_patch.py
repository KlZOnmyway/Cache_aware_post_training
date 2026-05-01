"""SGLang ``TopK.forward_*`` patch — applies controller logic at rollout time.

Hook point
----------

SGLang's MoE pipeline calls a layer-level ``TopK`` op (in
``sglang.srt.layers.moe.topk``) whose ``forward_native`` / ``forward_cuda``
take ``hidden_states`` and ``router_logits`` and return a topk output object.
Both forwards have ``self.layer_id`` so we can locate the right ``SwitchHead``.

We monkey-patch both forwards. When a request has an active
``RequestControllerState`` in its contextvar AND a ``SwitchHead`` is
registered for ``self.layer_id``, the patched forward:

  1. Computes ``pressure`` from the request's CreditsTracker.
  2. Runs ``SwitchHead(hidden_states, pressure)`` → STE-thresholded switch.
  3. Computes ``new_top2 = router_logits.topk(2)``.
  4. ``used_top2 = switch ? new_top2 : current_top2`` (per-token).
  5. ``n_new = cache.n_new(used_top2)`` and updates cache + credits + records.
  6. Returns a ``StandardTopKOutput`` with ``used_top2`` and the renormalized
     softmax probs.

Otherwise it falls through to SGLang's original forward.

SwitchHead bank
---------------

The controller's per-layer ``SwitchHead`` modules live in a process-global
dict ``_SWITCH_HEAD_BANK`` (layer_idx → nn.Module), populated by the
slime → sglang weight-sync hook each time the trainer pushes new weights.

CUDA graph
----------

Because the forced indices change per-step, capturing CUDA graphs would freeze
out the override. v0.1 requires ``--disable-cuda-graph``; M3+ should make the
mask injection graph-friendly with fixed-size buffers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from .request_state import RequestControllerState, get_current_state


# =============================================================================
# Process-global SwitchHead bank
# =============================================================================

_SWITCH_HEAD_BANK: Dict[int, Any] = {}


def register_switch_head(layer_idx: int, module: Any) -> None:
    """Register / replace a SwitchHead for the given MoE layer."""
    _SWITCH_HEAD_BANK[int(layer_idx)] = module


def get_switch_head(layer_idx: int) -> Optional[Any]:
    return _SWITCH_HEAD_BANK.get(int(layer_idx))


def clear_switch_heads() -> None:
    _SWITCH_HEAD_BANK.clear()


# =============================================================================
# Patch state
# =============================================================================

_orig_forward_native: Optional[Callable] = None
_orig_forward_cuda: Optional[Callable] = None
_applied: bool = False


def apply_patches() -> None:
    """Wrap ``TopK.forward_native`` and ``TopK.forward_cuda``. Idempotent."""
    global _orig_forward_native, _orig_forward_cuda, _applied
    if _applied:
        return
    try:
        import sglang.srt.layers.moe.topk as _topk  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "sglang is not importable. apply_patches() must run inside the "
            "SGLang server process where sglang is installed."
        ) from e

    _orig_forward_native = _topk.TopK.forward_native
    _orig_forward_cuda = _topk.TopK.forward_cuda

    def patched_native(self, hidden_states, router_logits, *args, **kwargs):
        out = _maybe_apply_controller(self, hidden_states, router_logits)
        if out is not None:
            return out
        return _orig_forward_native(self, hidden_states, router_logits, *args, **kwargs)

    def patched_cuda(self, hidden_states, router_logits, *args, **kwargs):
        out = _maybe_apply_controller(self, hidden_states, router_logits)
        if out is not None:
            return out
        return _orig_forward_cuda(self, hidden_states, router_logits, *args, **kwargs)

    _topk.TopK.forward_native = patched_native
    _topk.TopK.forward_cuda = patched_cuda = patched_cuda
    # ^ assignment kept simple; some Python versions don't like the walrus here
    _topk.TopK.forward_cuda = patched_cuda
    _applied(True)


def _applied(set_to: Optional[bool] = None) -> bool:
    global _applied  # noqa: PLW0603
    if set_to is not None:
        # Module-level state via a list to avoid the global-statement dance.
        _APPLIED_FLAG[0] = bool(set_to)
    return _APPLIED_FLAG[0]


_APPLIED_FLAG = [False]


def restore_patches() -> None:
    if not _applied():
        return
    import sglang.srt.layers.moe.topk as _topk  # type: ignore
    if _orig_forward_native is not None:
        _topk.TopK.forward_native = _orig_forward_native
    if _orig_forward_cuda is not None:
        _topk.TopK.forward_cuda = _orig_forward_cuda
    _applied(False)


# ----------------------------------------------------------------------
# The actual controller-aware MoE select
# ----------------------------------------------------------------------

def _maybe_apply_controller(topk_module, hidden_states: torch.Tensor,
                            router_logits: torch.Tensor):
    """If a request state + switch_head are active, run controller logic.

    Returns either:
      - a StandardTopKOutput-shaped value (controller applied), or
      - ``None`` (no controller; caller should fall through).
    """
    state = get_current_state()
    if state is None:
        return None
    layer_idx = getattr(topk_module, "layer_id", None)
    if layer_idx is None:
        return None
    head = get_switch_head(layer_idx)
    if head is None:
        return None

    # 1. Entry-time pressure → broadcast to per-token tensor
    pressure_scalar = float(state.credits.pressure)
    pressure = torch.full(
        (hidden_states.shape[0],),
        fill_value=pressure_scalar,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # 2. SwitchHead forward + threshold (no autograd at rollout time)
    with torch.no_grad():
        sigma = torch.sigmoid(head(hidden_states, pressure))      # [N]
        switch = (sigma > 0.5).to(dtype=hidden_states.dtype)      # [N]

    # 3. Router argmax-2
    new_top2 = torch.topk(router_logits, k=2, dim=-1).indices     # [N, 2]

    # 4. Per-token used_top2 = switch ? new : carry-over
    cur = state.current_top2_per_layer.get(layer_idx)
    if cur is None:
        used_top2 = new_top2
    else:
        cur_t = torch.tensor(cur, device=new_top2.device, dtype=new_top2.dtype)
        cur_b = cur_t.unsqueeze(0).expand_as(new_top2)
        cond = switch.bool().unsqueeze(-1)
        used_top2 = torch.where(cond, new_top2, cur_b)

    # 5. Per-token book-keeping. Tell the state which layer we're in so
    # ``record_layer_step`` writes the correct layer_idx into the records.
    state.on_layer_entry(layer_idx)
    cache = state.caches[layer_idx]
    for i in range(used_top2.shape[0]):
        new_t = tuple(int(e) for e in new_top2_at_row(new_top2, i))
        used_t = tuple(int(e) for e in used_top2_at_row(used_top2, i))
        sw_i = int(switch[i].item()) if switch.numel() > 1 else int(switch.item())
        n_new = cache.n_new(used_t) if sw_i == 1 else 0
        state.record_layer_step(
            switch=sw_i,
            used_top2=used_t,
            new_top2=new_t,
            n_new=n_new,
            pressure_in=pressure_scalar,
        )
        # update tracking
        state.current_top2_per_layer[topk_module.layer_id] = list(used_t)
        cache.push(used_t)
        state.credits.charge(switch_signal=sw_i, n_new=n_new)

    # 6. Build SGLang StandardTopKOutput with our chosen indices + renormalized probs
    used_probs = _gather_probs(router_logits, used_top2)
    return _make_standard_output(used_probs, used_top2, router_logits)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def new_top2_at_row(t: torch.Tensor, i: int):
    return t[i].tolist()


def _gather_probs(router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(router_logits, dim=-1)
    g = probs.gather(dim=-1, index=indices)
    s = g.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return g / s


def used_top2_at_row(t: torch.Tensor, i: int):
    return t[i].tolist()


def _make_topk_output(used_top2: torch.Tensor, used_probs: torch.Tensor, router_logits: torch.Tensor):
    """Construct an SGLang ``StandardTopKOutput`` (sglang ≥ 0.5: needs router_logits)."""
    from sglang.srt.layers.moe.topk import StandardTopKOutput  # type: ignore
    return StandardTopKOutput(topk_weights=used_probs, topk_ids=used_top2, router_logits=router_logits)


def _make_standard_output(used_probs: torch.Tensor, used_top2: torch.Tensor, router_logits: torch.Tensor):
    return _make_topk_output(used_top2=used_top2, used_probs=used_probs, router_logits=router_logits)


# ----------------------------------------------------------------------
# Final, clean apply_patches (the version above had transcription noise; this
# one supersedes it).
# ----------------------------------------------------------------------

def apply_patches() -> None:  # noqa: F811
    global _orig_forward_native, _orig_forward_cuda
    if _applied_flag():
        return
    try:
        import sglang.srt.layers.moe.topk as _topk  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "sglang is not importable. apply_patches() must run inside the "
            "SGLang server's Python process."
        ) from e

    _orig_forward_native = _topk.TopK.forward_native
    _orig_forward_cuda = _topk.TopK.forward_cuda

    def patched_native(self, hidden_states, router_logits, *args, **kwargs):
        out = _maybe_apply_controller(self, hidden_states, router_logits)
        return out if out is not None else _orig_forward_native(self, hidden_states, router_logits, *args, **kwargs)

    def patched_cuda(self, hidden_states, router_logits, *args, **kwargs):
        out = _maybe_apply_controller(self, hidden_states, router_logits)
        return out if out is not None else _orig_forward_cuda(self, hidden_states, router_logits, *args, **kwargs)

    _topk.TopK.forward_native = patched_native
    _topk.TopK.forward_cuda = patched_cuda
    _set_applied_flag(True)


def _applied_flag() -> bool:
    return _APPLIED_FLAG[0]


def _set_applied_flag(v: bool) -> None:
    _APPLIED_FLAG[0] = bool(v)


_APPLIED_FLAG = [False]


def restore_patches() -> None:
    """Undo apply_patches()."""
    global _orig_forward_native, _orig_forward_cuda
    if not _applied_flag():
        return
    import sglang.srt.layers.moe.topk as _topk  # type: ignore
    if _orig_forward_native is not None:
        _topk.TopK.forward_native = _orig_forward_native
    if _orig_forward_cuda is not None:
        _topk.TopK.forward_cuda = _orig_forward_cuda
    _orig_forward_native = None
    _orig_forward_cuda = None
    _set_applied_flag(False)


# ----------------------------------------------------------------------
# misc
# ----------------------------------------------------------------------

def new_top2_at_row(t: torch.Tensor, i: int):
    return t[i].tolist()


def used_top2_at_row(t: torch.Tensor, i: int):
    return t[i].tolist()


__all__ = [
    "apply_patches",
    "restore_patches",
    "register_switch_head",
    "get_switch_head",
    "clear_switch_heads",
]
