"""Per-MoE-layer forward wrapper that injects SwitchHead + cache + budget.

Lifecycle
---------

1. After the model is built, call

       install_controller_into_layers(model, adapter, args)

   which walks every MoE layer (using ``adapter.iter_moe_layers``) and:

     - attaches a fresh ``SwitchHead`` (per-layer trainable);
     - attaches a ``LayerCache`` (rolling window of expert ids);
     - attaches a ``ControllerReplay`` (per-layer record buffer);
     - replaces ``layer.forward`` with ``_layer_forward_with_controller``.

2. Each model forward pass starts by calling

       begin_controller_forward(model, hidden_states_proxy)

   from a top-level pre-forward hook. This allocates a fresh
   ``TokenBudgetState`` and propagates it onto every wrapped layer's
   ``_budget_state`` attribute, where the wrapped forward picks it up.

3. After the forward, the trainer calls

       summary = end_controller_forward(model)

   which returns a ``BudgetReadout`` (per-layer costs + token-level used) for
   the loss patch to consume.

Cross-cutting design
--------------------

Nothing in this module imports a specific model architecture — all model-
specific logic goes through the ``adapter`` argument. To support a new model:
just write a new ``MoEModelAdapter`` subclass; this file is reused as-is.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from slime_adapter.controller.cache_state import LayerCache
from slime_adapter.controller.ste import ste_binary
from slime_adapter.controller.switch_head import SwitchHead
from slime_adapter.modeling._base import MoELayerHandle, MoEModelAdapter

from .budget import LayerBudgetTracker, TokenBudgetState
from .compute_topk_patch import ControllerReplay


# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------

def install_controller_into_layers(
    model: nn.Module,
    adapter: MoEModelAdapter,
    args,
) -> List[MoELayerHandle]:
    """Walk every MoE layer in ``model`` and install the controller wrappers.

    Called once at startup, after model construction.

    Reads from ``args``:
      gate_init_bias (default -2.0), cache_window (16), cache_cap (30),
      use_pressure_input (True), budget_fraction (0.7).

    Stores on ``model``:
      _slime_adapter_budget: LayerBudgetTracker
      _slime_adapter_handles: List[MoELayerHandle]
    """
    init_bias = float(getattr(args, "gate_init_bias", -2.0))
    use_pressure = bool(getattr(args, "use_pressure_input", True))
    cw = int(getattr(args, "cache_window", LayerCache.DEFAULT_WINDOW))
    cc = int(getattr(args, "cache_cap", LayerCache.DEFAULT_CAP))
    fraction = float(getattr(args, "budget_fraction", 0.7))

    handles = list(adapter.iter_moe_layers(model))
    if not handles:
        raise RuntimeError(
            f"adapter {adapter.name!r} found no MoE layers in this model. "
            f"Verify ``adapter.iter_moe_layers`` and the model wrapping."
        )

    model._slime_adapter_budget = LayerBudgetTracker(total_credits=fraction * len(handles))
    model._slime_adapter_handles = handles
    model._slime_adapter_adapter = adapter

    for handle in handles:
        wrap_moe_layer(
            adapter=adapter,
            handle=handle,
            init_bias=float(getattr(args, "gate_init_bias", init_bias)),
            use_pressure_input=use_pressure,
            cache_window=cw,
            cache_cap=cc,
        )
    return handles


def wrap_moe_layer(
    *,
    adapter: MoEModelAdapter,
    handle: MoELayerHandle,
    init_bias: float = -2.0,
    use_pressure_input: bool = True,
    cache_window: int = 16,
    cache_cap: int = 30,
) -> None:
    """Wrap a single MoE layer's forward (idempotent)."""
    layer = handle.module
    if getattr(layer, "_slime_adapter_wrapped", False):
        return

    sh = SwitchHead(
        hidden_size=handle.hidden_size,
        init_bias=init_bias,
        use_pressure_input=use_pressure_input,
    )
    # Place the SwitchHead on the same device/dtype as the host layer.
    host_param = next((p for p in layer.parameters()), None)
    if host_param is not None:
        sh = sh.to(device=host_param.device, dtype=host_param.dtype if host_param.dtype.is_floating_point else None)
    adapter.install_switch_head(handle, sh, attr="switch_head")

    layer.cache_state = LayerCache(window=cache_window, cap=cache_cap)
    layer.controller_replay = ControllerReplay()
    layer.current_top2 = None                  # set on first forward
    layer._budget_state = None                 # set per-forward by begin_controller_forward
    layer._slime_adapter_ref = adapter
    layer._slime_adapter_original_forward = layer.forward

    def _forward(hidden_states, *fwd_args, **fwd_kwargs):
        return _layer_forward_with_controller(layer, hidden_states, *fwd_args, **fwd_kwargs)

    _forward.__name__ = f"{type(layer).__name__}_with_controller"
    layer.forward = _forward
    layer._slime_adapter_wrapped = True


def call_original_forward(layer: nn.Module, *args, **kwargs):
    """Invoke the layer's pre-wrap forward (bypassing our controller).

    Adapters use this from inside ``forward_with_forced_top_indices`` to call
    into the model's native MoE compute without re-entering our wrapper
    (which would cause infinite recursion).
    """
    fn = getattr(layer, "_slime_adapter_original_forward", None)
    if fn is None:
        raise RuntimeError(
            "Layer was not wrapped via wrap_moe_layer; "
            "call install_controller_into_layers(model, adapter, args) first."
        )
    return fn(*args, **kwargs)


def begin_controller_forward(model: nn.Module, hidden_states_proxy: torch.Tensor) -> TokenBudgetState:
    """Initialize per-forward budget state. Call once before each forward.

    Use as a forward pre-hook on the top-level model module, e.g.::

        model.register_forward_pre_hook(lambda m, inp: begin_controller_forward(m, inp[0]))
    """
    holder: LayerBudgetTracker = model._slime_adapter_budget
    state = holder.begin(hidden_states_proxy)
    for layer in _iter_wrapped_layers(model):
        layer._budget_state = state
    return state


def end_controller_forward(model: nn.Module):
    """Drain the per-forward budget state and clear it from layers.

    Returns the final TokenBudgetState (containing per-layer costs &
    accumulated used). The loss patch reads this to compute L_budget /
    L_barrier.
    """
    state = None
    for layer in _iter_wrapped_layers(model):
        if state is None and layer._budget_state is not None:
            state = layer._budget_state
        layer._budget_state = None
    return state


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------

def _layer_forward_with_controller(layer, hidden_states: torch.Tensor, *args, **kwargs):
    """Replacement forward installed on each MoE layer."""
    state: Optional[TokenBudgetState] = getattr(layer, "_budget_state", None)
    adapter: MoEModelAdapter = layer._slime_adapter_ref

    if state is None:
        # No active controller forward (e.g. eval pass) — fall back.
        return layer._slime_adapter_original_forward(hidden_states, *args, **kwargs)

    # 1. Pressure (forward-only, detached) → SwitchHead → STE
    pressure = state.pressure_at_entry()                      # [B, T] no-grad
    sigma = torch.sigmoid(layer.switch_head(hidden_states, pressure))  # [B, T]
    switch = ste_binary(sigma)                                # forward 0/1, backward identity

    # 2. Router argmax-2 (no grad)
    new_top2 = adapter.compute_router_top_k(layer, hidden_states, k=2)  # [B, T, 2]

    # 3. n_new vs cache (no grad)
    n_new = _compute_n_new_batched(new_top2, layer.cache_state)  # [B, T] long

    # 4. used_top2 = switch ? new_top2 : carry-over
    used_top2 = _select_used_top2(layer.current_top2, new_top2, switch)  # [B, T, 2]

    # 5. Charge layer cost (autograd attaches via STE-bound switch)
    state.charge_layer(switch, n_new)

    # 6. Update cache (per-layer global state — see budget.py for batched note)
    if used_top2.shape[1] > 0:
        layer.cache_state.push(tuple(used_top2[0, -1].tolist()))
    layer.current_top2 = used_top2[:, -1, :].detach()

    # 7. Record for replay / debugging
    layer.controller_replay.record(
        switch=switch.detach(),
        used_top2=used_top2.detach(),
        new_top2=new_top2.detach(),
        n_new=n_new.detach(),
        pressure_in=pressure.detach(),
    )

    # 8. Run actual MoE with forced indices
    return adapter.forward_with_forced_top_indices(
        moe_module=layer,
        hidden_states=hidden_states,
        forced_indices=used_top2,
    )


def _select_used_top2(
    current_top2: Optional[torch.Tensor],
    new_top2: torch.Tensor,
    switch: torch.Tensor,
) -> torch.Tensor:
    """``[B, T, k]`` = switch ? new : carry-over.

    Cold-start: when current_top2 is None, force used = new (n_new will then
    correctly bill the load).
    """
    if current_top2 is None:
        return new_top2
    if current_top2.dim() == new_top2.dim() - 1:
        # [B, k] → [B, T, k]
        current_top2 = current_top2.unsqueeze(1).expand_as(new_top2)
    cond = switch.unsqueeze(-1) > 0.5
    return torch.where(cond, new_top2, current_top2)


def _compute_n_new_batched(new_top2: torch.Tensor, cache: LayerCache) -> torch.Tensor:
    """``[B, T]`` long tensor: count of experts in ``new_top2`` not in cache.union.

    Cache is a single per-layer Python state object; we treat its union as the
    "fast memory" for all batch positions. This is an approximation for batched
    training; rollout time uses per-request state instead (see sglang_patches).
    """
    if cache.size == 0:
        return torch.full(new_top2.shape[:2], new_top2.shape[-1],
                          dtype=torch.long, device=new_top2.device)
    union_ids = torch.tensor(list(cache.union), dtype=new_top2.dtype, device=new_top2.device)
    # broadcast: new_top2 [B, T, k] vs union [U] -> [B, T, k, U]
    in_cache = (new_top2.unsqueeze(-1) == union_ids.view(1, 1, 1, -1)).any(dim=-1)
    return (~in_cache).long().sum(dim=-1)  # [B, T]


def _iter_wrapped_layers(model: nn.Module) -> Iterable[nn.Module]:
    for m in model.modules():
        if getattr(m, "_slime_adapter_wrapped", False):
            yield m


# Keep ``_iter_wrapped_layers`` accessible under the name used by other files
_iter_wrapped_layers = _iter_wrapped_layers
