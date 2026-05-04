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

from slime_adapter.controller.cache_state import LayerCache, BatchedLayerCache
from slime_adapter.controller.expert_set_encoder import ExpertSetEncoder
from slime_adapter.controller.ste import ste_binary
from slime_adapter.controller.switch_head import SwitchHead
from slime_adapter.loss.chunk_consistency import compute_chunk_consistency
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
      use_pressure_input (True), budget_fraction (0.7),
      lora_r (0 = disabled), lora_alpha (16).

    Stores on ``model``:
      _slime_adapter_budget: LayerBudgetTracker
      _slime_adapter_handles: List[MoELayerHandle]
    """
    init_bias = float(getattr(args, "gate_init_bias", -2.0))
    use_pressure = bool(getattr(args, "use_pressure_input", True))
    cw = int(getattr(args, "cache_window", LayerCache.DEFAULT_WINDOW))
    cc = int(getattr(args, "cache_cap", LayerCache.DEFAULT_CAP))
    fraction = float(getattr(args, "budget_fraction", 0.7))
    set_dim = int(getattr(args, "expert_set_embed_dim", 0))     # 0 = disable DeepSets ctx
    chunk_size = int(getattr(args, "chunk_size", 8))             # for L_chunk_consistency
    chunk_consistency_enabled = bool(getattr(args, "chunk_consistency_enabled", True))
    lora_r = int(getattr(args, "lora_r", 0))                    # 0 = no LoRA
    lora_alpha = int(getattr(args, "lora_alpha", 16))

    handles = list(adapter.iter_moe_layers(model))
    if not handles:
        raise RuntimeError(
            f"adapter {adapter.__class__.__name__} found no MoE layers in this model. "
            f"Verify ``adapter.iter_moe_layers`` and the model wrapping."
        )

    model._slime_adapter_budget = LayerBudgetTracker(total_credits=fraction * len(handles))
    model._slime_adapter_handles = handles
    model._slime_adapter_adapter = adapter

    for handle in handles:
        wrap_moe_layer(
            adapter=adapter,
            handle=handle,
            init_bias=init_bias,
            use_pressure_input=use_pressure,
            cache_window=cw,
            cache_cap=cc,
            expert_set_embed_dim=set_dim,
            chunk_size=chunk_size,
            chunk_consistency_enabled=chunk_consistency_enabled,
        )

    # LoRA on expert FFN layers + router unfreezing.
    # Parameter freezing is handled by slime's --only-train-params-name-list
    # in the production path; freeze_base_params is only needed for standalone
    # training (smoke tests without slime).
    if lora_r > 0:
        from slime_adapter.modeling.lora import (
            apply_expert_lora,
            patch_router_gate_recompute,
        )
        apply_expert_lora(model, adapter, r=lora_r, alpha=lora_alpha)
        patch_router_gate_recompute(model, adapter)

    return handles


def wrap_moe_layer(
    *,
    adapter: MoEModelAdapter,
    handle: MoELayerHandle,
    init_bias: float = -2.0,
    use_pressure_input: bool = True,
    cache_window: int = 16,
    cache_cap: int = 30,
    expert_set_embed_dim: int = 0,
    chunk_size: int = 8,
    chunk_consistency_enabled: bool = True,
) -> None:
    """Wrap a single MoE layer's forward (idempotent).

    expert_set_embed_dim > 0 enables DeepSets context: SwitchHead receives
    cache-set + router-top-K embeddings (each ``[B, T, set_dim]``) on top of
    the (hidden, pressure) inputs. 0 disables (legacy v4 behaviour).
    """
    layer = handle.module
    if getattr(layer, "_slime_adapter_wrapped", False):
        return

    set_dim = int(expert_set_embed_dim)
    sh = SwitchHead(
        hidden_size=handle.hidden_size,
        init_bias=init_bias,
        use_pressure_input=use_pressure_input,
        cache_set_dim=set_dim,
        topk_set_dim=set_dim,
    )
    # Place the SwitchHead on the same device/dtype as the host layer.
    host_param = next((p for p in layer.parameters()), None)
    if host_param is not None:
        sh = sh.to(device=host_param.device, dtype=host_param.dtype if host_param.dtype.is_floating_point else None)
    adapter.install_switch_head(handle, sh, attr="switch_head")

    # DeepSets encoder over experts (per-layer; embedding is layer-local).
    if set_dim > 0:
        encoder = ExpertSetEncoder(
            num_experts=handle.num_experts,
            embed_dim=max(32, set_dim // 2),
            set_dim=set_dim,
        )
        if host_param is not None:
            encoder = encoder.to(
                device=host_param.device,
                dtype=host_param.dtype if host_param.dtype.is_floating_point else None,
            )
        layer.expert_set_encoder = encoder
    else:
        layer.expert_set_encoder = None

    # Two cache states coexist: the legacy single-trajectory Python deque is kept
    # for diagnostics; the batched tensor cache is what the train forward uses.
    layer.cache_state = LayerCache(window=cache_window, cap=cache_cap)
    layer.batched_cache = BatchedLayerCache(
        num_experts=handle.num_experts,
        window=cache_window,
        cap=cache_cap,
    )
    layer.controller_replay = ControllerReplay()
    layer.current_top2 = None
    layer._budget_state = None
    layer._slime_adapter_ref = adapter
    layer._slime_adapter_original_forward = layer.forward
    # Chunk-consistency switch: True ⇒ use the real chunk_routing_consistency_loss
    # when router_logits are exposed by the adapter; False ⇒ always 0.
    layer._chunk_consistency_enabled = bool(chunk_consistency_enabled)
    layer._chunk_size = int(chunk_size)


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
    B = int(hidden_states_proxy.shape[0])
    device = hidden_states_proxy.device
    for layer in _iter_wrapped_layers(model):
        layer._budget_state = state
        # Reset per-layer batched cache for the new forward — every train
        # forward starts with an empty cache (rolling window grows from t=0).
        if hasattr(layer, "batched_cache") and layer.batched_cache is not None:
            layer.batched_cache.begin_batch(batch_size=B, device=device)
        # Also reset the legacy single-traj cache so its diagnostic state
        # doesn't leak between forwards.
        if hasattr(layer, "cache_state") and hasattr(layer.cache_state, "reset"):
            layer.cache_state.reset()
        layer.current_top2 = None
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

    # 1. Pressure (forward-only, detached) and router top-2 (no grad).
    pressure = state.pressure_at_entry()                                # [B, T] no-grad
    new_top2 = adapter.compute_router_top_k(layer, hidden_states, k=2)  # [B, T, 2]

    # 2. SwitchHead decision (with optional DeepSets context) +
    #    BatchedLayerCache update, sequential across T.
    sigma, switch, used_top2, n_new = _run_switch_and_cache(
        layer=layer,
        hidden_states=hidden_states,
        pressure=pressure,
        new_top2=new_top2,
        current_top2=layer.current_top2,
        device=hidden_states.device,
    )

    # 3. Charge cost + accumulate Bernoulli logπ for the joint-actor PG term.
    state.charge_layer_with_logp(switch, n_new, sigma)

    # 3.b Chunk-routing consistency.
    #     Adapter caches its router logits on ``layer._slime_router_logits``
    #     inside compute_router_top_k. If the caller enables
    #     ``layer._chunk_consistency_enabled`` we use the real loss; otherwise
    #     compute_chunk_consistency returns 0.
    router_logits = getattr(layer, "_slime_router_logits", None)
    chunk_enabled = bool(getattr(layer, "_chunk_consistency_enabled", True))
    chunk_loss = compute_chunk_consistency(
        router_logits=router_logits,
        chunk_size=int(getattr(layer, "_chunk_size", 8)),
        enabled=chunk_enabled and router_logits is not None,
        device=hidden_states.device,
    )
    if state.chunk_consistency_loss is None:
        state.chunk_consistency_loss = chunk_loss
    else:
        state.chunk_consistency_loss = state.chunk_consistency_loss + chunk_loss

    # 4. Update legacy single-trajectory diagnostic cache.
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


def _run_switch_and_cache(
    *,
    layer,
    hidden_states: torch.Tensor,            # [B, T, H]
    pressure: torch.Tensor,                  # [B, T]
    new_top2: torch.Tensor,                  # [B, T, k] long  (router preference)
    current_top2: Optional[torch.Tensor],    # [B, k] or None  (carried-over from previous token)
    device,
):
    """Run the per-layer SwitchHead + cache update.

    Two paths:

      * **Vectorized** (no DeepSets context, ``layer.expert_set_encoder is None``):
        single ``SwitchHead([B, T, H], pressure)`` call, n_new computed via
        ``_batched_n_new_loop`` after switch decisions are made.

      * **Sequential** (DeepSets context active): walk t = 0..T-1 so SwitchHead
        sees the *current* cache state and the router top-K candidates at every
        position, before the switch decision is made for that token.

    Returns ``(sigma, switch, used_top2, n_new)`` each with shape `[B, T]` /
    `[B, T, k]` / `[B, T]`.
    """
    encoder = getattr(layer, "expert_set_encoder", None)
    cache: BatchedLayerCache = layer.batched_cache
    B, T, _ = hidden_states.shape
    if cache._count is None or cache._batch_size != B or cache._count.device != hidden_states.device:
        cache.begin_batch(batch_size=B, device=hidden_states.device)

    if encoder is None:
        # ===== legacy vectorized path =====
        sigma = torch.sigmoid(layer.switch_head(hidden_states, pressure))
        switch = ste_binary(sigma)
        used_top2 = _select_used_top2(current_top2, new_top2, switch)
        n_new = _batched_n_new_loop(used_top2, layer=layer, device=device)
        return sigma, switch, used_top2, n_new

    # ===== sequential path with DeepSets =====
    sigmas: list[torch.Tensor] = []
    switches: list[torch.Tensor] = []
    n_news: list[torch.Tensor] = []
    used_steps: list[torch.Tensor] = []
    carry = current_top2  # [B, k] or None

    for t in range(T):
        # Cache reps from the cache state BEFORE pushing this token's decision.
        cache_mask = (layer.batched_cache.count > 0)            # [B, E] bool
        cache_rep = layer.expert_set_encoder.encode_mask(cache_mask)         # [B, set_dim]
        top2_rep  = layer.expert_set_encoder.encode_indices(new_top2[:, t]) # [B, set_dim]

        h_t = hidden_states[:, t]                              # [B, H]
        p_t = pressure[:, t]                                   # [B]
        sigma_t = torch.sigmoid(layer.switch_head(
            h_t, p_t, cache_set_repr=cache_rep, top_k_set_repr=top2_rep,
        ))                                                      # [B]
        switch_t = ste_binary(sigma_t)                          # [B]

        # used_top2_t = switch ? new_top2_t : carry
        new_t = new_top2[:, t]
        if carry is None:
            used_t = new_t
        else:
            cond = switch_t.unsqueeze(-1) > 0.5
            used_t = torch.where(cond, new_t, carry)

        # n_new before the push
        n_new_t = layer.batched_cache.n_new(used_t)             # [B] long
        layer.batched_cache.push(used_t)                        # advance window

        sigmas.append(sigma_t)
        switches.append(switch_t)
        n_news.append(n_new_t)
        used_steps.append(used_t)
        carry = used_t.detach()

    sigma = torch.stack(sigmas, dim=1)                           # [B, T]
    switch = torch.stack(switches, dim=1)                        # [B, T]
    used_top2 = torch.stack(used_steps, dim=1)                   # [B, T, k]
    n_new = torch.stack(n_news, dim=1).to(torch.int64)           # [B, T]
    return sigma, switch, used_top2, n_new


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


def _batched_n_new_loop(
    used_top2: torch.Tensor,           # [B, T, k] long
    *,
    layer: nn.Module,
    device,
) -> torch.Tensor:
    """Sequentially advance the BatchedLayerCache over T positions.

    For each step t, returns ``n_new[:, t] = count(used_top2[:, t, :] not in
    cache_at_step_t)`` and then pushes used_top2[:, t, :] onto the cache so the
    next step sees the updated history. Each batch row owns its own cache
    state — replaces the legacy "shared union across batch" approximation.
    """
    cache: BatchedLayerCache = layer.batched_cache
    B, T, _ = used_top2.shape
    if cache._count is None or cache._batch_size != B or cache._count.device != device:
        cache.begin_batch(batch_size=B, device=device)
    out = torch.empty(B, T, dtype=torch.int64, device=device)
    for t in range(T):
        slice_t = used_top2[:, t, :].long()                    # [B, k]
        out[:, t] = cache.n_new(slice_t)                       # [B]
        cache.push(slice_t)                                    # advance window
    return out


def _compute_n_new_batched(new_top2: torch.Tensor, cache: LayerCache) -> torch.Tensor:
    """**Legacy** single-Python-cache n_new (still used by unit tests).

    Computes ``[B, T]``: how many of ``new_top2[b, t]`` are NOT in
    ``cache.union``. Treats one Python LayerCache as the shared cache state
    across the whole batch — this is the approximation that the new
    BatchedLayerCache path replaces. Kept here so existing tests remain valid.
    """
    if cache.size == 0:
        return torch.full(new_top2.shape[:2], new_top2.shape[-1],
                          dtype=torch.long, device=new_top2.device)
    union_ids = torch.tensor(list(cache.union), dtype=new_top2.dtype, device=new_top2.device)
    in_cache = (new_top2.unsqueeze(-1) == union_ids.view(1, 1, 1, -1)).any(dim=-1)
    return (~in_cache).long().sum(dim=-1)


def _iter_wrapped_layers(model: nn.Module) -> Iterable[nn.Module]:
    for m in model.modules():
        if getattr(m, "_slime_adapter_wrapped", False):
            yield m


# Keep ``_iter_wrapped_layers`` accessible under the name used by other files
_iter_wrapped_layers = _iter_wrapped_layers
