"""Megatron-side import-time monkey patches.

Importing this package wires up:

  1. ``compute_topk_patch`` — extends slime's RoutingReplay so we can record
     and replay the per-(t, l) top-K indices used by the model.
  2. ``moe_forward_patch`` — wraps each MoE layer's forward to consult our
     ``SwitchHead`` and update the per-layer cache.
  3. ``budget`` — autograd-aware credit accumulator (per-token tensor).

The patches do **NOT** auto-apply on import — call ``install_all(model, args)``
from your training entry point after the model is built but before forward.

This keeps the import side-effect-free, so `import slime_adapter` is safe in
testing / CI environments without Megatron.
"""

from .compute_topk_patch import register_routing_replay_extensions, ControllerReplay
from .moe_forward_patch import (
    install_controller_into_layers,
    begin_controller_forward,
    end_controller_forward,
    wrap_moe_layer,
)
from .budget import LayerBudgetTracker, TokenBudgetState, BudgetReadout
from .driver import (
    install_forward_driver,
    install_forward_completion_hook,
    controller_forward_session,
)

__all__ = [
    "register_routing_replay_extensions",
    "ControllerReplay",
    "install_controller_into_layers",
    "wrap_moe_layer",
    "begin_controller_forward",
    "end_controller_forward",
    "LayerBudgetTracker",
    "TokenBudgetState",
    "BudgetReadout",
    "install_forward_driver",
    "install_forward_completion_hook",
    "controller_forward_session",
    "install_all",
]


def install_all(model, args, adapter=None, *, install_forward_hooks: bool = True):
    """Install all megatron-side patches onto a built model.

    Args:
        model: the model module (after Megatron build, DDP wrappers OK).
        args: namespace with controller flags (``--moe-arch``, ``--cache-window``,
              ``--cache-cap``, ``--budget-fraction``, ``--gate-init-bias``,
              ``--use-pressure-input`` etc.).
        adapter: pre-instantiated ``MoEModelAdapter``; if None, looked up via
            ``slime_adapter.modeling.get_adapter(args.moe_arch)``.
        install_forward_hooks: if True (default), also register the
            forward-pre / forward-post hooks that drive
            ``begin_controller_forward`` / ``end_controller_forward``
            automatically. Set False if you want explicit control via
            ``controller_forward_session``.

    Side effects:
        - registers slime's RoutingReplay extension on Megatron's compute_topk
        - attaches a ``SwitchHead`` and per-layer cache state to each MoE layer
        - patches each MoE layer's forward to invoke the controller
        - (optional) registers forward hooks on the top-level model so
          per-step budget state is auto-allocated and torn down

    Returns:
        Tuple of (handles, adapter) — handles list of removable hooks (empty
        if ``install_forward_hooks=False``); adapter is the resolved
        ``MoEModelAdapter`` instance.
    """
    from slime_adapter.modeling import get_adapter

    if adapter is None:
        adapter = get_adapter(getattr(args, "moe_arch", "qwen3_moe"))

    register_routing_replay_extensions()
    install_controller_into_layers(model, adapter=adapter, args=args)

    handles = []
    if install_forward_hooks:
        handles.append(install_forward_driver(model))
        handles.append(install_forward_completion_hook(model))
    return handles, adapter
