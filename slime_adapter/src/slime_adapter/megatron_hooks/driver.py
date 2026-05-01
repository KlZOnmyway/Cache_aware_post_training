"""Forward-driver helpers for slime's training loop.

There are two ways to ensure ``begin_controller_forward`` runs at the right
time relative to the model's actual forward:

  (a) **forward_pre_hook on the top-level model** (the easy path) — slime's
      Megatron actor build process eventually returns a model module; we
      register a pre_hook on it that takes the first positional arg as the
      hidden-state proxy.

  (b) **explicit context manager** — wrap the trainer's forward call in
      ``with controller_forward_session(model, hidden_proxy):`` block.

Use (a) when you want a one-shot install at the top of training; use (b)
when you have a custom forward path (multi-step, MTP, etc.) and want
explicit control.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import torch
import torch.nn as nn

from .moe_forward_patch import begin_controller_forward, end_controller_forward


# --------------------------------------------------------------------
# Variant (a): hook-based install
# --------------------------------------------------------------------

def install_forward_driver(
    model: nn.Module,
    *,
    proxy_arg_index: int = 0,
) -> torch.utils.hooks.RemovableHandle:
    """Register a forward pre-hook on ``model`` that initializes the controller forward.

    Args:
        model: top-level model module (the same one passed to
            ``install_controller_into_layers``).
        proxy_arg_index: which positional arg of ``model.forward`` is the
            hidden-state-shaped proxy. Most slime/Megatron models take
            ``(input_ids, ...)`` as the first arg — we use ``input_ids`` to
            allocate a [B, T] tensor of zeros which is enough to size the
            ``TokenBudgetState`` buffers.

    Returns:
        A handle whose ``.remove()`` undoes the hook.
    """

    def pre_hook(module, args, kwargs):
        if not args:
            return None  # unusual call shape; let downstream blow up naturally
        proxy = args[proxy_arg_index]
        # If it's input_ids ([B, T]), allocate a [B, T] zeros to feed the
        # budget state allocator. Use float to make the budget bookkeeping
        # tensor a useful dtype.
        if proxy.dim() == 2:
            shaped = torch.zeros(
                proxy.shape[0], proxy.shape[1], 1,
                device=proxy.device, dtype=torch.float32,
            )
        else:
            shaped = proxy
        begin_controller_forward(module, shaped)
        return None

    proxy_arg_index = int(proxy_arg_index)
    return model.register_forward_pre_hook(pre_hook, with_kwargs=True)


def install_forward_completion_hook(model: nn.Module) -> torch.utils.hooks.RemovableHandle:
    """Register a forward hook that finalizes the controller state after each forward.

    The forward output is left untouched — we only run book-keeping.
    """

    def _post(module, inputs, output):
        state = end_controller_forward(module)
        # Hand the per-step controller state to the loss patch via TLS.
        # Importing here keeps the dependency one-way (driver doesn't require
        # slime to be installed; only fails silently if loss module isn't loaded).
        try:
            from slime_adapter.loss.penalty_loss import set_last_controller_state
            set_last_controller_state(state)
        except Exception:
            pass
        return output

    return model.register_forward_hook(_post)


# --------------------------------------------------------------------
# Variant (b): explicit context manager
# --------------------------------------------------------------------

@contextlib.contextmanager
def controller_forward_session(
    model: nn.Module,
    hidden_proxy: torch.Tensor,
) -> "Iterator[Any]":
    """Run a single forward with the controller wired up.

    Usage::

        with controller_forward_session(model, hidden_proxy=embeds):
            output = model(input_ids, ...)
            # output now has access to model._slime_adapter_budget.current_state
    """
    state = begin_controller_forward(model, hidden_proxy)
    try:
        yield state
    finally:
        end_controller_forward(model)


# --------------------------------------------------------------------
# Re-exports
# --------------------------------------------------------------------

from .moe_forward_patch import begin_controller_forward, end_controller_forward  # noqa: E402

__all__ = [
    "install_forward_driver",
    "install_forward_completion_hook",
    "controller_forward_session",
    "begin_controller_forward",
    "end_controller_forward",
]


import torch.nn as nn  # noqa: E402
