"""Extension to slime's ``RoutingReplay`` to also carry controller-side fields.

slime already monkey-patches Megatron's ``compute_topk`` to record/replay
the layer's expert top-K (``slime.utils.routing_replay``). We mirror that
machinery for our **controller-side** state: per-layer-per-token

  - ``switch``            (binary 0/1 from STE)
  - ``used_top2``         (the experts actually used — equals new_top2 if switch=1, current_top2 otherwise)
  - ``new_top2``          (router argmax-2, even if not used)
  - ``n_new``             (cache misses incurred at this layer)
  - ``pressure_in``       (forward-only feature fed to switch_head)

Each MoE layer has one ``ControllerReplay`` instance (parallel to its
``RoutingReplay`` instance). The forward patch in ``moe_forward_patch``
records into it; the loss patch reads from it during backward.

This file does NOT touch Megatron itself. It just provides the data structures
and ``register_routing_replay_extensions()`` for env-time bootstrap.
"""

from __future__ import annotations

import os
from typing import Dict, List

import torch


class ControllerReplay:
    """Per-MoE-layer record/replay buffer for controller-side decisions.

    Mirrors the lifecycle of slime's ``RoutingReplay``: one instance per layer,
    accumulating entries over the forward pass; reset between train steps.
    """

    all_instances: List["ControllerReplay"] = []

    def __init__(self) -> None:
        self.entries: List[Dict[str, torch.Tensor]] = []
        self._forward_idx: int = 0
        self._backward_idx: int = 0
        ControllerReplay.all_instances.append(self)

    # -- record path ----------------------------------------------------
    def record(
        self,
        *,
        switch: torch.Tensor,
        used_top2: torch.Tensor,
        new_top2: torch.Tensor,
        n_new: torch.Tensor,
        pressure_in: torch.Tensor,
    ) -> None:
        """Append one (token-batch, layer) record. Tensors are moved to pinned CPU."""

        def _stash(t: torch.Tensor) -> torch.Tensor:
            # Pinned memory only makes sense when copying from a CUDA tensor.
            if torch.cuda.is_available() and t.is_cuda:
                buf = torch.empty_like(t, device="cpu", pin_memory=True)
                buf.copy_(t.detach(), non_blocking=True)
            else:
                buf = t.detach().to("cpu", copy=True)
            return buf

        self.entries.append({
            "switch": _stash(switch),
            "used_top2": _stash(used_top2),
            "new_top2": _stash(new_top2),
            "n_new": _stash(n_new),
            "pressure_in": _stash(pressure_in),
        })

    # -- replay path ----------------------------------------------------
    def pop_forward(self) -> Dict[str, torch.Tensor]:
        rec = self.entries[self._forward_idx]
        self._forward_idx += 1
        return {k: v.to(torch.cuda.current_device(), non_blocking=True) for k, v in rec.items()}

    def pop_backward(self) -> Dict[str, torch.Tensor]:
        rec = self.entries[self._backward_idx]
        self._backward_idx += 1
        return {k: v.to(torch.cuda.current_device(), non_blocking=True) for k, v in rec.items()}

    # -- lifecycle ------------------------------------------------------
    def reset(self) -> None:
        self.entries = []
        self._forward_idx = 0
        self._backward_idx = 0

    def reset_forward(self) -> None:
        self._forward_idx = 0

    @classmethod
    def reset_all(cls) -> None:
        for r in cls.all_instances:
            r.reset()

    @classmethod
    def reset_all_forward(cls) -> None:
        for r in cls.all_instances:
            r.reset_forward()


def register_routing_replay_extensions() -> None:
    """Bootstrap: ensure slime's routing_replay module is importable, no-op otherwise.

    Idempotent — safe to call multiple times.
    """
    if os.environ.get("SLIME_ADAPTER_REPLAY_INIT", "0") == "1":
        return
    try:
        import slime.utils.routing_replay  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "slime is not importable. Install via `pip install -e ../external/slime` "
            "before importing slime_adapter.megatron_hooks."
        ) from e
    os.environ["SLIME_ADAPTER_REPLAY_INIT"] = "1"
