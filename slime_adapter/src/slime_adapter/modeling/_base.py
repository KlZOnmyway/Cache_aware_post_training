"""Abstract MoE-architecture adapter.

This is the **only** layer of the slime_adapter codebase that talks to a
specific MoE model. Subclass this class for each architecture you want to
support (Qwen3-MoE, gpt-oss, Mixtral, DeepSeek-V3, ...). The controller core
(SwitchHead, LayerCache, CreditsTracker, STE), the loss patches, and the
rollout reward function are all model-agnostic and don't import this module.

Three responsibilities:

  1. **Discovery** — given a model object, enumerate its MoE layers
     (``iter_moe_layers``).
  2. **Routing** — read the layer's natural top-K choice (``compute_router_top_k``).
  3. **Forced forward** — run the MoE layer's compute with the experts forced
     to a given set of indices (``forward_with_forced_top_indices``). This is
     where the controller's switch decision actually takes effect.

Adapters for new architectures register themselves at import time via
``register_adapter`` (in ``_registry.py``). User-facing CLI selects one with
``--moe-arch``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional

import torch
import torch.nn as nn


@dataclass
class MoELayerHandle:
    """Lightweight handle to one MoE layer in a model.

    Attributes:
        layer_idx:    index among MoE layers (0..L-1).
        module:       the layer's MoE submodule (where router/experts live).
        hidden_size:  H — needed to size SwitchHead.
        num_experts:  E — total experts in this layer.
        native_top_k: the model's *native* top_k (e.g. 8 for Qwen3-MoE-30B).
                      Our controller uses k=2 regardless; this is informational.
    """

    layer_idx: int
    module: nn.Module
    hidden_size: int
    num_experts: int
    native_top_k: int


@dataclass
class ControllerRuntime:
    """Per-forward-pass shared scratchpad.

    Forward path appends to these as it walks the layers; loss path reads them
    after the model forward returns. Reset every train step.
    """

    layer_local_costs: List[torch.Tensor] = field(default_factory=list)  # each: [B, T]
    total_used_per_token: torch.Tensor | None = None  # [B, T] (set at token end)
    chunk_consistency_loss: torch.Tensor | None = None  # scalar
    record_actions: dict = field(default_factory=dict)  # for debugging / inspection

    def clear(self) -> None:
        self.layer_local_costs.clear()
        self.total_used_per_token = None
        self.chunk_consistency_loss = None
        self.record_actions.clear()


class MoEModelAdapter(abc.ABC):
    """Subclass once per MoE architecture. Methods marked @abstractmethod must
    be implemented; the rest have reasonable defaults."""

    #: Short adapter id, matches the ``--moe-arch`` CLI flag.
    name: str = "abstract"

    # ------------------------------------------------------------------
    # 1. Discovery
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def iter_moe_layers(self, model: nn.Module) -> "Iterator[MoELayerHandle]":
        """Yield one ``MoELayerHandle`` per MoE layer in the model."""

    def num_moe_layers(self, model: nn.Module) -> int:
        """Default: counts via ``iter_moe_layers``."""
        return sum(1 for _ in self.iter_moe_layers(model))

    # ------------------------------------------------------------------
    # 2. Routing introspection
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def compute_router_top_k(
        self,
        moe_module: nn.Module,
        hidden_states: torch.Tensor,
        k: int = 2,
    ) -> torch.Tensor:
        """Return ``[B, T, k]`` LongTensor of expert ids the router prefers.

        Implementations should use the model's own router scoring + topk to
        stay consistent with what gets recorded by slime's RoutingReplay.
        """

    # ------------------------------------------------------------------
    # 3. Forward override
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def forward_with_forced_top_indices(
        self,
        moe_module: nn.Module,
        hidden_states: torch.Tensor,
        forced_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run MoE forward but force the router output to ``forced_indices``.

        Args:
            moe_module:       the layer's MoE submodule.
            hidden_states:    [B, T, H].
            forced_indices:   [B, T, k] LongTensor of expert ids to use.

        Returns:
            [B, T, H] post-MoE output.

        For mcore-based MoE we typically achieve this by stuffing
        ``forced_indices`` into the routing replay buffer that ``compute_topk``
        will consume, then calling the layer's normal forward.
        """

    # ------------------------------------------------------------------
    # 4. SwitchHead installation
    # ------------------------------------------------------------------
    def install_switch_head(
        self,
        handle: MoELayerHandle,
        switch_head: nn.Module,
        attr: str = "switch_head",
    ) -> None:
        """Attach SwitchHead to the layer module so the forward hook can find it."""
        if hasattr(handle.module, attr):
            raise AttributeError(
                f"{type(handle.module).__name__} already has attribute {attr!r}; "
                f"refusing to overwrite. Did install_switch_head() run twice?"
            )
        handle.module.add_module(attr, switch_head)


__all__ = [
    "MoELayerHandle",
    "ControllerRuntime",
    "MoEModelAdapter",
]
