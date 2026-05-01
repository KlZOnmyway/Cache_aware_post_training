"""Qwen3-MoE concrete adapter (Megatron-LM / mcore + transformer-engine path).

Reference target: ``Qwen3-MoE-30B-A3B`` running through slime's standard
training stack (see ``slime/tests/test_qwen3_30B_A3B.py``). Qwen3-MoE
properties this adapter relies on:

  - Each transformer layer's MLP is a fully-MoE block accessible at ``layer.mlp``.
  - The MoE block exposes ``.router`` (mcore TopKRouter) with ``.weight``
    ([num_experts, hidden]) and optional ``.bias``.
  - Native ``router.top_k`` defaults to 8 — but our controller forces top-2
    via the slime ``RoutingReplay`` mechanism.

Design contract with the rest of slime_adapter
----------------------------------------------

This adapter does **not** override mcore's expert dispatch from scratch.
Instead it relies on slime's monkey-patched ``compute_topk`` (see
``slime/utils/routing_replay.py``): when ``ENABLE_ROUTING_REPLAY=1`` +
``ROUTING_REPLAY_STAGE=replay_forward``, ``compute_topk`` returns indices
popped from each layer's ``RoutingReplay.top_indices_list`` instead of
calling the model's natural router topk.

The trainer-side wiring is therefore:

  1. At rollout time (SGLang), our switch_head + cache produces ``used_top2``
     per (layer, token); the SGLang patch returns these as the layer's
     "selected experts" — slime then records them via its routing-replay
     forward hook.
  2. At training time (Megatron), slime's actor sets stage=replay_forward
     and the per-layer replay buffer is filled. Our adapter's
     ``forward_with_forced_top_indices`` is then a thin wrapper around the
     layer's original forward — slime handles the index override transparently.

For other models (Mixtral, DeepSeek-V3, GLM-4-MoE), copy this file and
adjust the router-walking helpers; the forward override is identical.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
import torch.nn as nn

from .._base import MoELayerHandle, MoEModelAdapter


class Qwen3MoEAdapter(MoEModelAdapter):
    name = "qwen3_moe"

    # ==================================================================
    # 1. Discovery: walk the model's transformer layers, yield MoE blocks.
    # ==================================================================
    def iter_moe_layers(self, model: nn.Module) -> Iterator[MoELayerHandle]:
        layers = self._locate_layer_list(model)
        H = self._infer_hidden_size(model, layers)
        moe_idx = 0
        for layer in layers:
            mlp = self._get_mlp(layer)
            if mlp is None or not self._is_moe(mlp):
                continue
            yield MoELayerHandle(
                layer_idx=moe_idx,
                module=mlp,
                hidden_size=H,
                num_experts=self._read_num_experts(mlp),
                native_top_k=self._read_native_top_k(mlp),
            )
            moe_idx += 1

    # ==================================================================
    # 2. Routing introspection: get the natural top-k for our controller.
    # ==================================================================
    def compute_router_top_k(self, moe_module, hidden_states, k: int = 2):
        """Return ``[..., k]`` LongTensor: router's natural argmax-k.

        We bypass the full mcore router forward (softmax + dispatch state) and
        read out the raw linear scores directly. softmax is monotonic so
        argmax-k(logits) == argmax-k(softmax(logits)).

        Side effect: caches the full ``[..., E]`` router logits on
        ``moe_module._slime_router_logits`` so the wrapped forward can pass
        them to ``compute_chunk_consistency`` later. Top-K selection itself
        runs under ``no_grad``; the chunk-loss path detaches anyway.
        """
        weight, bias = self._get_router_params(moe_module.router)
        logits = torch.nn.functional.linear(hidden_states, weight, bias)
        moe_module._slime_router_logits = logits          # [B, T, E] for chunk_loss
        with torch.no_grad():
            return logits.topk(k=k, dim=-1).indices

    # ==================================================================
    # 3. Forward override: trusts slime's RoutingReplay mechanism.
    # ==================================================================
    def forward_with_forced_top_indices(
        self,
        moe_module,
        hidden_states,
        forced_indices,
    ):
        """Call the layer's original (pre-wrap) forward.

        Assumption: slime's RoutingReplay mechanism has been wired up:
          - ``--use-routing-replay`` was passed.
          - The per-layer ``RoutingReplay.top_indices_list`` has been
            populated with our ``used_top2`` records (typically by the
            trainer's rollout-data → replay loader at the start of each
            training micro-step).
          - The trainer is in ``ROUTING_REPLAY_STAGE=replay_forward``.

        Under those conditions the layer's natural forward will use the
        recorded ``used_top2`` as its top-k indices — which IS what we want.

        For unit tests / non-slime backends, the caller is responsible for
        either populating the routing replay buffer beforehand OR for
        accepting that ``forced_indices`` is ignored here.
        """
        from slime_adapter.megatron_hooks.moe_forward_patch import call_original_forward
        return call_original_forward(moe_module, hidden_states)

    # ==================================================================
    # internal helpers (model walking + introspection)
    # ==================================================================
    @staticmethod
    def _locate_layer_list(model: nn.Module) -> list:
        """Walk past common wrappers (DDP, Float16Module, vp_stage) to
        find ``decoder.layers``."""
        for path in (
            "module.decoder.layers",
            "module.module.decoder.layers",
            "decoder.layers",
            "module.model.decoder.layers",
            "model.layers",
        ):
            obj = model
            ok = True
            for piece in path.split("."):
                if hasattr(obj, piece):
                    obj = getattr(obj, piece)
                else:
                    ok = False
                    break
            if ok:
                return list(obj)
        raise RuntimeError(
            f"Could not locate transformer layers on {type(model).__name__}. "
            f"Override _locate_layer_list for your model wrapping."
        )

    @staticmethod
    def _get_mlp(layer: nn.Module) -> Optional[nn.Module]:
        return getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)

    @staticmethod
    def _is_moe(mlp: nn.Module) -> bool:
        return mlp is not None and hasattr(mlp, "router")

    @staticmethod
    def _get_router_params(router: nn.Module):
        if hasattr(router, "weight"):
            return router.weight, getattr(router, "bias", None)
        for sub in ("layer", "linear", "gate"):
            sub_mod = getattr(router, sub, None)
            if sub_mod is not None and hasattr(sub_mod, "weight"):
                return sub_mod.weight, getattr(sub_mod, "bias", None)
        raise RuntimeError(f"Router {type(router).__name__} has no recognizable weight matrix.")

    @staticmethod
    def _read_num_experts(mlp: nn.Module) -> int:
        for path in ("num_experts", "router.num_experts", "num_local_experts"):
            obj = mlp
            try:
                for piece in path.split("."):
                    obj = getattr(obj, piece)
                return int(obj)
            except AttributeError:
                continue
        if hasattr(mlp, "experts"):
            try:
                return len(mlp.experts)
            except TypeError:
                pass
        raise RuntimeError("Cannot determine num_experts on Qwen3 MoE block.")

    @staticmethod
    def _read_native_top_k(mlp: nn.Module) -> int:
        for path in ("router.top_k", "top_k", "num_experts_per_tok"):
            obj = mlp
            try:
                for piece in path.split("."):
                    obj = getattr(obj, piece)
                return int(obj)
            except AttributeError:
                continue
        return 8  # Qwen3-MoE-30B-A3B default

    @classmethod
    def _infer_hidden_size(cls, model: nn.Module, layers: list) -> int:
        for path in (
            "config.hidden_size",
            "module.config.hidden_size",
            "module.module.config.hidden_size",
        ):
            obj = model
            try:
                for piece in path.split("."):
                    obj = getattr(obj, piece)
                return int(obj)
            except AttributeError:
                continue
        # Fallback: read from a router weight shape.
        for layer in layers:
            mlp = cls._get_mlp(layer)
            if mlp and hasattr(mlp, "router"):
                w, _ = cls._get_router_params(mlp.router)
                if w is not None:
                    return int(w.shape[-1])
        raise RuntimeError("Cannot infer hidden_size for Qwen3-MoE.")


__all__ = ["Qwen3MoEAdapter"]
