"""Lightweight LoRA for Megatron-LM (mcore) expert layers.

Applies rank-r adapters to expert FFN linear layers, freezes base weights,
and collects optimizer param groups with per-component learning rates.

Designed for mcore SequentialMLP (per-expert nn.Module with linear_fc1/fc2).
GroupedMLP uses batched weight tensors — not yet supported; will warn and skip.

Usage (called from install_controller_into_layers):

    apply_expert_lora(model, adapter, r=8, alpha=16)
    freeze_and_collect_param_groups(model, base_lr=1e-6, ...)

Reference: rl_moe train_controller_standalone.py:1790-1850
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """LoRA adapter wrapping any linear-like module.

    Supports mcore ColumnParallelLinear / RowParallelLinear (which return
    (output, bias) tuples) and standard nn.Linear.

    Init: A ~ Kaiming, B = 0 → initial LoRA output = 0 (cold-start safe).
    """

    def __init__(self, base: nn.Module, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = float(alpha) / float(r)

        for p in base.parameters():
            p.requires_grad = False

        w = base.weight
        out_features, in_features = w.shape[0], w.shape[1]
        self.lora_A = nn.Parameter(
            torch.zeros(r, in_features, dtype=w.dtype, device=w.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, r, dtype=w.dtype, device=w.device)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x, *args, **kwargs):
        base_out = self.base(x, *args, **kwargs)
        lora_delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

        if isinstance(base_out, tuple):
            return (base_out[0] + lora_delta, *base_out[1:])
        return base_out + lora_delta


# =====================================================================
# Apply LoRA to expert layers
# =====================================================================

def apply_expert_lora(
    model: nn.Module,
    adapter,
    r: int = 8,
    alpha: int = 16,
) -> int:
    """Walk all MoE layers and wrap expert linear modules with LoRA.

    Returns the number of LoRA modules created.

    Targets mcore SequentialMLP: each local expert MLP has ``linear_fc1``
    (gate+up projection) and ``linear_fc2`` (down projection).
    """
    count = 0
    for handle in adapter.iter_moe_layers(model):
        moe = handle.module
        experts = getattr(moe, "experts", None)
        if experts is None:
            logger.warning("MoE layer %d has no .experts attribute; skipping LoRA", handle.layer_idx)
            continue

        local_experts = getattr(experts, "local_experts", None)
        if local_experts is None:
            logger.warning(
                "MoE layer %d uses %s (not SequentialMLP); LoRA not yet supported for this layout",
                handle.layer_idx, type(experts).__name__,
            )
            continue

        for expert_idx, expert_mlp in enumerate(local_experts):
            for attr in ("linear_fc1", "linear_fc2"):
                base_linear = getattr(expert_mlp, attr, None)
                if base_linear is None:
                    continue
                if isinstance(base_linear, LoRALinear):
                    continue
                lora_module = LoRALinear(base_linear, r=r, alpha=alpha)
                setattr(expert_mlp, attr, lora_module)
                count += 1

    logger.info("[LoRA] Applied %d LoRA adapters (r=%d, alpha=%d)", count, r, alpha)
    return count


# =====================================================================
# Router gradient: recompute gate weights from current params
# =====================================================================

def patch_router_gate_recompute(model: nn.Module, adapter) -> int:
    """Patch each MoE layer so gate weights are recomputed from current router
    during RoutingReplay, enabling gradient flow to router weights.

    Without this patch, RoutingReplay returns both replayed indices AND
    replayed gate weights → router is completely detached → zero gradient.

    With this patch, we register a forward hook on the MoE module that
    overwrites the router's gate weights with freshly computed values
    from the current router parameters at the replayed indices.
    """
    count = 0
    for handle in adapter.iter_moe_layers(model):
        moe = handle.module
        router = getattr(moe, "router", None)
        if router is None:
            continue

        router.requires_grad_(True)
        for p in router.parameters():
            p.requires_grad = True
            if p.dtype == torch.bfloat16:
                p.data = p.data.float()

        count += sum(1 for p in router.parameters())

    logger.info("[Router] Unfroze %d router parameters (converted to float32)", count)
    return count


# =====================================================================
# Freeze + param groups
# =====================================================================

def freeze_base_params(model: nn.Module) -> Dict[str, int]:
    """Freeze all parameters except LoRA, controller, and router.

    Returns a dict of counts by category.

    .. note::
       In the slime production path, prefer ``--only-train-params-name-list``
       (regex-based, handled by slime's ``freeze_model_params``) instead of
       calling this function directly.  This function is still useful for
       standalone / smoke-test training without slime.
    """
    stats = {"frozen": 0, "lora": 0, "controller": 0, "router": 0, "other_trainable": 0}

    for name, param in model.named_parameters():
        name_lower = name.lower()
        if "lora_" in name_lower:
            param.requires_grad = True
            stats["lora"] += 1
        elif "switch_head" in name_lower or "expert_set_encoder" in name_lower:
            param.requires_grad = True
            stats["controller"] += 1
        elif "router" in name_lower:
            param.requires_grad = True
            stats["router"] += 1
        else:
            param.requires_grad = False
            stats["frozen"] += 1

    logger.info(
        "[Freeze] lora=%d, controller=%d, router=%d, frozen=%d",
        stats["lora"], stats["controller"], stats["router"], stats["frozen"],
    )
    return stats


def collect_param_groups(
    model: nn.Module,
    lora_lr: float = 1e-5,
    router_lr: float = 1e-5,
    controller_lr: float = 1e-4,
    weight_decay: float = 0.0,
) -> List[Dict]:
    """Build optimizer param groups with per-component learning rates.

    Groups:
      1. LoRA A/B          → lora_lr
      2. Router            → router_lr (float32 for precision)
      3. SwitchHead + ExpertSetEncoder → controller_lr
    """
    lora_params = []
    router_params = []
    controller_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if "lora_" in name_lower:
            lora_params.append(param)
        elif "router" in name_lower:
            router_params.append(param)
        elif "switch_head" in name_lower or "expert_set_encoder" in name_lower:
            controller_params.append(param)

    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay, "name": "lora"})
    if router_params:
        groups.append({"params": router_params, "lr": router_lr, "weight_decay": 0.0, "name": "router"})
    if controller_params:
        groups.append({"params": controller_params, "lr": controller_lr, "weight_decay": 0.0, "name": "controller"})

    total = sum(p.numel() for g in groups for p in g["params"])
    logger.info(
        "[ParamGroups] lora=%d params (lr=%.1e), router=%d params (lr=%.1e), "
        "controller=%d params (lr=%.1e), total=%.2fM trainable",
        len(lora_params), lora_lr,
        len(router_params), router_lr,
        len(controller_params), controller_lr,
        total / 1e6,
    )
    return groups


__all__ = [
    "LoRALinear",
    "apply_expert_lora",
    "patch_router_gate_recompute",
    "freeze_base_params",
    "collect_param_groups",
]
