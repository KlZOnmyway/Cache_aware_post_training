"""slime_adapter — cache-aware MoE distillation plugin for slime.

The package itself does NOT import slime / Megatron / SGLang at top level so
that the model-agnostic core can be used in isolation (unit tests, ablations,
rollout-only mode, …). Side-effecting monkey-patches live under the
`slime_adapter.megatron_hooks` and `slime_adapter.sglang_patches` subpackages
and are only triggered when those modules are imported.

Public API:

    from slime_adapter import SwitchHead, LayerCache, CreditsTracker, ste_binary
    from slime_adapter.modeling import get_adapter, register_adapter

Side-effect imports (apply at module load time):

    import slime_adapter.megatron_hooks       # patches Megatron's compute_topk
    import slime_adapter.sglang_patches       # patches sglang select_experts
    import slime_adapter.loss.penalty_loss    # patches slime policy_loss_function
"""

from .controller.switch_head import SwitchHead
from .controller.ste import ste_binary, ste_binary_with_temperature
from .controller.cache_state import LayerCache
from .controller.credits import CreditsTracker

__all__ = [
    "SwitchHead",
    "LayerCache",
    "CreditsTracker",
    "ste_binary",
    "ste_binary_with_temperature",
]

__version__ = "0.1.0"
