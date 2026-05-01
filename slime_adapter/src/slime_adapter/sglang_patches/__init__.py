"""SGLang-side patches: per-request controller state + MoE select_experts hook.

Importing this package does NOT auto-apply the patch — call ``apply_patches()``
once at SGLang server startup (e.g. from a custom server entrypoint).

Architecture:
  - ``request_state.RequestControllerState``:
        per-request scratchpad (cache deque per layer + credits tracker).
  - ``moe_select_patch.patch_select_experts``:
        replaces ``sglang.srt.layers.moe.topk.select_experts`` (or whatever the
        active SGLang version exposes) with a controller-aware version that:
          (1) reads/writes RequestControllerState
          (2) honors switch decisions
          (3) returns the controller-decided used_top2 to the rest of the MoE.
"""

from .request_state import RequestControllerState, get_current_state, set_current_state, CURRENT_STATE
from .moe_select_patch import (
    apply_patches,
    restore_patches,
    register_switch_head,
    get_switch_head,
    clear_switch_heads,
)
from .weight_sync import (
    serialize_switch_heads,
    deserialize_and_register,
    push_to_sglang_servers,
)

__all__ = [
    "RequestControllerState",
    "get_current_state",
    "set_current_state" if False else "set_current_state",
    "CURRENT_STATE",
    "apply_patches",
    "restore_patches",
    "register_switch_head",
    "get_switch_head",
    "clear_switch_heads",
    "serialize_switch_heads",
    "deserialize_and_register",
    "push_to_sglang_servers",
]
