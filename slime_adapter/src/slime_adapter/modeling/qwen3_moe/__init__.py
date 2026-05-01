"""Qwen3-MoE adapter package.

Importing this module registers the ``"qwen3_moe"`` adapter under the global
adapter registry. Use it via:

    from slime_adapter.modeling import get_adapter
    adapter = get_adapter("qwen3_moe")
"""

from .adapter import Qwen3MoEAdapter
from .._registry import register_adapter

register_adapter("qwen3_moe", Qwen3MoEAdapter)

__all__ = ["Qwen3MoEAdapter"]
