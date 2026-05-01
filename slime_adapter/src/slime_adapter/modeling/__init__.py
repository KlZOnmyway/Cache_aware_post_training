"""Model-specific MoE adapters.

The ``MoEModelAdapter`` abstract class (in ``slime_adapter.modeling._base``)
defines the interface that the rest of slime_adapter uses to talk to a
specific MoE architecture. Concrete adapters live in subpackages here:

  - ``slime_adapter.modeling.qwen3_moe``  — Qwen3-MoE (30B-A3B etc.)
  - (add yours here)

Adapters are registered by name via ``register_adapter``; user code selects
one with ``--moe-adapter qwen3_moe`` (or programmatically via ``get_adapter``).
"""

from ._base import MoEModelAdapter, MoELayerHandle
from ._registry import register_adapter, get_adapter, list_adapters

# Trigger registration of built-in adapters by importing them. We do this here
# rather than lazily because the registry is small and import-time side effects
# are easier to reason about than dynamic discovery.
from . import qwen3_moe  # noqa: F401 — registers itself

__all__ = [
    "MoEModelAdapter",
    "MoELayerHandle",
    "register_adapter",
    "get_adapter",
    "list_adapters",
]
