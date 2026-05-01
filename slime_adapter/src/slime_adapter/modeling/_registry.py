"""Tiny registry mapping adapter name → adapter class.

Adapters register themselves at import time via ``register_adapter``. User code
gets an instantiated adapter via ``get_adapter("qwen3_moe")``.
"""

from __future__ import annotations

from typing import Dict, List, Type

from ._base import MoEModelAdapter

_REGISTRY: Dict[str, Type[MoEModelAdapter]] = {}


def register_adapter(name: str, cls: Type[MoEModelAdapter]) -> Type[MoEModelAdapter]:
    """Register an adapter class. Usable as a decorator.

    Idempotent: re-registering the same (name, cls) pair is a no-op. Different
    cls under the same name raises.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"adapter name must be a non-empty string, got {name!r}")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Adapter name {name!r} is already registered to {existing.__name__}; "
            f"refusing to override with {cls.__name__}."
        )
    _REGISTRY[name] = cls
    return cls


def get_adapter(name: str) -> MoEModelAdapter:
    """Return a fresh instance of the registered adapter."""
    if name not in _REGISTRY:
        raise KeyError(
            f"No MoE adapter registered under {name!r}. "
            f"Available: {sorted(_REGISTRY)}. "
            f"Did you forget to `import slime_adapter.modeling.<your_arch>`?"
        )
    return _REGISTRY[name]()


def list_adapters() -> List[str]:
    """Return the sorted list of registered adapter names."""
    return sorted(_REGISTRY)
