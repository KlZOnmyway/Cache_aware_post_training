"""Per-request controller state for the SGLang rollout side.

Each generated rollout maintains its own:

  - ``LayerCache`` per MoE layer (16-token rolling window of used top-2).
  - ``CreditsTracker`` that resets at each new generated token.
  - List of per-(layer, token) records (switch, used_top2, new_top2, n_new, pressure).

The patched ``select_experts`` looks up the active ``RequestControllerState``
via SGLang's per-request context (we use a Python ``contextvars.ContextVar``).

The records are serialized into ``Sample.metadata['controller_records']`` at
the end of generation, then on the trainer side we read them via slime's
``RoutingReplay`` extension to replay the same routing/budget trajectory in
the teacher-forced training forward.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from slime_adapter.controller.cache_state import LayerCache
from slime_adapter.controller.credits import CreditsTracker

# Context var: per-request, set by SGLang's request handler at start, read by
# the patched select_experts. The patches themselves live in moe_select_patch.
CURRENT_STATE: contextvars.ContextVar[Optional["RequestControllerState"]] = (
    contextvars.ContextVar("slime_adapter_current_request_state", default=None)
)


@dataclass
class LayerRecord:
    layer_idx: int
    token_idx: int
    switch: int            # 0/1
    used_top2: Tuple[int, ...]
    new_top2: Tuple[int, ...]
    n_new: int
    pressure_in: float


@dataclass
class RequestControllerState:
    """One instance per in-flight inference request."""

    num_moe_layers: int
    cache_window: int = LayerCache.DEFAULT_WINDOW
    cache_cap: int = LayerCache.DEFAULT_CAP
    budget_fraction: float = 0.7

    caches: dict[int, LayerCache] = field(default_factory=dict)
    credits: CreditsTracker | None = None
    current_top2_per_layer: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    records: List[LayerRecord] = field(default_factory=list)

    current_token_idx: int = 0
    current_layer_idx: int = 0

    def __post_init__(self) -> None:
        if not self.caches:
            self.caches = {
                i: LayerCache(window=self.cache_window, cap=self.cache_cap)
                for i in range(self.num_moe_layers)
            }
        if self.credits is None:
            self.credits = CreditsTracker.from_config(
                num_moe_layers=self.num_moe_layers,
                fraction=self.budget_fraction,
            )

    # ---- token-boundary hook (called by SGLang generation loop) ----
    def on_new_token(self, token_idx: int) -> None:
        self.current_token_idx = int(token_idx)
        self.current_layer_idx = 0
        self.credits.reset_for_new_token()

    # ---- per-layer step (called by patched select_experts) ----
    def on_layer_entry(self, layer_idx: int) -> float:
        self.current_layer_idx = int(layer_idx)
        return self.credits.pressure

    def record_layer_step(
        self,
        *,
        switch: int,
        used_top2: Sequence[int],
        new_top2: Sequence[int],
        n_new: int,
        pressure_in: float,
    ) -> None:
        used_tup = tuple(int(e) for e in used_top2)
        new_tup = tuple(int(e) for e in new_top2)
        self.records.append(LayerRecord(
            layer_idx=self.current_layer_idx,
            token_idx=self.current_token_idx,
            switch=int(switch),
            used_top2=used_tup,
            new_top2=new_tup,
            n_new=int(n_new),
            pressure_in=float(pressure_in),
        ))
        self.current_top2_per_layer[self.current_layer_idx] = used_tup
        self.caches[self.current_layer_idx].push(used_tup)
        self.credits.charge(switch_signal=switch, n_new=n_new)

    # ---- read-back ----
    def serialize_for_sample_metadata(self) -> dict:
        """Compact form attached to Sample.metadata for the trainer."""
        return {
            "controller_records": [
                {
                    "layer": r.layer_idx,
                    "token": r.token_idx,
                    "switch": r.switch,
                    "used_top2": list(r.used_top2),
                    "new_top2": list(r.new_top2),
                    "n_new": r.n_new,
                    "pressure_in": r.pressure_in,
                }
                for r in self.records
            ],
            "controller_config": {
                "num_moe_layers": self.num_moe_layers,
                "cache_window": self.cache_window,
                "cache_cap": self.cache_cap,
                "budget_fraction": self.budget_fraction,
            },
        }


from typing import Sequence  # noqa: E402  — used in records signatures


def get_current_state() -> RequestControllerState | None:
    """Return the contextvar-active state, or None if no rollout is active."""
    return CURRENT_STATE.get()


def set_current_state(state: Optional["RequestControllerState"]) -> contextvars.Token:
    """Make ``state`` the active one. Returns a token to pass back to ``contextvars.reset``."""
    return CURRENT_STATE.set(state)


from typing import Optional  # noqa: E402
