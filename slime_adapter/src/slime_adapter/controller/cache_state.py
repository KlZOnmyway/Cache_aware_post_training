"""Per-layer rolling-window cache of expert ids actually used at each token.

Hardware semantics:
  At each token the layer's MoE computation uses some set of experts. The cache
  tracks the union of the experts actually used in the last ``window`` tokens.
  When we want to switch to a fresh top2 whose experts aren't already in the
  cache, those experts must be loaded → counted as ``n_new`` cache misses.

This module is **pure Python**; it has no torch dependency and no notion of
training vs inference. The Megatron training side and the SGLang rollout side
both use the same class.

Quick API:

    cache = LayerCache(window=16, cap=30)
    n_new = cache.n_new(top2)              # without mutating
    cache.push(top2)                        # advance one token
    n, evicted = cache.n_new_and_push(top2) # combined
    cache.union                             # set[int] currently in cache
    cache.size                              # |union|
    cache.snapshot() / cache.restore(s)     # for reproducible replay
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence, Tuple


class LayerCache:
    __slots__ = ("window", "cap", "_entries", "_counts")

    DEFAULT_WINDOW = 16
    DEFAULT_CAP = 30

    def __init__(self, window: int = DEFAULT_WINDOW, cap: int = DEFAULT_CAP) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if cap < 1:
            raise ValueError(f"cap must be >= 1, got {cap}")
        self.window: int = int(window)
        self.cap: int = int(cap)
        self._entries: List[Tuple[int, ...]] = []
        self._counts: Counter = Counter()

    # ----- core ops -----
    def n_new(self, top2: Sequence[int]) -> int:
        """Number of experts in ``top2`` not currently in the cache."""
        return sum(1 for e in top2 if self._counts.get(int(e), 0) == 0)

    def push(self, top2: Sequence[int]) -> int:
        """Append ``top2`` to the rolling window. Returns experts evicted.

        Window-only eviction (cap dropped in v4 to match BatchedLayerCache).
        """
        ids = tuple(int(e) for e in top2)
        evicted = 0
        if len(self._entries) >= self.window:
            evicted += self._evict_one()
        self._entries.append(ids)
        for e in ids:
            self._counts[e] += 1
        return evicted

    def n_new_and_push(self, top2: Sequence[int]) -> Tuple[int, int]:
        n = self.n_new(top2)
        evicted = self.push(top2)
        return n, evicted

    # ----- read-only views -----
    @property
    def union(self) -> set[int]:
        return {e for e, c in self._counts.items() if c > 0}

    @property
    def size(self) -> int:
        return self._distinct_size()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, expert_id: int) -> bool:
        return self._counts.get(int(expert_id), 0) > 0

    def __repr__(self) -> str:
        return (
            f"LayerCache(window={self.window}, cap={self.cap}, "
            f"entries={len(self._entries)}, distinct={self.size})"
        )

    # ----- snapshot / restore -----
    def snapshot(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(self._entries)

    def restore(self, snap: Iterable[Sequence[int]]) -> None:
        self.reset()
        for entry in snap:
            self.push(entry)

    def reset(self) -> None:
        self._entries = []
        self._counts = Counter()

    # ----- internals -----
    def _distinct_size(self) -> int:
        return sum(1 for c in self._counts.values() if c > 0)

    def _evict_oldest(self) -> int:
        return self._evict_one()

    def _evict_one(self) -> int:
        if not self._entries:
            return 0
        old = self._entries.pop(0)
        evicted = 0
        for e in old:
            self._counts[e] -= 1
            if self._counts[e] <= 0:
                del self._counts[e]
                evicted += 1
        return evicted


# =====================================================================
# BatchedLayerCache — torch tensor implementation for the train forward
# =====================================================================

class BatchedLayerCache:
    """Per-layer rolling expert cache, batched across (B, T) on-device.

    Lazy lifecycle:

        cache = BatchedLayerCache(num_experts=128, window=16, cap=30)
        # before each train forward:
        cache.begin_batch(batch_size=B, device=hidden.device)
        # at each token position t (sequential over T inside the layer wrap):
        n_new_t = cache.n_new(used_top2_t)        # [B] long
        cache.push(used_top2_t)                   # advance the window

    State:
      count   : [B, num_experts] int64    occurrences in window
      history : list of [B, k]   int64    last <= window pushes (newest at end)

    Both window-eviction and cap-eviction are enforced exactly as in
    ``LayerCache`` (the rollout-side Python implementation), so train-time and
    rollout-time agree on what counts as ``n_new``.

    n_new is integer; nothing in this object is differentiable. Switching to
    BatchedLayerCache replaces the previous "shared-union-across-batch"
    approximation in ``moe_forward_patch._compute_n_new_batched`` with a
    proper per-(b, t) account.
    """

    DEFAULT_WINDOW = 16

    def __init__(
        self,
        num_experts: int,
        window: int = DEFAULT_WINDOW,
        cap: int = 0,
    ) -> None:
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.E = int(num_experts)
        self.window = int(window)
        self._count: "torch.Tensor | None" = None
        self._history: List["torch.Tensor"] = []
        self._batch_size: int = 0

    # ----- lifecycle -----

    def begin_batch(self, batch_size: int, *, device=None, dtype=None) -> None:
        """Allocate fresh state for a new forward. Must be called once per
        forward pass before any push/n_new."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._batch_size = int(batch_size)
        self._count = torch.zeros(
            self._batch_size, self.E,
            dtype=torch.int64, device=device,
        )
        self._history = []

    # ----- core ops -----

    def n_new(self, top_k: "torch.Tensor") -> "torch.Tensor":
        """``top_k: [B, k]`` (long) → ``[B]`` long count of missing experts."""
        if self._count is None:
            raise RuntimeError("call begin_batch(batch_size, ...) first")
        gathered = self._count.gather(1, top_k.long())            # [B, k]
        in_cache = gathered > 0
        return (~in_cache).sum(dim=-1).to(torch.int64)             # [B]

    def push(self, used_top_k: "torch.Tensor") -> None:
        """Advance the rolling window with one push per batch position.

        Window-only eviction (cap dropped in v4): the union size is bounded
        implicitly by ``window × k`` (≈ 32 for window=16, k=2). No CPU sync
        required.
        """
        if self._count is None:
            raise RuntimeError("call begin_batch(batch_size, ...) first")
        used = used_top_k.detach().long()                          # [B, k]
        # add new entry
        ones = torch.ones_like(used)
        self._count.scatter_add_(1, used, ones)
        self._history.append(used)
        # window-eviction (no cap pass — pure rolling window of last `window` entries)
        while len(self._history) > self.window:
            old = self._history.pop(0)
            self._count.scatter_add_(1, old, -torch.ones_like(old))

    def n_new_and_push(self, used_top_k: "torch.Tensor") -> "torch.Tensor":
        """Combined: returns ``[B]`` n_new before the push."""
        n = self.n_new(used_top_k)
        self.push(used_top_k)
        return n

    # ----- read-only -----

    @property
    def count(self) -> "torch.Tensor":
        """``[B, num_experts]`` int64; >0 means expert is currently in cache."""
        if self._count is None:
            raise RuntimeError("call begin_batch first")
        return self._count

    def union_mask(self) -> "torch.Tensor":
        """``[B, num_experts]`` bool union view."""
        if self._count is None:
            raise RuntimeError("call begin_batch first")
        return self._count > 0

    def size_per_batch(self) -> "torch.Tensor":
        """``[B]`` int64: distinct expert count per batch position."""
        return self.union_mask().sum(dim=-1).long()

    def __repr__(self) -> str:
        if self._count is None:
            return f"BatchedLayerCache(uninitialized, window={self.window})"
        return (
            f"BatchedLayerCache(B={self._batch_size}, E={self.E}, "
            f"window={self.window}, "
            f"history_len={len(self._history)}, "
            f"distinct_max={int((self._count > 0).sum(dim=-1).max())})"
        )


# =====================================================================
# Module-level torch import (lazy — keeps the pure-Python LayerCache
# importable in environments without torch).
# =====================================================================

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover - torch is a hard dep at runtime
    torch = None  # type: ignore


