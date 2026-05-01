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
        """Append ``top2`` to the rolling window. Returns experts evicted."""
        ids = tuple(int(e) for e in top2)
        evicted = 0
        if len(self._entries) >= self.window:
            evicted += self._evict_one()
        self._entries.append(ids)
        for e in ids:
            self._counts[e] += 1
        while self._distinct_size() > self.cap and self._entries:
            evicted += self._evict_one()
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


