"""A semantic response cache with TTL, LRU eviction, a similarity threshold and an optional guard."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from .embedding import HashingEmbedder


@dataclass
class CacheEntry:
    prompt: str
    vector: list[float]
    value: str
    expires_at: float


class SemanticCache:
    def __init__(
        self,
        capacity: int = 512,
        ttl_seconds: float = 300,
        threshold: float = 0.96,
        embedder: HashingEmbedder | None = None,
        guard: Callable[[str, str], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self.threshold = threshold
        self.embedder = embedder or HashingEmbedder()
        self.guard = guard
        self.clock = clock
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

    @staticmethod
    def _similarity(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True))  # both are unit vectors

    def lookup(self, prompt: str) -> tuple[CacheEntry | None, float]:
        """Best live entry that passes the guard, and its similarity (hit or not)."""
        now = self.clock()
        vector = self.embedder.encode(prompt)
        best: CacheEntry | None = None
        best_score = -1.0
        for key in [k for k, e in self._entries.items() if e.expires_at <= now]:
            del self._entries[key]
        for entry in self._entries.values():
            if self.guard is not None and not self.guard(prompt, entry.prompt):
                continue
            score = self._similarity(vector, entry.vector)
            if score > best_score:
                best, best_score = entry, score
        return best, best_score

    def get(self, prompt: str) -> str | None:
        entry, score = self.lookup(prompt)
        if entry is None or score < self.threshold:
            return None
        self._entries.move_to_end(entry.prompt)
        return entry.value

    def put(self, prompt: str, value: str) -> None:
        self._entries.pop(prompt, None)
        self._entries[prompt] = CacheEntry(prompt, self.embedder.encode(prompt), value, self.clock() + self.ttl_seconds)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
