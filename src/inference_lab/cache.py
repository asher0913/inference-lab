from __future__ import annotations

import math
import time
from collections import OrderedDict
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
    ) -> None:
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self.threshold = threshold
        self.embedder = HashingEmbedder()
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

    @staticmethod
    def _similarity(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True)) / (
            math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)) or 1
        )

    def get(self, prompt: str) -> str | None:
        now = time.monotonic()
        vector = self.embedder.encode(prompt)
        best_key: str | None = None
        best_score = -1.0
        expired: list[str] = []
        for key, entry in self._entries.items():
            if entry.expires_at <= now:
                expired.append(key)
                continue
            score = self._similarity(vector, entry.vector)
            if score > best_score:
                best_key, best_score = key, score
        for key in expired:
            self._entries.pop(key, None)
        if best_key is not None and best_score >= self.threshold:
            entry = self._entries.pop(best_key)
            self._entries[best_key] = entry
            return entry.value
        return None

    def put(self, prompt: str, value: str) -> None:
        self._entries.pop(prompt, None)
        self._entries[prompt] = CacheEntry(
            prompt=prompt,
            vector=self.embedder.encode(prompt),
            value=value,
            expires_at=time.monotonic() + self.ttl_seconds,
        )
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
