from __future__ import annotations

import hashlib
import math
import re

_WORD = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _WORD.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            tokens.extend(raw[index : index + 2] for index in range(max(1, len(raw) - 1)))
        else:
            tokens.append(raw)
    return tokens


class HashingEmbedder:
    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

