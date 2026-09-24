"""Hashed bag-of-n-grams embeddings and guards for the semantic cache."""

from __future__ import annotations

import hashlib
import math
import re

_WORD = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_QUOTED = re.compile(r"(?:^|(?<=[\s:(]))(?:'([^']+)'|\"([^\"]+)\")(?=$|[\s,.:;?!)])")  # not apostrophes
NEGATIONS = {"not", "no", "never", "without", "don't", "doesn't", "isn't", "aren't", "cannot"}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _WORD.findall(text.lower()):
        if re.fullmatch(r"[一-鿿]+", raw):
            tokens.extend(raw[index : index + 2] for index in range(max(1, len(raw) - 1)))
        else:
            tokens.append(raw)
    return tokens


class HashingEmbedder:
    """Signed feature hashing of word n-grams (n = 1 .. ``ngram``), L2-normalised.

    With ``ngram=1`` word order is invisible: "convert 5 km to miles" and
    "convert 5 miles to km" get the same vector. Bigrams make order visible.
    """

    def __init__(self, dimensions: int = 256, ngram: int = 1) -> None:
        self.dimensions = dimensions
        self.ngram = ngram

    def features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        feats = list(tokens)
        for n in range(2, self.ngram + 1):
            feats += ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        return feats

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature in self.features(text):
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def literal_guard(query: str, cached: str) -> bool:
    """A cached answer may be reused only if numbers, quoted text and negation agree.

    These are the parts of a prompt that change the answer while barely
    changing its embedding.
    """

    def signature(text: str):
        lowered = text.lower()
        quoted = {a or b for a, b in _QUOTED.findall(lowered)}
        negated = bool(NEGATIONS & set(re.findall(r"[a-z']+", lowered)))
        return sorted(_NUMBER.findall(lowered)), quoted, negated

    return signature(query) == signature(cached)


class SentenceTransformerEmbedder:
    """A neural sentence embedder (``pip install -e '.[neural]'``); downloads the model on first use."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model
        self.model = SentenceTransformer(model)

    def encode(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()
