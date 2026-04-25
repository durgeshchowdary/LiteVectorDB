"""Pluggable embedding providers."""
import hashlib
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an array shaped (len(texts), dimension)."""


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic local provider used by default."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def _hash_string(self, value: str) -> int:
        digest = hashlib.sha256(value.encode("utf-8")).digest()[:8]
        return int.from_bytes(digest, "big")

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        for i, word in enumerate(words):
            vector[self._hash_string(word) % self.dimension] += 1.0 / (1 + i * 0.1)
        lowered = text.lower()
        for n, weight in ((2, 0.3), (3, 0.2)):
            for i in range(max(0, len(lowered) - n + 1)):
                vector[self._hash_string(lowered[i:i + n]) % self.dimension] += weight
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self._embed_one(text) for text in texts]).astype(np.float32)


class StubExternalEmbeddingProvider(BaseEmbeddingProvider):
    """Future integration point; intentionally makes no external calls."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._fallback = LocalEmbeddingProvider(dimension)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._fallback.embed(texts)


def create_embedding_provider(name: str, dimension: int) -> BaseEmbeddingProvider:
    if name == "local":
        return LocalEmbeddingProvider(dimension)
    if name in {"external", "stub", "openai", "gemini", "sentence-transformers"}:
        return StubExternalEmbeddingProvider(dimension)
    raise ValueError(f"Unknown embedding provider '{name}'")
