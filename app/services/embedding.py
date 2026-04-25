"""Embedding service facade."""
from typing import List

import numpy as np

from app.config import get_config
from app.services.embedding_provider import create_embedding_provider


class EmbeddingService:
    def __init__(self, dimension: int = None):
        self.config = get_config()
        self.dimension = dimension or self.config.vector.dimension
        self.provider = create_embedding_provider(self.config.embedding.provider, self.dimension)

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.provider.embed(texts)


_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
