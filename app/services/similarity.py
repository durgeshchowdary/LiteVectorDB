"""Similarity computation service for LiteVectorDB."""
import numpy as np
from typing import List, Tuple


class SimilarityService:
    """Cosine similarity computation service optimized for batch operations."""
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (1.0 = identical, 0.0 = orthogonal)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    @staticmethod
    def cosine_similarity_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a query vector and multiple vectors.
        
        Args:
            query: Query vector of shape (dimension,)
            vectors: Matrix of vectors of shape (n, dimension)
            
        Returns:
            Array of similarity scores of shape (n,)
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = vectors / norms
        
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(len(vectors))
        
        query_normalized = query / query_norm
        
        # Compute dot products (which equals cosine similarity for normalized vectors)
        similarities = np.dot(normalized, query_normalized)
        
        return similarities
    
    @staticmethod
    def top_k_indices(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-k indices and scores.
        
        Args:
            scores: Array of similarity scores
            k: Number of top results to return
            
        Returns:
            Tuple of (indices, scores) for top-k results
        """
        if k >= len(scores):
            # Return all sorted by score descending
            sorted_indices = np.argsort(scores)[::-1]
            return sorted_indices, scores[sorted_indices]
        
        # Use argpartition for O(n) partial sort, then sort the top-k
        top_k_indices = np.argpartition(scores, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
        
        return top_k_indices, scores[top_k_indices]
    
    @staticmethod
    def compute_similarities_and_rank(
        query: np.ndarray, 
        vectors: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute similarities and return top-k ranked results.
        
        Args:
            query: Query vector of shape (dimension,)
            vectors: Matrix of vectors of shape (n, dimension)
            k: Number of top results
            
        Returns:
            Tuple of (top_k_indices, top_k_scores)
        """
        scores = SimilarityService.cosine_similarity_batch(query, vectors)
        return SimilarityService.top_k_indices(scores, k)


# Global similarity service instance
_similarity_service = None


def get_similarity_service() -> SimilarityService:
    """Get the global similarity service instance."""
    global _similarity_service
    if _similarity_service is None:
        _similarity_service = SimilarityService()
    return _similarity_service