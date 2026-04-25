"""Index service for LiteVectorDB.

Implements bucket-based partitioning for efficient approximate nearest neighbor search.
Vectors are assigned to buckets based on their hash, enabling targeted search.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional
from app.config import get_config


class IndexService:
    """Bucket-based indexing for vector partitioning."""
    
    def __init__(self, num_buckets: int = None):
        self.config = get_config()
        self.num_buckets = num_buckets or self.config.index.num_buckets
        self.buckets: Dict[int, List[str]] = {}  # bucket_id -> list of chunk_ids
        self.chunk_to_bucket: Dict[str, int] = {}  # chunk_id -> bucket_id
        self.index_type = self.config.index.index_type
        self.centroids: Optional[np.ndarray] = None
        self._initialize_buckets()
    
    def _initialize_buckets(self) -> None:
        """Initialize empty buckets."""
        for i in range(self.num_buckets):
            self.buckets[i] = []
    
    def get_bucket_for_vector(self, vector: np.ndarray) -> int:
        """Determine bucket for a vector based on its hash.
        
        Uses a subset of vector dimensions to compute bucket assignment,
        providing rough clustering without expensive k-means.
        """
        # Use hash of vector components for deterministic bucket assignment
        # Take first 8 dimensions and hash them
        key_components = vector[:min(8, len(vector))]
        hash_value = hash(tuple(key_components.round(4)))
        bucket_id = abs(hash_value) % self.num_buckets
        return bucket_id
    
    def get_buckets_for_search(self, query_vector: np.ndarray, search_buckets: int = None) -> List[int]:
        """Get buckets to search for a query vector.
        
        Searches the primary bucket plus nearby buckets for better recall.
        """
        search_buckets = search_buckets or 3  # Search primary + 2 neighbors
        if self.centroids is not None and len(self.centroids):
            query_norm = np.linalg.norm(query_vector) or 1
            centroid_norms = np.linalg.norm(self.centroids, axis=1)
            scores = np.dot(self.centroids, query_vector) / np.where(centroid_norms * query_norm == 0, 1, centroid_norms * query_norm)
            probes = min(self.config.index.centroid_probe_count, len(scores))
            return [int(i) for i in np.argsort(scores)[::-1][:probes]]
        primary_bucket = self.get_bucket_for_vector(query_vector)
        
        buckets = [primary_bucket]
        for i in range(1, search_buckets):
            buckets.append((primary_bucket + i) % self.num_buckets)
            buckets.append((primary_bucket - i) % self.num_buckets)
        
        return list(set(buckets))
    
    def add_chunk_to_bucket(self, chunk_id: str, vector: np.ndarray) -> int:
        """Add a chunk to the appropriate bucket.
        
        Args:
            chunk_id: ID of the chunk
            vector: Vector embedding
            
        Returns:
            Bucket ID where chunk was added
        """
        bucket_id = self.get_bucket_for_vector(vector)
        self.buckets[bucket_id].append(chunk_id)
        self.chunk_to_bucket[chunk_id] = bucket_id
        return bucket_id

    def build_index(self, vectors: np.ndarray, chunk_ids: List[str]) -> Dict[str, object]:
        """Build a lightweight ANN index. Uses centroid buckets when possible."""
        self.clear()
        if vectors is None or len(vectors) == 0:
            return {"indexed_chunks": 0, "index_type": self.index_type}
        vectors = vectors.astype(np.float32)
        if self.index_type == "centroid" and len(vectors) >= 2:
            k = min(self.config.index.num_buckets, max(1, int(np.sqrt(len(vectors)))))
            self.num_buckets = k
            self.buckets = {i: [] for i in range(k)}
            self.centroids = self._kmeans(vectors, k)
            assignments = self._nearest_centroids(vectors, self.centroids)
            for chunk_id, bucket_id in zip(chunk_ids, assignments):
                bucket = int(bucket_id)
                self.buckets[bucket].append(chunk_id)
                self.chunk_to_bucket[chunk_id] = bucket
        else:
            for chunk_id, vector in zip(chunk_ids, vectors):
                self.add_chunk_to_bucket(chunk_id, vector)
        return {"indexed_chunks": len(chunk_ids), "index_type": self.get_stats()["index_type"], "buckets": self.num_buckets}

    def _nearest_centroids(self, vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True).T
        scores = vectors @ centroids.T / np.where(vector_norms * centroid_norms == 0, 1, vector_norms * centroid_norms)
        return np.argmax(scores, axis=1)

    def _kmeans(self, vectors: np.ndarray, k: int, iterations: int = 8) -> np.ndarray:
        if len(vectors) <= k:
            return vectors.copy()
        sample_idx = np.linspace(0, len(vectors) - 1, k, dtype=int)
        centroids = vectors[sample_idx].copy()
        for _ in range(iterations):
            assignments = self._nearest_centroids(vectors, centroids)
            for bucket in range(k):
                members = vectors[assignments == bucket]
                if len(members):
                    centroids[bucket] = members.mean(axis=0)
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroids = centroids / np.where(norms == 0, 1, norms)
        return centroids
    
    def remove_chunk_from_bucket(self, chunk_id: str) -> Optional[int]:
        """Remove a chunk from its bucket.
        
        Args:
            chunk_id: ID of the chunk to remove
            
        Returns:
            Bucket ID where chunk was, or None if not found
        """
        bucket_id = self.chunk_to_bucket.get(chunk_id)
        if bucket_id is not None and chunk_id in self.buckets[bucket_id]:
            self.buckets[bucket_id].remove(chunk_id)
            del self.chunk_to_bucket[chunk_id]
        return bucket_id
    
    def get_chunks_in_buckets(self, bucket_ids: List[int]) -> Set[str]:
        """Get all chunk IDs in the specified buckets."""
        chunks = set()
        for bucket_id in bucket_ids:
            if bucket_id in self.buckets:
                chunks.update(self.buckets[bucket_id])
        return chunks
    
    def get_all_chunk_ids(self) -> Set[str]:
        """Get all chunk IDs in the index."""
        return set(self.chunk_to_bucket.keys())
    
    def get_bucket_stats(self) -> Dict[int, int]:
        """Get statistics about bucket distribution."""
        return {bid: len(chunks) for bid, chunks in self.buckets.items()}
    
    def save_index(self, path: Path) -> None:
        """Save index state to file."""
        index_data = {
            "num_buckets": self.num_buckets,
            "index_type": self.index_type,
            "buckets": self.buckets,
            "chunk_to_bucket": self.chunk_to_bucket,
            "centroids": self.centroids.tolist() if self.centroids is not None else None,
        }
        with open(path, 'w') as f:
            json.dump(index_data, f)
    
    def load_index(self, path: Path) -> bool:
        """Load index state from file.
        
        Returns:
            True if index was loaded successfully, False if file doesn't exist
        """
        if not path.exists():
            return False
        
        try:
            with open(path, 'r') as f:
                index_data = json.load(f)
            
            self.num_buckets = index_data.get("num_buckets", self.num_buckets)
            self.index_type = index_data.get("index_type", self.index_type)
            self.buckets = {int(k): v for k, v in index_data.get("buckets", {}).items()}
            self.chunk_to_bucket = index_data.get("chunk_to_bucket", {})
            centroids = index_data.get("centroids")
            self.centroids = np.array(centroids, dtype=np.float32) if centroids else None
            
            # Ensure all buckets exist
            for i in range(self.num_buckets):
                if i not in self.buckets:
                    self.buckets[i] = []
            
            return True
        except (json.JSONDecodeError, IOError) as e:
            # Corrupted index file - start fresh
            return False
    
    def clear(self) -> None:
        """Clear all index data."""
        self._initialize_buckets()
        self.chunk_to_bucket.clear()
        self.centroids = None

    def get_stats(self) -> Dict[str, object]:
        return {
            "index_type": "centroid" if self.centroids is not None else "bucket",
            "num_buckets": self.num_buckets,
            "indexed_chunks": len(self.chunk_to_bucket),
            "non_empty_buckets": sum(1 for chunks in self.buckets.values() if chunks),
        }


# Global index service instance
_index_service = None


def get_index_service() -> IndexService:
    """Get the global index service instance."""
    global _index_service
    if _index_service is None:
        _index_service = IndexService()
    return _index_service
