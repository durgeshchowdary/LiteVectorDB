"""Persistence service for LiteVectorDB.

Handles loading and saving vectors, metadata, and index state.
Uses append-only patterns and memory-mapped arrays for efficiency.
"""
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from app.config import get_config
from app.services.wal import atomic_write_json

logger = logging.getLogger(__name__)


class PersistenceService:
    """Persistence layer for vectors and metadata."""
    
    def __init__(self, data_dir: Path = None):
        self.config = get_config()
        self.data_dir = data_dir or self.config.storage.data_dir
        self._ensure_data_dir()
        
        # In-memory storage
        self.vectors: Optional[np.ndarray] = None
        self.metadata: Dict[str, dict] = {}  # chunk_id -> metadata
        self.documents: Dict[str, dict] = {}  # doc_id -> document metadata
        self._next_vector_idx = 0
        
        # Track vector dimension
        self._vector_dim = self.config.vector.dimension
    
    def _ensure_data_dir(self) -> None:
        """Ensure data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def vectors_path(self) -> Path:
        return self.data_dir / self.config.storage.vectors_file
    
    @property
    def metadata_path(self) -> Path:
        return self.data_dir / self.config.storage.metadata_file
    
    @property
    def index_path(self) -> Path:
        return self.data_dir / self.config.storage.index_file
    
    def load(self) -> bool:
        """Load existing data from storage.
        
        Returns:
            True if data was loaded, False if no data exists
        """
        loaded_any = False
        
        # Load vectors
        if self.vectors_path.exists():
            try:
                self.vectors = np.load(self.vectors_path, mmap_mode='r')
                self._vector_dim = self.vectors.shape[1] if self.vectors.ndim > 1 else self.config.vector.dimension
                loaded_any = True
                logger.info(f"Loaded {len(self.vectors)} vectors from {self.vectors_path}")
            except Exception as e:
                logger.warning(f"Failed to load vectors: {e}")
                self.vectors = None
        
        # Load metadata
        if self.metadata_path.exists():
            try:
                self.metadata = {}
                self.documents = {}
                with open(self.metadata_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            chunk_id = entry.get('chunk_id')
                            if chunk_id:
                                self.metadata[chunk_id] = entry
                                
                                # Track unique documents
                                doc_id = entry.get('doc_id')
                                if doc_id and doc_id not in self.documents:
                                    self.documents[doc_id] = {
                                        'doc_id': doc_id,
                                        'text_length': entry.get('text_length', 0),
                                        'chunk_count': 0
                                    }
                                if doc_id:
                                    self.documents[doc_id]['chunk_count'] = \
                                        self.documents[doc_id].get('chunk_count', 0) + 1
                                self._next_vector_idx = max(
                                    self._next_vector_idx,
                                    int(entry.get("vector_idx", -1)) + 1
                                )
                loaded_any = True
                logger.info(f"Loaded {len(self.metadata)} metadata entries")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self.metadata = {}
                self.documents = {}
        
        return loaded_any
    
    def save_vector(self, vector: np.ndarray) -> int:
        """Append a vector to storage.
        
        Args:
            vector: Vector to save
            
        Returns:
            Index of the saved vector
        """
        # Convert to float16 if compression enabled
        if self.config.vector.use_float16 and self.config.vector.compression_enabled:
            vector = vector.astype(np.float16)
        else:
            vector = vector.astype(np.float32)
        
        if self.vectors is None:
            # First vector - initialize array
            self.vectors = vector.reshape(1, -1)
            index = 0
        else:
            # Append to existing array
            index = len(self.vectors)
            self.vectors = np.vstack([self.vectors, vector])
        
        return index
    
    def save_vectors_batch(self, vectors: np.ndarray) -> int:
        """Append multiple vectors to storage.
        
        Args:
            vectors: Array of vectors to save
            
        Returns:
            Starting index of the saved vectors
        """
        # Convert to float16 if compression enabled
        if self.config.vector.use_float16 and self.config.vector.compression_enabled:
            vectors = vectors.astype(np.float16)
        else:
            vectors = vectors.astype(np.float32)
        
        if self.vectors is None:
            self.vectors = vectors
            index = 0
        else:
            index = len(self.vectors)
            self.vectors = np.vstack([self.vectors, vectors])
        
        return index
    
    def save_metadata(self, chunk_id: str, metadata: dict) -> None:
        """Save metadata for a chunk."""
        self.metadata[chunk_id] = metadata
    
    def save_metadata_batch(self, entries: List[dict]) -> None:
        """Save metadata for multiple chunks (append to JSONL)."""
        with open(self.metadata_path, 'a') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
                chunk_id = entry.get('chunk_id')
                doc_id = entry.get('doc_id')
                if chunk_id:
                    self.metadata[chunk_id] = entry
                if doc_id and doc_id not in self.documents:
                    self.documents[doc_id] = {
                        'doc_id': doc_id,
                        'text_length': entry.get('text_length', 0),
                        'chunk_count': 0
                    }
                if doc_id:
                    self.documents[doc_id]['chunk_count'] = \
                        self.documents[doc_id].get('chunk_count', 0) + 1
                self._next_vector_idx = max(
                    self._next_vector_idx,
                    int(entry.get("vector_idx", -1)) + 1
                )
            f.flush()
            os.fsync(f.fileno())

    def reserve_vector_indices(self, count: int) -> int:
        """Reserve logical vector indexes for backward-compatible metadata."""
        start = self._next_vector_idx
        self._next_vector_idx += count
        return start

    def get_vector_refs(self, chunk_ids: List[str]) -> List[dict]:
        refs = []
        for chunk_id in chunk_ids:
            entry = self.metadata.get(chunk_id)
            if entry and "shard_id" in entry and "shard_offset" in entry:
                refs.append({"shard_id": entry["shard_id"], "shard_offset": entry["shard_offset"]})
        return refs

    def active_document_exists(self, doc_id: str, deleted_ids: set[str] | None = None) -> bool:
        deleted_ids = deleted_ids or set()
        return any(
            entry.get("doc_id") == doc_id and chunk_id not in deleted_ids
            for chunk_id, entry in self.metadata.items()
        )

    def get_chunks_for_document(self, doc_id: str, include_deleted: bool = True, deleted_ids: set[str] | None = None) -> List[str]:
        deleted_ids = deleted_ids or set()
        chunks = []
        for chunk_id, entry in self.metadata.items():
            if entry.get("doc_id") != doc_id:
                continue
            if not include_deleted and chunk_id in deleted_ids:
                continue
            chunks.append(chunk_id)
        return chunks
    
    def flush_vectors(self) -> None:
        """Write vectors to disk."""
        if self.vectors is not None and len(self.vectors) > 0:
            tmp = self.vectors_path.with_suffix(".npy.tmp")
            with open(tmp, "wb") as f:
                np.save(f, self.vectors)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.vectors_path)
            logger.info(f"Flushed {len(self.vectors)} vectors to disk")
    
    def get_vector(self, index: int) -> Optional[np.ndarray]:
        """Get a vector by index."""
        if self.vectors is None or index >= len(self.vectors):
            return None
        return self.vectors[index].astype(np.float32)
    
    def get_vectors(self, indices: List[int]) -> Optional[np.ndarray]:
        """Get multiple vectors by indices."""
        if self.vectors is None or not indices:
            return None
        
        valid_indices = [i for i in indices if i < len(self.vectors)]
        if not valid_indices:
            return None
        
        # Return as float32 for computation
        return self.vectors[valid_indices].astype(np.float32)
    
    def get_metadata(self, chunk_id: str) -> Optional[dict]:
        """Get metadata for a chunk."""
        return self.metadata.get(chunk_id)
    
    def get_chunk_ids(self) -> List[str]:
        """Get all chunk IDs."""
        return list(self.metadata.keys())
    
    def get_document_ids(self) -> List[str]:
        """Get all document IDs."""
        return list(self.documents.keys())
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists."""
        return doc_id in self.documents
    
    def get_stats(self) -> Dict:
        """Get storage statistics."""
        memory_bytes = 0
        if self.vectors is not None:
            memory_bytes = self.vectors.nbytes
        
        return {
            "documents": len(self.documents),
            "chunks": len(self.metadata),
            "vector_dim": self._vector_dim,
            "memory_bytes": memory_bytes,
            "vectors_loaded": self.vectors is not None
        }
    
    def clear(self) -> None:
        """Clear all data."""
        self.vectors = None
        self.metadata.clear()
        self.documents.clear()
        
        # Remove files
        for path in [self.vectors_path, self.metadata_path, self.index_path]:
            if path.exists():
                path.unlink()


# Global persistence service instance
_persistence_service = None


def get_persistence_service() -> PersistenceService:
    """Get the global persistence service instance."""
    global _persistence_service
    if _persistence_service is None:
        _persistence_service = PersistenceService()
    return _persistence_service
