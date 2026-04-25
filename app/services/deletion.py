"""Deletion service with tombstones for LiteVectorDB."""
import json
from pathlib import Path
from typing import Set, Optional, List
from datetime import datetime
from dataclasses import dataclass

from app.config import get_config


@dataclass
class Tombstone:
    """Represents a deleted document/chunk."""
    chunk_id: str
    document_id: str
    deleted_at: str
    reason: str = "user_delete"


class DeletionService:
    """Manages soft deletes using tombstones."""
    
    def __init__(self):
        self.config = get_config()
        self._tombstones: dict[str, Tombstone] = {}
        self._load_tombstones()
    
    def _load_tombstones(self) -> None:
        """Load tombstones from disk."""
        path = self.config.storage.deleted_ids_path()
        if path.exists():
            try:
                with open(path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self._tombstones[data["chunk_id"]] = Tombstone(
                                chunk_id=data["chunk_id"],
                                document_id=data["document_id"],
                                deleted_at=data["deleted_at"],
                                reason=data.get("reason", "user_delete")
                            )
            except Exception as e:
                print(f"Warning: Failed to load tombstones: {e}")
    
    def _save_tombstone(self, tombstone: Tombstone) -> None:
        """Append a tombstone to the deleted IDs file."""
        path = self.config.storage.deleted_ids_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'a') as f:
            f.write(json.dumps({
                "chunk_id": tombstone.chunk_id,
                "document_id": tombstone.document_id,
                "deleted_at": tombstone.deleted_at,
                "reason": tombstone.reason
            }) + "\n")
    
    def is_deleted(self, chunk_id: str) -> bool:
        """Check if a chunk is deleted."""
        return chunk_id in self._tombstones
    
    def delete_chunk(self, chunk_id: str, document_id: str, reason: str = "user_delete") -> None:
        """Mark a chunk as deleted."""
        if chunk_id not in self._tombstones:
            tombstone = Tombstone(
                chunk_id=chunk_id,
                document_id=document_id,
                deleted_at=datetime.now().isoformat(),
                reason=reason
            )
            self._tombstones[chunk_id] = tombstone
            self._save_tombstone(tombstone)
    
    def delete_document(self, document_id: str, chunk_ids: List[str], reason: str = "user_delete") -> int:
        """Delete all chunks for a document."""
        deleted_count = 0
        for chunk_id in chunk_ids:
            if chunk_id not in self._tombstones:
                self.delete_chunk(chunk_id, document_id, reason)
                deleted_count += 1
        return deleted_count
    
    def restore_chunk(self, chunk_id: str) -> bool:
        """Restore a deleted chunk (remove tombstone)."""
        if chunk_id in self._tombstones:
            del self._tombstones[chunk_id]
            # Note: We don't remove from the file, but the in-memory state is restored
            # For true restoration, you'd need to rebuild the file
            return True
        return False
    
    def get_deleted_chunks(self) -> Set[str]:
        """Get all deleted chunk IDs."""
        return set(self._tombstones.keys())
    
    def get_tombstone(self, chunk_id: str) -> Optional[Tombstone]:
        """Get tombstone for a chunk."""
        return self._tombstones.get(chunk_id)
    
    def get_stats(self) -> dict:
        """Get deletion statistics."""
        return {
            "deleted_count": len(self._tombstones),
            "tombstones": [t.to_dict() for t in self._tombstones.values()]
        }
    
    def clear_all(self) -> None:
        """Clear all tombstones (use with caution)."""
        self._tombstones.clear()
        path = self.config.storage.deleted_ids_path()
        if path.exists():
            path.unlink()
    
    def purge_old_tombstones(self, days: int = 30) -> int:
        """Remove tombstones older than specified days."""
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        to_remove = []
        
        for chunk_id, tombstone in self._tombstones.items():
            deleted_at = datetime.fromisoformat(tombstone.deleted_at)
            if deleted_at < cutoff:
                to_remove.append(chunk_id)
        
        for chunk_id in to_remove:
            del self._tombstones[chunk_id]
        
        return len(to_remove)


# Add to_dict method to Tombstone
Tombstone.to_dict = lambda self: {
    "chunk_id": self.chunk_id,
    "document_id": self.document_id,
    "deleted_at": self.deleted_at,
    "reason": self.reason
}