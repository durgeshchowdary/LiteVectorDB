"""Text chunking service for LiteVectorDB."""
import uuid
from typing import List
from app.config import get_config
from app.models import ChunkData


class ChunkerService:
    """Text chunking service with sliding window approach."""
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.config = get_config()
        self.chunk_size = chunk_size or self.config.chunking.chunk_size
        self.overlap = overlap or self.config.chunking.overlap
    
    def chunk_text(self, text: str, doc_id: str, metadata: dict = None) -> List[ChunkData]:
        """Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            doc_id: Document ID to associate with chunks
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of ChunkData objects
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Use character-based chunking for better preservation of word boundaries
        text_length = len(text)
        start_idx = 0
        chunk_count = 0
        
        while start_idx < text_length:
            end_idx = min(start_idx + self.chunk_size, text_length)
            
            # Try to break at word boundary if not at end
            if end_idx < text_length:
                # Look for space within last 50 chars
                last_space = text.rfind(' ', start_idx, end_idx + 50)
                if last_space > start_idx:
                    end_idx = last_space
            
            chunk_text = text[start_idx:end_idx].strip()
            
            if chunk_text:  # Only create chunk if there's content
                chunk = ChunkData(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    metadata=metadata.copy()
                )
                chunks.append(chunk)
                chunk_count += 1
            
            # Move start position with overlap
            start_idx = end_idx - self.overlap
            
            # Prevent infinite loop if overlap >= chunk_size
            if start_idx <= chunks[-1].start_idx if chunks else start_idx <= 0:
                start_idx = end_idx
            
            # Safety limit
            if chunk_count > 10000:
                break
        
        return chunks
    
    def chunk_batch(self, texts: List[tuple], metadata: dict = None) -> List[ChunkData]:
        """Chunk multiple texts.
        
        Args:
            texts: List of (doc_id, text) tuples
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of all ChunkData objects
        """
        all_chunks = []
        for doc_id, text in texts:
            chunks = self.chunk_text(text, doc_id, metadata)
            all_chunks.extend(chunks)
        return all_chunks


# Global chunker service instance
_chunker_service = None


def get_chunker_service() -> ChunkerService:
    """Get the global chunker service instance."""
    global _chunker_service
    if _chunker_service is None:
        _chunker_service = ChunkerService()
    return _chunker_service