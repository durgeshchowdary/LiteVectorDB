"""Data models for LiteVectorDB."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import uuid


class DocumentInput(BaseModel):
    """Input model for document ingestion."""
    doc_id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., min_length=1, description="Document text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    
    @field_validator('doc_id')
    @classmethod
    def validate_doc_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("doc_id cannot be empty")
        return v.strip()


class ChunkData(BaseModel):
    """Chunk data model."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    text: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorEntry(BaseModel):
    """Vector entry with metadata."""
    chunk_id: str
    doc_id: str
    vector: List[float]
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchInput(BaseModel):
    """Input model for search."""
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    metadata_filter: Dict[str, Any] = Field(default_factory=dict, description="Metadata filter")


class SearchResult(BaseModel):
    """Search result model."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    documents: int
    chunks: int
    vector_dim: int
    memory_usage_estimate: str


class IngestionResponse(BaseModel):
    """Response for document ingestion."""
    doc_id: str
    chunks_created: int
    ingestion_time_ms: float


class BatchDocumentInput(BaseModel):
    """Input model for batch document ingestion."""
    documents: List[DocumentInput] = Field(..., min_length=1)
    
    @field_validator('documents')
    @classmethod
    def validate_documents(cls, v: List[DocumentInput]) -> List[DocumentInput]:
        doc_ids = [d.doc_id for d in v]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("Duplicate doc_ids in batch")
        return v


class BatchIngestionResponse(BaseModel):
    """Response for batch document ingestion."""
    documents_processed: int
    total_chunks: int
    total_time_ms: float
    results: List[Dict[str, Any]] = Field(default_factory=list)
