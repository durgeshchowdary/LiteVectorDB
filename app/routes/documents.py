"""Document ingestion routes."""
from fastapi import APIRouter, HTTPException, status
from typing import List

from app.models import (
    DocumentInput, 
    IngestionResponse, 
    BatchDocumentInput, 
    BatchIngestionResponse
)
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(doc_input: DocumentInput):
    """Ingest a single document.
    
    Chunks the text, generates embeddings, and stores vectors.
    """
    try:
        vector_store = get_vector_store()
        return vector_store.ingest_document(doc_input)
    except ValueError as e:
        if "MAX_TEXT_SIZE" in str(e):
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/batch", response_model=BatchIngestionResponse, status_code=status.HTTP_201_CREATED)
async def ingest_documents_batch(batch_input: BatchDocumentInput):
    """Ingest multiple documents in batch.
    
    More efficient for bulk ingestion.
    """
    try:
        vector_store = get_vector_store()
        return vector_store.ingest_documents_batch(batch_input.documents)
    except ValueError as e:
        if "MAX_BATCH_SIZE" in str(e) or "MAX_TEXT_SIZE" in str(e):
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its chunks.
    
    Uses soft delete (tombstones) - data is marked as deleted but not physically removed.
    """
    try:
        vector_store = get_vector_store()
        return vector_store.delete_document(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.put("/{doc_id}", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
async def update_document(doc_id: str, doc_input: DocumentInput):
    """Update a document.
    
    Deletes the old version and inserts the new one.
    """
    try:
        vector_store = get_vector_store()
        return vector_store.update_document(doc_id, doc_input)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
