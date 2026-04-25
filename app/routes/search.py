"""Search routes."""
from fastapi import APIRouter, HTTPException, status
from typing import List

from app.models import SearchInput, SearchResult
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=List[SearchResult])
async def search_documents(search_input: SearchInput):
    """Search for similar documents.
    
    Embeds the query text and finds the most similar document chunks.
    """
    try:
        vector_store = get_vector_store()
        results = vector_store.search(search_input)
        return results
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))