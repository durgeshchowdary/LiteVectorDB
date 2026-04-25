"""Health check routes."""
from fastapi import APIRouter

from app.models import HealthResponse
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.
    
    Returns system status and statistics.
    """
    vector_store = get_vector_store()
    health = vector_store.health_check()
    return health