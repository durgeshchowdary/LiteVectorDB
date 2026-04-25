"""FastAPI main application for LiteVectorDB."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_config
from app.routes import documents, search, health, admin

# Configure logging
config = get_config()
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format=config.logging.format
)

logger = logging.getLogger("litevectordb")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting LiteVectorDB...")
    logger.info(f"Configuration: dimension={config.vector.dimension}, "
                f"buckets={config.index.num_buckets}, "
                f"compression={config.vector.compression_enabled}")
    yield
    logger.info("Shutting down LiteVectorDB...")


# Create FastAPI app
app = FastAPI(
    title="LiteVectorDB",
    description="A lightweight vector database for low-resource environments",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(health.router)
app.include_router(admin.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "LiteVectorDB",
        "version": "0.1.0",
        "description": "A lightweight vector database for low-resource environments"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)