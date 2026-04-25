"""Configuration system for LiteVectorDB."""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    """Storage configuration."""
    data_dir: Path = Field(default_factory=lambda: Path("app/data"))
    shards_dir: Path = Field(default_factory=lambda: Path("app/data/shards"))
    vectors_file: str = "vectors.npy"
    metadata_file: str = "metadata.jsonl"
    index_file: str = "index.json"
    shard_index_file: str = "shard_index.json"
    deleted_ids_file: str = "deleted_ids.jsonl"
    wal_file: str = "wal.jsonl"
    
    def vectors_path(self) -> Path:
        return self.data_dir / self.vectors_file
    
    def metadata_path(self) -> Path:
        return self.data_dir / self.metadata_file
    
    def index_path(self) -> Path:
        return self.data_dir / self.index_file
    
    def shard_index_path(self) -> Path:
        return self.data_dir / self.shard_index_file
    
    def deleted_ids_path(self) -> Path:
        return self.data_dir / self.deleted_ids_file
    
    def wal_path(self) -> Path:
        return self.data_dir / self.wal_file
    
    def shards_path(self) -> Path:
        return self.shards_dir


class VectorConfig(BaseModel):
    """Vector configuration."""
    dimension: int = 384
    compression_enabled: bool = True
    use_float16: bool = True


class IndexConfig(BaseModel):
    """Index configuration."""
    num_buckets: int = 100
    bucket_threshold: int = 1000
    index_type: str = "centroid"
    centroid_probe_count: int = 3


class SearchConfig(BaseModel):
    """Search configuration."""
    default_top_k: int = 5
    max_top_k: int = 100


class ChunkingConfig(BaseModel):
    """Chunking configuration."""
    chunk_size: int = 512
    overlap: int = 50


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    json_logs: bool = False


class ShardingConfig(BaseModel):
    """Sharding configuration."""
    enabled: bool = True
    shard_count: int = 4
    lazy_load: bool = True


class ConcurrencyConfig(BaseModel):
    """Concurrency configuration."""
    use_file_locks: bool = True
    lock_timeout_seconds: float = 30.0
    single_writer_mode: bool = True


class BackpressureConfig(BaseModel):
    """Backpressure configuration."""
    max_batch_size: int = 100
    max_text_size: int = 100000
    max_pending_writes: int = 1000


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""
    provider: str = "local"  # "local" or "external"


class Config(BaseModel):
    """Main configuration for LiteVectorDB."""
    storage: StorageConfig = Field(default_factory=StorageConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    sharding: ShardingConfig = Field(default_factory=ShardingConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    backpressure: BackpressureConfig = Field(default_factory=BackpressureConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            storage=StorageConfig(
                data_dir=Path(os.getenv("LVDB_DATA_DIR", "app/data")),
                shards_dir=Path(os.getenv("LVDB_SHARDS_DIR", "app/data/shards")),
            ),
            vector=VectorConfig(
                dimension=int(os.getenv("LVDB_DIMENSION", "384")),
                compression_enabled=os.getenv("LVDB_COMPRESSION", "true").lower() == "true",
                use_float16=os.getenv("LVDB_FLOAT16", "true").lower() == "true",
            ),
            index=IndexConfig(
                num_buckets=int(os.getenv("LVDB_BUCKETS", "100")),
                bucket_threshold=int(os.getenv("LVDB_BUCKET_THRESHOLD", "1000")),
                index_type=os.getenv("LVDB_INDEX_TYPE", "centroid"),
                centroid_probe_count=int(os.getenv("LVDB_CENTROID_PROBES", "3")),
            ),
            search=SearchConfig(
                default_top_k=int(os.getenv("LVDB_DEFAULT_TOP_K", "5")),
                max_top_k=int(os.getenv("LVDB_MAX_TOP_K", "100")),
            ),
            chunking=ChunkingConfig(
                chunk_size=int(os.getenv("LVDB_CHUNK_SIZE", "512")),
                overlap=int(os.getenv("LVDB_OVERLAP", "50")),
            ),
            sharding=ShardingConfig(
                enabled=os.getenv("LVDB_SHARDING", "true").lower() == "true",
                shard_count=int(os.getenv("LVDB_SHARD_COUNT", "4")),
                lazy_load=os.getenv("LVDB_LAZY_LOAD_SHARDS", "true").lower() == "true",
            ),
            concurrency=ConcurrencyConfig(
                use_file_locks=os.getenv("LVDB_FILE_LOCKS", "true").lower() == "true",
                lock_timeout_seconds=float(os.getenv("LVDB_LOCK_TIMEOUT", "30")),
                single_writer_mode=os.getenv("LVDB_SINGLE_WRITER", "true").lower() == "true",
            ),
            backpressure=BackpressureConfig(
                max_batch_size=int(os.getenv("LVDB_MAX_BATCH_SIZE", "100")),
                max_text_size=int(os.getenv("LVDB_MAX_TEXT_SIZE", "100000")),
                max_pending_writes=int(os.getenv("LVDB_MAX_PENDING_WRITES", "1000")),
            ),
            embedding=EmbeddingConfig(
                provider=os.getenv("LVDB_EMBEDDING_PROVIDER", "local"),
            ),
        )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
