"""Metrics and logging service for LiteVectorDB."""
import logging
import time
import json
import psutil
from typing import Dict, Any, Optional
from contextlib import contextmanager
from app.config import get_config


class MetricsService:
    """Metrics collection and logging service."""
    
    def __init__(self):
        self.config = get_config()
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format
        )
    
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger("litevectordb")
    
    def log_ingestion(self, doc_id: str, num_chunks: int, time_ms: float) -> None:
        """Log document ingestion metrics."""
        self.logger.info(
            f"Document ingested: doc_id={doc_id}, chunks={num_chunks}, time_ms={time_ms:.2f}"
        )

    def log_json(self, event: str, fields: Dict[str, Any]) -> None:
        """Emit a structured JSON log line."""
        payload = {"event": event, **fields}
        self.logger.info(json.dumps(payload, default=str, sort_keys=True))
    
    def log_search(self, query: str, top_k: int, time_ms: float, results_found: int) -> None:
        """Log search metrics."""
        self.logger.info(
            f"Search completed: query='{query[:50]}...', top_k={top_k}, "
            f"time_ms={time_ms:.2f}, results={results_found}"
        )
    
    def log_batch_ingestion(self, num_docs: int, num_chunks: int, time_ms: float) -> None:
        """Log batch ingestion metrics."""
        self.logger.info(
            f"Batch ingested: documents={num_docs}, chunks={num_chunks}, time_ms={time_ms:.2f}"
        )
    
    @contextmanager
    def measure_time(self, operation: str):
        """Context manager for measuring operation time."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.logger.debug(f"{operation} took {elapsed_ms:.2f}ms")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage estimate."""
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "rss_mb": mem_info.rss / (1024 * 1024),
                "vms_mb": mem_info.vms / (1024 * 1024),
            }
        except Exception:
            return {"rss_mb": 0, "vms_mb": 0}
    
    def get_memory_usage_str(self) -> str:
        """Get memory usage as formatted string."""
        mem = self.get_memory_usage()
        return f"RSS: {mem['rss_mb']:.1f}MB, VMS: {mem['vms_mb']:.1f}MB"
    
    def log_memory_usage(self, context: str = "") -> None:
        """Log current memory usage."""
        mem_str = self.get_memory_usage_str()
        self.logger.info(f"Memory usage{': ' + context if context else ''}: {mem_str}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "available_memory_mb": psutil.virtual_memory().available / (1024 * 1024),
        }
    
    def log_wal_operation(self, operation: str, entry_id: int) -> None:
        """Log WAL operation."""
        self.logger.debug(f"WAL operation: {operation}, entry_id={entry_id}")
    
    def log_deletion(self, chunk_id: str, document_id: str) -> None:
        """Log deletion operation."""
        self.logger.info(f"Deleted: chunk_id={chunk_id}, document_id={document_id}")
    
    def log_shard_assignment(self, chunk_id: str, shard_id: str) -> None:
        """Log shard assignment."""
        self.logger.debug(f"Shard assignment: chunk_id={chunk_id} -> shard_id={shard_id}")
    
    def log_lock_acquired(self, resource: str) -> None:
        """Log lock acquisition."""
        self.logger.debug(f"Lock acquired: resource={resource}")
    
    def log_lock_released(self, resource: str) -> None:
        """Log lock release."""
        self.logger.debug(f"Lock released: resource={resource}")
    
    def log_backpressure(self, reason: str, current_value: int, limit: int) -> None:
        """Log backpressure event."""
        self.logger.warning(f"Backpressure triggered: {reason}, current={current_value}, limit={limit}")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all statistics."""
        return {
            "memory": self.get_memory_usage(),
            "system": self.get_system_info(),
        }


# Global metrics service instance
_metrics_service = None


def get_metrics_service() -> MetricsService:
    """Get the global metrics service instance."""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service
