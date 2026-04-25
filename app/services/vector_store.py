"""Main orchestration layer for LiteVectorDB."""
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from app.config import get_config
from app.models import BatchIngestionResponse, DocumentInput, IngestionResponse, SearchInput, SearchResult
from app.services.chunker import get_chunker_service
from app.services.deletion import DeletionService
from app.services.embedding import get_embedding_service
from app.services.index import get_index_service
from app.services.locks import file_lock
from app.services.metrics import get_metrics_service
from app.services.persistence import get_persistence_service
from app.services.sharding import ShardingService
from app.services.similarity import get_similarity_service
from app.services.wal import WALService

logger = logging.getLogger(__name__)


class VectorStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.config = get_config()
        self.embedding_service = get_embedding_service()
        self.chunker_service = get_chunker_service()
        self.similarity_service = get_similarity_service()
        self.index_service = get_index_service()
        self.persistence_service = get_persistence_service()
        self.metrics_service = get_metrics_service()
        self.sharding_service = ShardingService()
        self.deletion_service = DeletionService()
        self.wal_service = WALService()

        self.chunk_id_to_index: Dict[str, int] = {}
        self.startup_summary: Dict[str, Any] = {}
        self.legacy_vectors_cache = None

        # Query embedding cache for repeated search queries.
        self.query_cache: Dict[str, np.ndarray] = {}

        # Hot vector cache: preloaded vectors keyed by filter signature.
        # Avoids repeated shard I/O on repeated searches with the same filter.
        self.hot_vector_cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}

        self._initialize()
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "VectorStore":
        return cls()

    def _initialize(self) -> None:
        loaded = self.persistence_service.load()
        wal_recovery = self.wal_service.recover()
        shard_validation = self.sharding_service.validate_shards()
        deleted = self.deletion_service.get_deleted_chunks()

        active_metadata = {
            chunk_id: entry
            for chunk_id, entry in self.persistence_service.metadata.items()
            if chunk_id not in deleted
        }

        self.sharding_service.rebuild_from_metadata(active_metadata)

        if loaded:
            self.rebuild_index(save=False)

        self.startup_summary = {
            "wal_recovery_events": wal_recovery,
            "shard_validation": shard_validation,
            "chunks_loaded": len(self.persistence_service.metadata),
        }
        self.metrics_service.log_json("startup_health", self.startup_summary)

    def _write_lock(self):
        return file_lock("writer")

    def _legacy_vector_path(self) -> str:
        return os.path.join("app", "data", "vectors.npy")

    def _load_legacy_vectors(self):
        if self.legacy_vectors_cache is not None:
            return self.legacy_vectors_cache

        legacy_path = self._legacy_vector_path()
        if not os.path.exists(legacy_path):
            return None

        try:
            self.legacy_vectors_cache = np.load(legacy_path, mmap_mode="r")
            logger.info("Loaded legacy vectors from %s", legacy_path)
            self.metrics_service.log_json("legacy_vectors_loaded", {
                "path": legacy_path,
                "count": int(len(self.legacy_vectors_cache)),
            })
            return self.legacy_vectors_cache
        except Exception as exc:
            logger.warning("Failed to load legacy vectors.npy: %s", exc)
            self.metrics_service.log_json("legacy_vectors_load_failed", {
                "path": legacy_path,
                "error": str(exc),
            })
            return None

    def _get_vectors_for_chunk_ids(self, chunk_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        if not chunk_ids:
            return np.empty((0, self.config.vector.dimension), dtype=np.float32), []

        sharded_ids = []
        sharded_refs = []
        legacy_ids = []
        legacy_indices = []

        for chunk_id in chunk_ids:
            metadata = self.persistence_service.get_metadata(chunk_id)
            if not metadata:
                continue

            if "shard_id" in metadata and "shard_offset" in metadata:
                sharded_ids.append(chunk_id)
                sharded_refs.append({
                    "shard_id": metadata["shard_id"],
                    "shard_offset": metadata["shard_offset"],
                })
            elif "vector_idx" in metadata:
                legacy_ids.append(chunk_id)
                legacy_indices.append(int(metadata["vector_idx"]))

        all_vectors = []
        all_ids = []

        if sharded_refs:
            try:
                shard_vectors = self.sharding_service.get_vectors_by_refs(sharded_refs)
                if shard_vectors is not None and len(shard_vectors) > 0:
                    all_vectors.append(np.asarray(shard_vectors, dtype=np.float32))
                    all_ids.extend(sharded_ids[:len(shard_vectors)])
            except Exception as exc:
                logger.warning("Failed loading shard vectors: %s", exc)

        if legacy_indices:
            legacy_vectors = self._load_legacy_vectors()
            if legacy_vectors is not None:
                valid_vectors = []
                valid_ids = []

                for chunk_id, vector_idx in zip(legacy_ids, legacy_indices):
                    if 0 <= vector_idx < len(legacy_vectors):
                        valid_vectors.append(np.asarray(legacy_vectors[vector_idx], dtype=np.float32))
                        valid_ids.append(chunk_id)

                if valid_vectors:
                    all_vectors.append(np.vstack(valid_vectors).astype(np.float32))
                    all_ids.extend(valid_ids)

        if not all_vectors:
            return np.empty((0, self.config.vector.dimension), dtype=np.float32), []

        return np.vstack(all_vectors).astype(np.float32), all_ids

    def _validate_doc(self, doc_input: DocumentInput) -> None:
        if len(doc_input.text) > self.config.backpressure.max_text_size:
            raise ValueError(
                f"Text size {len(doc_input.text)} exceeds MAX_TEXT_SIZE={self.config.backpressure.max_text_size}"
            )

    def ingest_document(self, doc_input: DocumentInput) -> IngestionResponse:
        with self._write_lock():
            return self._ingest_document_locked(doc_input, operation="insert", update_index=True)

    def _ingest_document_locked(
        self,
        doc_input: DocumentInput,
        operation: str,
        update_index: bool = True,
    ) -> IngestionResponse:
        start = time.perf_counter()
        self._validate_doc(doc_input)

        deleted = self.deletion_service.get_deleted_chunks()
        if self.persistence_service.active_document_exists(doc_input.doc_id, deleted):
            raise ValueError(f"Document with doc_id '{doc_input.doc_id}' already exists")

        tx_id = self.wal_service.begin(operation, doc_input.doc_id, {"text_length": len(doc_input.text)})

        try:
            chunks = self.chunker_service.chunk_text(doc_input.text, doc_input.doc_id, doc_input.metadata)
            if not chunks:
                raise ValueError("No chunks created from document text")

            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_service.embed_batch(texts)

            if self.config.vector.use_float16 and self.config.vector.compression_enabled:
                stored_vectors = embeddings.astype(np.float16)
            else:
                stored_vectors = embeddings.astype(np.float32)

            chunk_ids = [chunk.chunk_id for chunk in chunks]
            shard_refs = self.sharding_service.append_vectors(doc_input.doc_id, chunk_ids, stored_vectors)
            start_idx = self.persistence_service.reserve_vector_indices(len(chunks))

            entries = []
            for i, chunk in enumerate(chunks):
                ref = shard_refs[i]
                entry = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "text_length": len(chunk.text),
                    "start_idx": chunk.start_idx,
                    "end_idx": chunk.end_idx,
                    "vector_idx": start_idx + i,
                    "shard_id": ref["shard_id"],
                    "shard_offset": ref["shard_offset"],
                    "metadata": chunk.metadata,
                    "created_at": time.time(),
                }
                entries.append(entry)
                self.chunk_id_to_index[chunk.chunk_id] = start_idx + i
                self.index_service.add_chunk_to_bucket(chunk.chunk_id, embeddings[i])

            self.persistence_service.save_metadata_batch(entries)
            self.hot_vector_cache.clear()  # Invalidate cache on new ingestion

            if update_index:
                self.rebuild_index(save=True)
            else:
                self.index_service.save_index(self.persistence_service.index_path)

            elapsed = (time.perf_counter() - start) * 1000

            self.wal_service.commit(tx_id, {
                "chunks": len(chunks),
                "shard_id": shard_refs[0]["shard_id"],
            })

            self.metrics_service.log_ingestion(doc_input.doc_id, len(chunks), elapsed)
            self.metrics_service.log_json("ingest", {
                "doc_id": doc_input.doc_id,
                "ingestion_time_ms": elapsed,
                "chunks": len(chunks),
                "shard_used": shard_refs[0]["shard_id"],
            })

            return IngestionResponse(
                doc_id=doc_input.doc_id,
                chunks_created=len(chunks),
                ingestion_time_ms=elapsed,
            )

        except Exception as exc:
            self.wal_service.fail(tx_id, {"error": str(exc)})
            raise

    def ingest_documents_batch(self, documents: List[DocumentInput]) -> BatchIngestionResponse:
        if len(documents) > self.config.backpressure.max_batch_size:
            raise ValueError(
                f"Batch size {len(documents)} exceeds MAX_BATCH_SIZE={self.config.backpressure.max_batch_size}"
            )

        ids = [doc.doc_id for doc in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate doc_ids in batch")

        start = time.perf_counter()
        results = []
        total_chunks = 0

        with self._write_lock():
            for doc in documents:
                try:
                    response = self._ingest_document_locked(
                        doc,
                        operation="batch_insert",
                        update_index=False,
                    )
                    results.append({
                        "doc_id": doc.doc_id,
                        "status": "success",
                        "chunks": response.chunks_created,
                    })
                    total_chunks += response.chunks_created
                except Exception as exc:
                    results.append({
                        "doc_id": doc.doc_id,
                        "status": "failed",
                        "error": str(exc),
                    })

            self.hot_vector_cache.clear()  # Invalidate cache after batch ingestion
            self.rebuild_index(save=True)

        elapsed = (time.perf_counter() - start) * 1000
        self.metrics_service.log_batch_ingestion(len(documents), total_chunks, elapsed)

        return BatchIngestionResponse(
            documents_processed=len(documents),
            total_chunks=total_chunks,
            total_time_ms=elapsed,
            results=results,
        )

    def search(self, search_input: SearchInput) -> List[SearchResult]:
        start = time.perf_counter()

        cache_key = search_input.query.strip().lower()
        if cache_key in self.query_cache:
            query_vector = self.query_cache[cache_key]
        else:
            query_vector = self.embedding_service.embed_text(search_input.query)
            self.query_cache[cache_key] = query_vector

        if search_input.metadata_filter:
            candidate_chunk_ids = list(self.index_service.get_all_chunk_ids())
        else:
            bucket_ids = self.index_service.get_buckets_for_search(query_vector)
            candidate_chunk_ids = list(self.index_service.get_chunks_in_buckets(bucket_ids))

            if not candidate_chunk_ids:
                candidate_chunk_ids = list(self.index_service.get_all_chunk_ids())

        deleted = self.deletion_service.get_deleted_chunks()
        filtered_ids = []

        for chunk_id in candidate_chunk_ids:
            if chunk_id in deleted:
                continue

            metadata = self.persistence_service.get_metadata(chunk_id)
            if not metadata:
                continue

            if search_input.metadata_filter and not self._matches_filter(
                metadata.get("metadata", {}),
                search_input.metadata_filter,
            ):
                continue

            filtered_ids.append(chunk_id)

        if not filtered_ids:
            return []

        # Hot vector cache: reuse loaded vectors for repeated filter signatures
        # to avoid shard I/O on every search request.
        hot_key = "all" if not search_input.metadata_filter else str(search_input.metadata_filter)

        if hot_key in self.hot_vector_cache:
            vectors, aligned_ids = self.hot_vector_cache[hot_key]
        else:
            vectors, aligned_ids = self._get_vectors_for_chunk_ids(filtered_ids)
            self.hot_vector_cache[hot_key] = (vectors, aligned_ids)

        if vectors is None or len(vectors) == 0 or not aligned_ids:
            return []

        scores = self.similarity_service.cosine_similarity_batch(query_vector, vectors)
        top_k = min(search_input.top_k, len(scores))
        top_indices, top_scores = self.similarity_service.top_k_indices(scores, top_k)

        results = []
        for idx, score in zip(top_indices, top_scores):
            chunk_id = aligned_ids[int(idx)]
            metadata = self.persistence_service.get_metadata(chunk_id)

            if metadata:
                results.append(SearchResult(
                    chunk_id=metadata["chunk_id"],
                    doc_id=metadata["doc_id"],
                    text=metadata["text"],
                    score=float(score),
                    metadata=metadata.get("metadata", {}),
                ))

        elapsed = (time.perf_counter() - start) * 1000
        hit_rate = len(filtered_ids) / max(1, len(self.persistence_service.metadata))

        self.metrics_service.log_search(search_input.query, search_input.top_k, elapsed, len(results))
        self.metrics_service.log_json("search", {
            "search_latency_ms": elapsed,
            "index_hit_rate": hit_rate,
            "candidate_count": len(filtered_ids),
            "query_cache_size": len(self.query_cache),
            "hot_vector_cache_size": len(self.hot_vector_cache),
        })

        return results

    def _matches_filter(self, chunk_meta: dict, filter_dict: dict) -> bool:
        return all(chunk_meta.get(key) == value for key, value in filter_dict.items())

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        with self._write_lock():
            return self._delete_document_locked(doc_id, reason="user_delete")

    def _delete_document_locked(self, doc_id: str, reason: str) -> Dict[str, Any]:
        deleted = self.deletion_service.get_deleted_chunks()
        chunk_ids = self.persistence_service.get_chunks_for_document(
            doc_id,
            include_deleted=False,
            deleted_ids=deleted,
        )

        if not chunk_ids:
            raise ValueError(f"Document '{doc_id}' not found")

        tx_id = self.wal_service.begin("delete", doc_id, {"chunks": len(chunk_ids)})

        try:
            deleted_count = self.deletion_service.delete_document(doc_id, chunk_ids, reason=reason)

            for chunk_id in chunk_ids:
                self.index_service.remove_chunk_from_bucket(chunk_id)
                self.sharding_service.remove_chunk_from_shard(chunk_id)
                self.chunk_id_to_index.pop(chunk_id, None)

            self.index_service.save_index(self.persistence_service.index_path)
            self.hot_vector_cache.clear()  # Invalidate cache on deletion
            self.wal_service.commit(tx_id, {"deleted_count": deleted_count})
            self.metrics_service.log_deletion(doc_id, doc_id)

            return {
                "doc_id": doc_id,
                "chunks_deleted": deleted_count,
                "status": "deleted",
            }

        except Exception as exc:
            self.wal_service.fail(tx_id, {"error": str(exc)})
            raise

    def update_document(self, doc_id: str, new_doc_input: DocumentInput) -> IngestionResponse:
        if new_doc_input.doc_id != doc_id:
            new_doc_input = DocumentInput(
                doc_id=doc_id,
                text=new_doc_input.text,
                metadata=new_doc_input.metadata,
            )

        with self._write_lock():
            self._delete_document_locked(doc_id, reason="update")
            return self._ingest_document_locked(
                new_doc_input,
                operation="update",
                update_index=True,
            )

    def rebuild_index(self, save: bool = True) -> Dict[str, Any]:
        start = time.perf_counter()
        deleted = self.deletion_service.get_deleted_chunks()

        chunk_ids = [
            chunk_id for chunk_id in self.persistence_service.get_chunk_ids()
            if chunk_id not in deleted and self.persistence_service.get_metadata(chunk_id)
        ]

        vectors, aligned_chunk_ids = self._get_vectors_for_chunk_ids(chunk_ids)

        summary = self.index_service.build_index(vectors, aligned_chunk_ids)

        if save:
            self.index_service.save_index(self.persistence_service.index_path)

        elapsed = (time.perf_counter() - start) * 1000

        summary.update({
            "status": "ok",
            "rebuild_time_ms": elapsed,
            "deleted_skipped": len(deleted),
            "shard_validation": self.sharding_service.validate_shards(),
            "legacy_fallback_used": False,
        })

        self.metrics_service.log_json("rebuild_index", summary)
        return summary

    def get_stats(self) -> Dict[str, Any]:
        storage = self.persistence_service.get_stats()
        sharding = self.sharding_service.get_stats()
        wal = self.wal_service.get_stats()
        deleted_count = len(self.deletion_service.get_deleted_chunks())
        total_chunks = storage["chunks"]
        active_chunks = max(0, total_chunks - deleted_count)

        return {
            "total_documents": len({
                entry["doc_id"] for cid, entry in self.persistence_service.metadata.items()
                if cid not in self.deletion_service.get_deleted_chunks()
            }),
            "total_chunks": active_chunks,
            "total_vectors": max(sharding["total_vectors"], active_chunks),
            "total_shards": sharding["shard_count"],
            "deleted_count": deleted_count,
            "avg_vectors_per_shard": sharding["avg_vectors_per_shard"],
            "index_type": self.index_service.get_stats()["index_type"],
            "wal_pending": wal["pending"],
            "memory_usage_estimate": self.metrics_service.get_memory_usage_str(),
            "vector_dim": storage["vector_dim"],
        }

    def health_check(self) -> Dict[str, Any]:
        stats = self.get_stats()

        return {
            "status": "ok",
            "documents": stats["total_documents"],
            "chunks": stats["total_chunks"],
            "vector_dim": stats["vector_dim"],
            "memory_usage_estimate": stats["memory_usage_estimate"],
        }


def get_vector_store() -> VectorStore:
    return VectorStore.get_instance()