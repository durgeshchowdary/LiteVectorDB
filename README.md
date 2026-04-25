# LiteVectorDB

LiteVectorDB is a lightweight FastAPI + NumPy vector database for low-resource RAG deployments. It keeps the Phase 1 API intact while adding Phase 2 database-engine features: sharded vector files, tombstone deletes, WAL recovery, safe writes, admin inspection, batch ingestion, and lightweight ANN indexing.

## Phase 2 Architecture

Storage is append-oriented:

```text
app/data/
  metadata.jsonl        chunk metadata and vector references
  deleted_ids.jsonl     tombstones for logical deletes
  wal.jsonl             pending/committed/failed write records
  index.json            bucket or centroid ANN index
  shard_index.json      shard catalog with checksums
  shards/
    shard_0.npy
    shard_1.npy
    shard_2.npy
    shard_3.npy
```

Writes are protected by a single file lock named `writer`. Each mutating operation writes a pending WAL record, updates shard and metadata files using fsync where needed, then writes a committed WAL record. JSON index files use temp-file plus atomic rename.

## Sharding

Vectors are stored in `app/data/shards/shard_N.npy`. A document is assigned to a shard with:

```text
sha256(doc_id) % LVDB_SHARD_COUNT
```

All chunks for that document go to the same shard. `shard_index.json` tracks shard id, file path, vector count, chunk ids, last update timestamp, checksum, and health. Search loads only the shard files needed by the current candidate set.

## Deletes And Updates

Deletes are logical. `DELETE /documents/{doc_id}` appends chunk tombstones to `deleted_ids.jsonl`, removes chunks from the in-memory ANN index, and leaves physical vectors in shard files for auditability.

Updates are delete plus insert: old chunks are tombstoned with reason `update`, then the replacement document is inserted as new chunks. This preserves the audit trail and avoids in-place vector rewrites.

## WAL And Recovery

`wal.jsonl` records:

- `operation`
- `doc_id`
- `timestamp`
- `status`: `pending`, `committed`, or `failed`
- checksum

Startup verifies WAL checksums and reports unresolved pending transactions in the startup health log and `/admin/health/detailed`. LiteVectorDB does not silently replay ambiguous partial writes; it surfaces them so operators can inspect the append-only files.

## Indexing

The default Phase 2 index is centroid-based clustering using small NumPy k-means. It reduces scan space by searching the nearest centroid buckets. If centroids are missing or corrupt, the system falls back to the original bucket index or all indexed chunks.

Config:

```bash
LVDB_INDEX_TYPE=centroid
LVDB_BUCKETS=100
LVDB_CENTROID_PROBES=3
```

## Configuration

Important environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `LVDB_DATA_DIR` | `app/data` | Storage directory |
| `LVDB_SHARDS_DIR` | `app/data/shards` | Shard file directory |
| `LVDB_SHARD_COUNT` | `4` | Number of vector shards |
| `LVDB_DIMENSION` | `384` | Embedding dimension |
| `LVDB_COMPRESSION` | `true` | Enable vector compression |
| `LVDB_FLOAT16` | `true` | Store vectors as float16 |
| `LVDB_MAX_BATCH_SIZE` | `100` | Batch ingestion limit |
| `LVDB_MAX_TEXT_SIZE` | `100000` | Per-document text limit |
| `LVDB_MAX_PENDING_WRITES` | `1000` | Backpressure setting |
| `LVDB_EMBEDDING_PROVIDER` | `local` | `local` or stub external provider |
| `LVDB_LOCK_TIMEOUT` | `30` | Writer lock timeout seconds |

## Run

```bash
cd litevectordb
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Examples

Health:

```bash
curl http://localhost:8000/health
```

Ingest:

```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"doc1","text":"LiteVectorDB stores vectors in small shards.","metadata":{"source":"demo"}}'
```

Batch ingest:

```bash
curl -X POST http://localhost:8000/documents/batch \
  -H "Content-Type: application/json" \
  -d '{"documents":[{"doc_id":"doc2","text":"Batch document one","metadata":{}},{"doc_id":"doc3","text":"Batch document two","metadata":{}}]}'
```

Search:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"small vector shards","top_k":5}'
```

Delete:

```bash
curl -X DELETE http://localhost:8000/documents/doc1
```

Update:

```bash
curl -X PUT http://localhost:8000/documents/doc2 \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"doc2","text":"Updated document text","metadata":{"version":2}}'
```

Admin stats:

```bash
curl http://localhost:8000/admin/stats
curl http://localhost:8000/admin/shards
curl -X POST http://localhost:8000/admin/rebuild-index
```

## Benchmarks

Start the API, then run:

```bash
python scripts/benchmark.py --vectors 10000 50000 100000 --queries 100 --output benchmark.json
```

The benchmark reports ingestion time, average search latency, p95/p99 latency, memory estimate, index type, shard count, and shard file sizes.

## Limitations

- Local embeddings are deterministic lexical hashes, not semantic model embeddings.
- Deletes are logical until a future compaction pass rewrites shard files.
- File locking is suitable for single-node deployments, not distributed writes.
- The centroid ANN is intentionally lightweight and lower-recall than HNSW or IVF-PQ.
- Authentication and dashboard UI are not included.

## Phase 3 Roadmap

- Distributed nodes
- Replication
- Leader election
- Real embedding adapters for Gemini, OpenAI, and sentence-transformers
- HNSW or IVF-PQ-like ANN
- Dashboard UI
- Online compaction and backup/restore tooling
