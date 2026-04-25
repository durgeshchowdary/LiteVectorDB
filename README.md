# LiteVectorDB 🧠

A lightweight, production-ready vector database designed for real-world AI systems and low-resource deployments.

---

## 🚀 Why LiteVectorDB?

Most vector databases (FAISS, Pinecone) focus on raw speed.  
LiteVectorDB focuses on:

- Deployability
- Reliability
- System-level performance

---

## ⚡ Performance

- ✅ 5000 requests tested
- ✅ 100 concurrent users
- ✅ 100% success rate
- 🚀 **~214 QPS achieved**

---

## 🔬 Benchmark

| System | Latency |
|--------|--------|
| LiteVectorDB | ~25ms (end-to-end) |
| FAISS | ~4ms (in-memory only) |

### Insight

FAISS measures only vector similarity.  
LiteVectorDB measures full pipeline:

- API
- embedding
- shard retrieval
- metadata filtering
- ranking

---

## 🧠 Features

- 🔹 Sharding-based storage
- 🔹 Write-Ahead Logging (WAL) for crash recovery
- 🔹 ANN search (centroid-based indexing)
- 🔹 Metadata filtering
- 🔹 Hot vector caching (**2.6x performance improvement**)

---

## 🏗 Architecture

### Ingestion Flow
Document → Chunking → Embedding → Sharding → Persistence

### Search Flow
Query → Embedding → ANN Buckets → Vector Retrieval → Ranking

---

## 🎯 Goal

LiteVectorDB is built as a deployable alternative to FAISS for:

- AI search systems
- RAG pipelines
- low-resource deployments

---

## 🔜 Next Step

Integration into:

- ResearchMind (AI search engine)
- TaxBee (AI tax assistant)

---

## 🛠 Tech Stack

- Python
- FastAPI
- NumPy
- Custom ANN indexing

---