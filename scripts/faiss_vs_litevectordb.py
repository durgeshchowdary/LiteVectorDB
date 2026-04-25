import time
import statistics
import numpy as np
import requests
import faiss

URL = "http://127.0.0.1:8012"


def litevectordb_query():
    t0 = time.perf_counter()
    r = requests.post(
        f"{URL}/search",
        json={"query": "tax deduction 80C HRA PPF ELSS", "top_k": 5},
        timeout=30,
    )
    r.raise_for_status()
    latency = time.perf_counter() - t0
    return latency, len(r.json())


def benchmark_litevectordb(warmup=5, runs=20):
    for _ in range(warmup):
        litevectordb_query()

    latencies = []
    results_count = 0

    for _ in range(runs):
        latency, count = litevectordb_query()
        latencies.append(latency)
        results_count = count

    return {
        "avg": statistics.mean(latencies),
        "p95": sorted(latencies)[int(0.95 * len(latencies)) - 1],
        "min": min(latencies),
        "max": max(latencies),
        "results": results_count,
    }


def benchmark_faiss(n=50000, dim=384, runs=20):
    xb = np.random.random((n, dim)).astype("float32")
    xq = np.random.random((1, dim)).astype("float32")

    index = faiss.IndexFlatL2(dim)

    t0 = time.perf_counter()
    index.add(xb)
    build_time = time.perf_counter() - t0

    # warmup
    for _ in range(5):
        index.search(xq, 5)

    latencies = []
    for _ in range(runs):
        t1 = time.perf_counter()
        index.search(xq, 5)
        latencies.append(time.perf_counter() - t1)

    return {
        "build_time": build_time,
        "avg": statistics.mean(latencies),
        "p95": sorted(latencies)[int(0.95 * len(latencies)) - 1],
        "min": min(latencies),
        "max": max(latencies),
    }


if __name__ == "__main__":
    print("🔥 Benchmarking LiteVectorDB API...")
    lite = benchmark_litevectordb()

    print("🔥 Benchmarking FAISS raw in-memory search...")
    faiss_result = benchmark_faiss()

    print("\n===== RESULTS =====")
    print(f"LiteVectorDB API Avg Latency: {lite['avg']:.4f}s")
    print(f"LiteVectorDB API P95 Latency: {lite['p95']:.4f}s")
    print(f"LiteVectorDB Min/Max: {lite['min']:.4f}s / {lite['max']:.4f}s")
    print(f"LiteVectorDB Results: {lite['results']}")

    print(f"\nFAISS Build Time: {faiss_result['build_time']:.4f}s")
    print(f"FAISS Raw Search Avg Latency: {faiss_result['avg']:.4f}s")
    print(f"FAISS Raw Search P95 Latency: {faiss_result['p95']:.4f}s")
    print(f"FAISS Min/Max: {faiss_result['min']:.4f}s / {faiss_result['max']:.4f}s")

    print("\nNOTE:")
    print("FAISS result is raw in-memory vector search only.")
    print("LiteVectorDB result includes HTTP API, query embedding, metadata handling, shard retrieval, and ranking.")