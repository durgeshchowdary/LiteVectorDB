"""Benchmark script for LiteVectorDB."""
import time
import json
import random
import string
import numpy as np
from typing import List, Dict, Any
import statistics


def generate_random_text(min_words: int = 50, max_words: int = 200) -> str:
    """Generate random text for benchmarking."""
    words = []
    for _ in range(random.randint(min_words, max_words)):
        word_len = random.randint(3, 12)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)
    return ' '.join(words)


def generate_documents(count: int) -> List[Dict[str, Any]]:
    """Generate random documents for benchmarking."""
    documents = []
    for i in range(count):
        doc = {
            "doc_id": f"bench_doc_{i}",
            "text": generate_random_text(),
            "metadata": {
                "category": random.choice(["tech", "science", "business", "sports"]),
                "priority": random.randint(1, 5)
            }
        }
        documents.append(doc)
    return documents


def run_benchmark(
    base_url: str = "http://127.0.0.1:8000",
    vector_counts: List[int] = [10000, 50000, 100000],
    search_queries: int = 100
) -> Dict[str, Any]:
    """Run comprehensive benchmark."""
    import urllib.request
    import urllib.error
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vector_counts": vector_counts,
        "search_queries": search_queries,
        "benchmarks": []
    }
    
    for vector_count in vector_counts:
        print(f"\n{'='*60}")
        print(f"Benchmarking with {vector_count} vectors")
        print(f"{'='*60}")
        
        # Clear existing data
        print("Clearing existing data...")
        try:
            req = urllib.request.Request(
                f"{base_url}/admin/data/clear?confirm=true",
                method="POST"
            )
            urllib.request.urlopen(req, timeout=30)
        except Exception as e:
            print(f"Warning: Could not clear data: {e}")
        
        # Generate documents
        print(f"Generating {vector_count} documents...")
        documents = generate_documents(vector_count)
        
        # Batch ingest
        print(f"Ingesting {vector_count} documents...")
        start_time = time.perf_counter()
        
        # Split into batches of 100
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            payload = json.dumps({"documents": batch}).encode('utf-8')
            
            req = urllib.request.Request(
                f"{base_url}/documents/batch",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            try:
                response = urllib.request.urlopen(req, timeout=120)
                result = json.loads(response.read())
                print(f"  Batch {i//batch_size + 1}: {result.get('total_chunks', 0)} chunks")
            except Exception as e:
                print(f"  Error in batch {i//batch_size + 1}: {e}")
        
        ingestion_time = time.perf_counter() - start_time
        print(f"Ingestion completed in {ingestion_time:.2f}s")
        
        # Get stats
        req = urllib.request.Request(f"{base_url}/health")
        response = urllib.request.urlopen(req, timeout=10)
        health = json.loads(response.read())
        
        # Run search benchmarks
        print(f"Running {search_queries} search queries...")
        search_times = []
        
        for i in range(search_queries):
            query = generate_random_text(min_words=5, max_words=10)
            payload = json.dumps({
                "query": query,
                "top_k": 10
            }).encode('utf-8')
            
            req = urllib.request.Request(
                f"{base_url}/search",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            start_time = time.perf_counter()
            try:
                response = urllib.request.urlopen(req, timeout=30)
                search_results = json.loads(response.read())
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                search_times.append(elapsed_ms)
            except Exception as e:
                print(f"  Search error: {e}")
        
        if search_times:
            search_times.sort()
            p50 = search_times[len(search_times) // 2]
            p95 = search_times[int(len(search_times) * 0.95)]
            p99 = search_times[int(len(search_times) * 0.99)]
            avg_search = statistics.mean(search_times)
        else:
            p50 = p95 = p99 = avg_search = 0
        
        # Get admin/storage info
        req = urllib.request.Request(f"{base_url}/admin/stats/system")
        response = urllib.request.urlopen(req, timeout=10)
        system_stats = json.loads(response.read())
        req = urllib.request.Request(f"{base_url}/admin/stats")
        response = urllib.request.urlopen(req, timeout=10)
        admin_stats = json.loads(response.read())
        req = urllib.request.Request(f"{base_url}/admin/shards")
        response = urllib.request.urlopen(req, timeout=10)
        shard_stats = json.loads(response.read())
        
        # Store benchmark results
        benchmark_result = {
            "vector_count": vector_count,
            "ingestion_time_seconds": ingestion_time,
            "vectors_per_second": vector_count / ingestion_time if ingestion_time > 0 else 0,
            "avg_search_latency_ms": avg_search,
            "p50_search_latency_ms": p50,
            "p95_search_latency_ms": p95,
            "p99_search_latency_ms": p99,
            "memory_rss_mb": system_stats.get("memory", {}).get("rss_mb", 0),
            "memory_usage_estimate": admin_stats.get("memory_usage_estimate"),
            "index_type": admin_stats.get("index_type"),
            "total_shards": admin_stats.get("total_shards"),
            "shard_sizes": [
                {"shard_id": s.get("shard_id"), "file_size": s.get("file_size"), "vector_count": s.get("vector_count")}
                for s in shard_stats.get("shards", [])
            ],
            "health": health
        }
        
        results["benchmarks"].append(benchmark_result)
        
        # Print summary
        print(f"\nResults for {vector_count} vectors:")
        print(f"  Ingestion time: {ingestion_time:.2f}s ({vector_count/ingestion_time:.0f} vectors/sec)")
        print(f"  Avg search latency: {avg_search:.2f}ms")
        print(f"  P95 search latency: {p95:.2f}ms")
        print(f"  Memory (RSS): {system_stats.get('memory', {}).get('rss_mb', 0):.1f}MB")
        print(f"  Index: {admin_stats.get('index_type')} across {admin_stats.get('total_shards')} shards")
    
    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print benchmark summary."""
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    for bench in results["benchmarks"]:
        print(f"\n{bench['vector_count']} vectors:")
        print(f"  Ingestion: {bench['ingestion_time_seconds']:.2f}s ({bench['vectors_per_second']:.0f}/s)")
        print(f"  Search latency: avg={bench['avg_search_latency_ms']:.2f}ms, p95={bench['p95_search_latency_ms']:.2f}ms")
        print(f"  Memory: {bench['memory_rss_mb']:.1f}MB")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LiteVectorDB Benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL")
    parser.add_argument("--vectors", nargs="+", type=int, default=[10000, 50000, 100000], 
                        help="Vector counts to benchmark")
    parser.add_argument("--queries", type=int, default=100, help="Number of search queries")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    print("LiteVectorDB Benchmark")
    print(f"Target: {args.url}")
    print(f"Vector counts: {args.vectors}")
    print(f"Search queries: {args.queries}")
    
    results = run_benchmark(
        base_url=args.url,
        vector_counts=args.vectors,
        search_queries=args.queries
    )
    
    print_summary(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
