"""Demo script for LiteVectorDB.

Demonstrates document ingestion and search functionality.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import DocumentInput, SearchInput
from app.services.vector_store import get_vector_store


def main():
    """Run demo."""
    print("=" * 60)
    print("LiteVectorDB Demo")
    print("=" * 60)
    
    # Get vector store
    vector_store = get_vector_store()
    
    # Check health first
    print("\n1. Health Check:")
    health = vector_store.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Documents: {health['documents']}")
    print(f"   Chunks: {health['chunks']}")
    print(f"   Vector Dim: {health['vector_dim']}")
    print(f"   Memory: {health['memory_usage_estimate']}")
    
    # Sample documents
    documents = [
        DocumentInput(
            doc_id="doc1",
            text="""LiteVectorDB is a lightweight vector database designed for low-resource environments. 
It provides efficient vector storage and similarity search without the overhead of larger systems.
The project aims to solve deployment issues faced with FAISS on platforms like Railway and Hugging Face Spaces.""",
            metadata={"source": "documentation", "type": "intro"}
        ),
        DocumentInput(
            doc_id="doc2",
            text="""Performance is a key design goal for LiteVectorDB. The system can handle at least 100k vectors
while keeping memory usage under 512MB. It uses float16 compression to reduce memory footprint
and implements bucket-based indexing for efficient approximate nearest neighbor search.""",
            metadata={"source": "documentation", "type": "performance"}
        ),
        DocumentInput(
            doc_id="doc3",
            text="""The architecture of LiteVectorDB is modular and extensible. It consists of several key components:
the embedding service for generating vector representations, the chunker for text segmentation,
the index service for bucket-based partitioning, and the persistence layer for data durability.
Each component can be replaced or enhanced independently.""",
            metadata={"source": "documentation", "type": "architecture"}
        ),
    ]
    
    # Ingest documents
    print("\n2. Ingesting Documents:")
    for doc in documents:
        print(f"   - Ingesting: {doc.doc_id}")
        response = vector_store.ingest_document(doc)
        print(f"     Created {response.chunks_created} chunks in {response.ingestion_time_ms:.2f}ms")
    
    # Check health after ingestion
    print("\n3. Health Check After Ingestion:")
    health = vector_store.health_check()
    print(f"   Documents: {health['documents']}")
    print(f"   Chunks: {health['chunks']}")
    print(f"   Memory: {health['memory_usage_estimate']}")
    
    # Run searches
    print("\n4. Running Searches:")
    
    queries = [
        "How does LiteVectorDB handle memory efficiency?",
        "What is the architecture like?",
        "performance optimization for vectors"
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        search_input = SearchInput(query=query, top_k=3)
        results = vector_store.search(search_input)
        
        for i, result in enumerate(results, 1):
            print(f"   {i}. [score: {result.score:.4f}] {result.text[:80]}...")
            print(f"      doc_id: {result.doc_id}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()