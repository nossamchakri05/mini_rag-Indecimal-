import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ingestion import load_documents, split_documents
from embeddings import get_embedding_model
from vector_store import create_vector_store

def debug_pipeline():
    print("--- Debugging Ingestion ---")
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"ERROR: {data_dir} does not exist.")
        return

    docs = load_documents(data_dir)
    print(f"Documents loaded: {len(docs)}")
    if len(docs) == 0:
        print("ERROR: No documents found. Check file extensions.")
        return

    for i, doc in enumerate(docs):
        print(f"Doc {i}: {doc.metadata['source']} (Length: {len(doc.page_content)})")
        print(f"Preview: {doc.page_content[:100]}...")

    print("\n--- Debugging Chunking ---")
    chunks = split_documents(docs)
    print(f"Chunks created: {len(chunks)}")
    if len(chunks) > 0:
        print(f"Chunk 0 Preview: {chunks[0].page_content[:100]}...")

    print("\n--- Debugging Embeddings & Vector Store ---")
    try:
        embeddings = get_embedding_model()
        print("Embedding model loaded.")
        
        vector_store = create_vector_store(chunks, embeddings)
        print("Vector store created.")
        
        query = "delays"
        print(f"\nTesting Retrieval for query: '{query}'")
        results = vector_store.similarity_search(query, k=3)
        
        for i, res in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Source: {res.metadata['source']}")
            print(f"Content: {res.page_content[:200]}...")
            
    except Exception as e:
        print(f"ERROR in Vector Store/Embeddings: {e}")

if __name__ == "__main__":
    debug_pipeline()
