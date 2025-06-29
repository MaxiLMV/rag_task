# Development file, is not necessary for the RAG algorithm to function

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import sys

CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "nvidia_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def query_chroma(question: str, top_k=5):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_embedding = model.encode(question).tolist()

    client = PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Ask a question: ")
    else:
        question = " ".join(sys.argv[1:])

    results = query_chroma(question)

    print("\nTop Matching Chunks:\n")
    for i, chunk in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(chunk.strip())
        print()