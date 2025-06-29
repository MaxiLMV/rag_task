# Development file, is not necessary for the RAG algorithm to function

import re
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import requests

CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "nvidia_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "deepseek-r1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

embedder = SentenceTransformer(EMBEDDING_MODEL)
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

def retrieve_context(question, top_k=5):
    q_embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[q_embedding], n_results=top_k)
    return results["documents"][0]  # list of top-k chunks

def build_prompt(chunks, question):
    context = "\n\n".join(chunks)
    prompt = (
        f"Answer the following question in English using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt

def ask_ollama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": -1
    })

    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        raise Exception(f"Ollama error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    question = input("Ask a question: ")

    chunks = retrieve_context(question)
    prompt = build_prompt(chunks, question)
    answer = re.sub(r"<think>.*?</think>", "", ask_ollama(prompt), flags=re.DOTALL).strip()

    print("\nSource Chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---\n{chunk.strip()}")

    print("\nAnswer:")
    print(answer)