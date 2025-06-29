from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import requests
import re

CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "nvidia_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "deepseek-r1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Initializes the Flask web server instance
app = Flask(__name__)

embedder = SentenceTransformer(EMBEDDING_MODEL)
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

# Retrieves the k most relevant document chunks from Chroma
# Higher k may lead to increased accuracy, but longer processing time
# Increasing k leads to diminishing returns
def retrieve_context(question, top_k=5):
    q_embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[q_embedding], n_results=top_k)
    return results["documents"][0]  # list of top-k chunks

# Constructs the prompt for the LLM to use
def build_prompt(chunks, question):
    context = "\n\n".join(chunks)
    return (
        f"Answer the following question in English using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

# Sends the prompt to the local Ollama server hosted LLM
def ask_ollama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # Important on a local system, as it prevents session-context
        "keep_alive": -1
    })

    if response.status_code == 200:
        # Removes empty spaces and garbage data output from DeepSeek's DeepThink feature
        return re.sub(r"<think>.*?</think>", "", response.json()["response"], flags=re.DOTALL).strip()
    else:
        raise Exception(f"Ollama error: {response.status_code} - {response.text}")

# A POST endpoint at /ask to handle incoming questions
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    # Returns the code 400 with an error message in cases of "zero question" requests
    if not question.strip():
        return jsonify({"error": "No question provided."}), 400

    # Retrieves relevant chunks → completes the prompt → gets response from Ollama
    chunks = retrieve_context(question)
    prompt = build_prompt(chunks, question)
    answer = ask_ollama(prompt)

    # Returns the final JSON to the client with an answer and the sources
    return jsonify({
        "question": question,
        "answer": answer,
        "sources": chunks
    })

# Starts the app
if __name__ == "__main__":
    app.run(debug=True)
