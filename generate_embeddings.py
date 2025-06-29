from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from load_and_chunk import load_pdf

CHROMA_COLLECTION_NAME = "nvidia_docs"


# Main embedding function that embeds chunks with the help of the all-MiniLM-L6-v2 model
def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings


# Stores the vector embeddings and their corresponding chunks into a Chroma collection
def store_in_chroma(chunks, embeddings, persist_dir="chroma_db"):
    client = PersistentClient(path=persist_dir)

    # Tries to delete the existing collection in case the collection is being recreated
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
    except:
        pass

    # Creates a new collection to store the embeddings
    collection = client.create_collection(name=CHROMA_COLLECTION_NAME)

    # Iterates over each chunk and its corresponding embedding, assigns unique ID for each
    for i, (text, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[text],
            embeddings=[embedding.tolist()],
            ids=[f"chunk_{i}"]
        )

    # Should match the amount of produced chunks in load_and_chunk.py
    print(f"Stored {len(chunks)} chunks in Chroma at: {persist_dir}")


if __name__ == "__main__":
    chunks = load_pdf("data/NVIDIAa.pdf")

    embeddings = embed_chunks(chunks)

    store_in_chroma(chunks, embeddings)
