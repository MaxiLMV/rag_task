# RAG API

This project implements a Retrieval-Augmented Generation (RAG) system that answers natural language questions based on a PDF document.

---

## How It Works

1. **PDF Loading & Chunking**  
   The PDF is split into two types of chunks:
   - Text (preserving full sentences)
   - Structured tables

2. **Embedding & Vector DB**  
   Each chunk is embedded using `sentence-transformers` and stored in a persistent `ChromaDB` collection.

3. **Context Retrieval & Answering**  
   When a question is asked:
   - It is embedded and compared against the vector DB.
   - The most relevant chunks are selected.
   - A prompt is built and passed to a local **Ollama** LLM (In this case, `deepseek-r1:8b`).

4. **API**  
   A Flask API (`/ask`) takes a question and returns:
   - The answer
   - The relevant source chunks used

---

## Setup Instructions

### 1. Clone the Repository

```bash
https://github.com/MaxiLMV/rag_task.git
cd rag_task
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Install Ollama and LLM

Install Ollama [here](https://ollama.com/download).

After you make sure the executable is running, open a terminal and enter this command:
```bash
ollama run deepseek-r1:8b
```
This command will install the model and also run it for the first time. You can stop it with `CTRL + D` or by typing `/bye` in the terminal.

The `deepseek-r1:8b` model is used in this configuration, but most local LLMs can be used. 

You can find a list of models [here](https://ollama.com/search). Note that the larger the model, the slower it will run on a regular PC.

### 4. Generate Embeddings
```bash
python generate_embeddings.py
```
## Running the system

### 1. Start the Ollama server
Launch the Ollama executable, open a terminal and enter this command:
```bash
ollama run deepseek-r1:8b
``` 
You may type "hello" to see if the model is operational.
### 2. Start the Flask API
```bash
python rag_api.py
```

## Tests
You may want to run some tests, which you can do with the following command:
```bash
python -m pytest test_rag_api.py
```
This set of tests contains tests for:
- A simple question
- A blank question
- An unrelated question
- Multiple questions in one
- A very long question
## Example Usage
Enter this command in the terminal (the "question" field can be changed for any question).
```bash
(Invoke-WebRequest -Method POST http://localhost:5000/ask -Body '{ "question": "When was NVIDIA founded?" }' -ContentType "application/json").Content
```

This particular question should generate a response with the year 1993 in it as well as the chunks used as sources.
