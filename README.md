# Podcast RAG

A small FastAPI app for searching podcast transcripts and asking questions about them. It uses hybrid retrieval to find relevant passages, then sends the selected context to a local Ollama model.

## How it works

1. Upload a transcript.
2. The app cleans and splits it into chunks.
3. Each chunk is indexed with:
   - Chroma and `BAAI/bge-small-en-v1.5` for semantic search
   - BM25 for keyword search
4. Results are combined with reciprocal-rank fusion.
5. A cross-encoder reranks the results.
6. Ollama generates a grounded answer from the retrieved passages.

Indexed data is stored in `rag_store/`. Chat sessions are stored in Redis.

## Requirements

- Python 3.10+
- Redis
- Ollama
- The Ollama model `qwen2.5:3b`

The project does not currently include a dependency file. Install the main packages with:

```powershell
pip install fastapi uvicorn pydantic-settings python-dotenv redis chromadb sentence-transformers rank-bm25 numpy python-multipart ollama
```

## Run locally

Start Redis:

```powershell
redis-server
```

Start Ollama and download the chat model:

```powershell
ollama serve
ollama pull qwen2.5:3b
```

From the project root, start the API:

```powershell
uvicorn app.main:app --reload
```

Open the web UI at [http://localhost:8000/](http://localhost:8000/).

API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Configuration

Create a `.env` file in the project root when you need to change the defaults:

```env
API_KEY=dev-key-123
REDIS_URL=redis://localhost:6379/0
RAG_PERSIST_DIR=./rag_store
RAG_CHUNK_SIZE=500
RAG_OVERLAP=88
RAG_TOP_K=20
RAG_TOP_N=5
ENV=development
UPLOAD_DIR=./app/uploads
```

`API_KEY` protects the ingest, chat, query, and file-download endpoints. Replace the development default before sharing the service.

## API

### Check health

```powershell
curl http://localhost:8000/health
```

### Upload a transcript

The API currently accepts `.txt` and `.pdf` filenames. Text transcripts are the supported input format for the current ingestion code.

```powershell
curl -X POST http://localhost:8000/ingest `
  -H "x-api-key: dev-key-123" `
  -F "file=@episode.txt"
```

### Ask a question with chat

```powershell
curl -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -H "x-api-key: dev-key-123" `
  -d '{
    "query": "What is the main topic of this episode?",
    "session_id": null,
    "llm_api_key": "unused",
    "provider": "ollama",
    "top_k": 5
  }'
```

The current chat implementation uses local Ollama. The `provider` and `llm_api_key` fields are kept in the request schema but are not used yet.

### Delete a chat session

```powershell
curl -X DELETE http://localhost:8000/chat/session-id `
  -H "x-api-key: dev-key-123"
```

### Download an uploaded file

```powershell
curl http://localhost:8000/files/episode.txt -o episode.txt
```

## Project layout

```text
app/
  main.py                 FastAPI application and startup
  config.py               Environment settings
  router/                 API endpoints
  services/               Async wrappers for RAG work
  rag_pipeline/           Chunking, search, fusion, and reranking
  schemas/                Request and response models
  static/index.html       Browser UI
  static/index.py         Optional Streamlit UI
rag_store/                Persistent chunks and Chroma data
```

## Notes

- The first model use may download model files and take a while.
- Redis must be running before the API starts.
- Do not commit `.env`, API keys, uploaded files, or generated `rag_store/` data.
- Debug messages are printed by the API and can help trace startup, ingestion, retrieval, and chat requests.
