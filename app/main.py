import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys 

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent[1]))

from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from flask import app


from app.memory import RedisSessionStore
from app.rag_pipeline.rag import RAG
from app.router import files
from .config import get_settings
from app.router.chat import chat_router
from app.router.ingest import ingest_router
from app.router.health import health_router 
from app.router.query import query_router
from app.router.files import files

# from app.router.upload import upload

STATIC_DIR = Path(__file__).resolve().parent / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize shared resources
    print("[startup] Creating executor and loading settings", flush=True)
    executor = ThreadPoolExecutor(max_workers=4)
    try:
        settings = get_settings()
        loop = asyncio.get_event_loop()
        print(
            f"[startup] persist_dir={settings.RAG_PERSIST_DIR}, "
            f"top_k={settings.RAG_TOP_K}, top_n={settings.RAG_TOP_N}",
            flush=True,
        )

        print("[startup] Constructing RAG pipeline", flush=True)
        rag = RAG(
            persist_dir=settings.RAG_PERSIST_DIR,
            chunk_size=settings.RAG_CHUNK_SIZE,
            overlap=settings.RAG_OVERLAP,
            top_k_retrieve=settings.RAG_TOP_K,
            top_n_rerank=settings.RAG_TOP_N
        )
        print("[startup] Restoring RAG state", flush=True)
        restored = await loop.run_in_executor(executor, rag.load_state)
        print(
            f"[startup] State restore complete: restored={restored}, "
            f"sources={len(rag._ingested_sources)}, chunks={len(rag._all_chunks)}",
            flush=True,
        )

        import redis

        redis_client = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )
        app.state.redis = redis_client
        app.state.session_store = RedisSessionStore(redis_client)
    except Exception as exc:
        print(f"[startup] Failed during initialization: {exc!r}", flush=True)
        executor.shutdown(wait=True)
        raise
    

    app.state.rag = rag
    app.state.executor = executor

    yield  # Control is returned to the application

    # Cleanup resources if needed
    executor.shutdown(wait=True)


app = FastAPI(
    title="Podcast RAG API",
    description="A FastAPI application for a Retrieval-Augmented Generation (RAG) pipeline for podcasts.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(ingest_router)
app.include_router(chat_router)
app.include_router(health_router)
app.include_router(query_router)
app.include_router(files)
# app.include_router(upload)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/",include_in_schema=False)
async def serve_ui():
    print(f"[request] GET / -> {STATIC_DIR / 'index.html'}", flush=True)
    return FileResponse(STATIC_DIR / "index.html")

