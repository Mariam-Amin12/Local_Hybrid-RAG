from fastapi import APIRouter, Request
from app.config import get_settings

health_router = APIRouter()


@health_router.get("/health", tags=["Health"])
async def health_check(request: Request):

    settings = get_settings()
    rag= request.app.state.rag
    print(
        f"[request] GET /health -> sources={len(rag._ingested_sources)}, "
        f"chunks={len(rag._all_chunks)}",
        flush=True,
    )
    return {"status": "ok", "env": settings.ENV,"sources_loaded":len(rag._ingested_sources)}