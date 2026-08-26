
from app.schemas.chat import ChatResponse
from app.schemas.query import AskRequest, AskResponse, SourceChunk

from fastapi import APIRouter, Depends

from app.dependencies import get_executor, get_rag, verify_api_key
from app.services import rag_service


query_router = APIRouter()

@query_router.post("/ask", response_model=ChatResponse ,tags=["Query"])
async def ask(
    request: AskRequest,
    rag=Depends(get_rag),
    executor=Depends(get_executor),
    _=Depends(verify_api_key),
):  
    print(f"[request] POST /ask query={request.query!r} top_k={request.top_k}", flush=True)

    # Call the RAG service to get the response
    hits = await rag_service.retrieve_chunks(rag, executor, request.query)

    if not hits:
        print("[request] POST /ask -> no hits", flush=True)
        return ChatResponse(
            query=request.query,    
            content="No relevant information found.",
            sources=[],
            total_hits=0,
        )

    hits = hits[:request.top_k]  # Limit to top_k results

    context = await rag_service.build_context(rag, executor, hits)
    sources =[
        SourceChunk(
            text=text,
            source=meta.get("source", "unknown"),
            score=score,
        )
        for text,meta,score in hits
    ]

    return AskResponse(
        query=request.query,
        context=context,
        sources=sources,
        total_hits=len(hits),
    )
    