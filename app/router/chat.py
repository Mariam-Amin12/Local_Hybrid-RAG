import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib import response
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends
from groq import Groq

from app.dependencies import get_executor, get_rag, get_redis_session_store, verify_api_key
from app. schemas. chat import ChatMessage, ChatRequest, ChatResponse
from app. schemas.query import SourceChunk
from app.services import rag_service

logger = logging.getLogger(__name__)
chat_router = APIRouter()

LOG_FILE = Path("chat_log.jsonl")

def _log_chat(session_id: str, query: str, answer: str) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "query": query,
        "answer": answer,
    }

    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.debug("Chat turn logged for session %s", session_id)


def _call_llm_sync(
    messages: list[ChatMessage],
    context: str,
    provider: str,
    api_key: str,
    active_sources: list,
) -> str:
    print(
        f"[llm] Starting provider={provider!r}, messages={len(messages)}, "
        f"context_chars={len(context)}, sources={len(active_sources)}",
        flush=True,
    )
    if not context or context.strip() == "No relevant excerpts found.":
        print("[llm] No usable context; returning fallback", flush=True)
        return "I couldn't find relevant information in the transcripts to answer this question."

    source_list = ", ".join(active_sources) if active_sources else "unknown"
    system_prompt = f"""You are a podcast transcript assistant. Answer questions using the transcripts.

    STRICT RULES:
    1. ONLY use information explicitly present in the TRANSCRIPT EXCERPTS below.
    2. If the excerpts do not contain a clear answer, say: "This topic is not covered in the available transcripts."
    3. When answering, cite which source and approximate timestamp (e.g., "In [source_name] around 12:30").
    4. If multiple transcripts discuss the same topic, compare or synthesize their perspectives.
    5. Do NOT guess, infer, or use outside knowledge under any circumstances.
    6. Keep answers concise and grounded.

    TRANSCRIPT EXCERPTS:
    {context}"""

    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages += [{"role": m.role, "content": m.content} for m in messages]
    import ollama 
    response = ollama.chat(
    model="qwen2.5:3b-instruct",
    messages=full_messages,
    options={
        "temperature": 0.3,
        "num_predict": 1824,
    },
)

    answer = response["message"]["content"]
    print(f"[llm] Response received: {len(answer)} characters", flush=True)
    return answer


@chat_router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    rag=Depends(get_rag),
    executor=Depends(get_executor),
    _=Depends(verify_api_key),
    memory=Depends(get_redis_session_store)
):
    session_id = request.session_id or str(uuid.uuid4())
    print(f"[request] POST /chat session={session_id}, query={request.query!r}", flush=True)
    history = memory.get_history(session_id) if session_id else []
    # Retrieve relevant chunks from the RAG service
    hits = await rag_service.retrieve_chunks(rag, executor, request.query)

    if not hits:
        print("[request] POST /chat -> no hits", flush=True)
        user_message = ChatMessage(role="user", content=request.query)
        answer = ChatMessage(role="assistant", content="No relevant information found.")
        memory.save_turn(session_id, user_message, answer)
        background_tasks.add_task(
            _log_chat, session_id, request.query, answer.content
        )
        return ChatResponse(
            answer=answer.content,
            session_id=session_id,
            sources=[],
            total_hits=0,
        )

    hits = hits[:request.top_k]  # Limit to top_k results
    print(f"[request] POST /chat using {len(hits)} hits", flush=True)

    context = await rag_service.build_context(rag, executor, hits)

    active_sources = [meta.get("source", "unknown") for _, meta, _ in hits]
    messages=history+[ChatMessage(role="user", content=request.query)]



    # Call the LLM synchronously in a thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        executor,
        _call_llm_sync,
        messages,
        context,
        request.provider,
        request.llm_api_key,
        active_sources,
    )

    # Log the chat turn in the background
    user_message = ChatMessage(role="user", content=request.query)
    ai_message = ChatMessage(role="assistant", content=answer)
    memory.save_turn(session_id, user_message, ai_message)
    background_tasks.add_task(
        _log_chat, session_id, request.query, answer
    )
    sources = [
        SourceChunk(

            text=text,
            source=meta.get("source", "unknown"),       
            score=score,
        )
        for text, meta, score in hits
    ]


    return ChatResponse(
        answer=answer,
        session_id=session_id,
        sources=sources,
        total_hits=len(hits),
    )

@chat_router.delete("/chat/{session_id}", tags=["Chat"])
async def delete_chat_session(
    session_id: str,
    _=Depends(verify_api_key),
    memory=Depends(get_redis_session_store)
):
    print(f"[request] DELETE /chat/{session_id}", flush=True)
    memory.clear_session(session_id)
    return {"message": f"Chat session {session_id} deleted."}

