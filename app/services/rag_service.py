


import asyncio


async def ingest_file(rag, executor, tmp_path: str) -> dict:
    
    loop = asyncio.get_running_loop()
    print(f"[service] Queueing ingestion for {tmp_path}", flush=True)
    stats = await loop.run_in_executor(executor, rag.ingest, tmp_path)
    print(f"[service] Ingestion complete: {stats}", flush=True)
    return stats


async def retrieve_chunks(rag, executor, query: str) -> list:

    loop = asyncio.get_running_loop()
    print(f"[service] Queueing retrieval for query: {query!r}", flush=True)
    hits = await loop.run_in_executor(executor, rag.retrieve, query)
    print(f"[service] Retrieval complete: {len(hits)} hits", flush=True)
    return hits


async def build_context(rag, executor, hits: list) -> str:

    loop = asyncio.get_running_loop()
    print(f"[service] Building context from {len(hits)} hits", flush=True)
    context = await loop.run_in_executor(executor, rag.format_context, hits)
    print(f"[service] Context built: {len(context)} characters", flush=True)
    return context