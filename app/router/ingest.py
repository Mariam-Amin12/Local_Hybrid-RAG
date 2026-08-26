import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    UploadFile,
)

from app.dependencies import get_executor, get_rag, verify_api_key
from app.schemas.ingest import IngestResponse
from app.services import rag_service

logger = logging.getLogger(__name__)

ingest_router = APIRouter()

_ALLOWED_EXTENSIONS = {".pdf", ".txt"}
LOG_FILE = Path("ingest_log.jsonl")


def _log_ingestion(
    source: str,
    chunks: int,
    cached: bool,
) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "chunks": chunks,
        "cached": cached,
    }

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    logger.debug(
        "Ingestion logged: %s (%d chunks)",
        source,
        chunks,
    )


@ingest_router.post(
    "/ingest",
    response_model=IngestResponse,
)
async def ingest_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    rag=Depends(get_rag),
    executor=Depends(get_executor),
    _=Depends(verify_api_key),
):
    # 1. Validate filename
    filename = Path(file.filename or "upload").name
    _, ext = os.path.splitext(filename)
    print(f"[request] POST /ingest filename={filename!r}, extension={ext!r}", flush=True)

    if ext.lower() not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}",
        )

    # 2. Create temporary file path
    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"{uuid.uuid4().hex}_{filename}",
    )

    try:
        # 3. Save uploaded file
        contents = await file.read()
        print(f"[ingest] Read {len(contents)} bytes from {filename!r}", flush=True)

        with open(tmp_path, "wb") as f:
            f.write(contents)

        # 4. Ingest the file
        stats = await rag_service.ingest_file(
            rag,
            executor,
            tmp_path,
        )
        print(f"[ingest] Completed {filename!r}: {stats}", flush=True)

    except Exception:
        logger.exception(
            "Failed to ingest file: %s",
            filename,
        )
        raise

    finally:
        # 5. Always remove temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"[ingest] Removed temporary file {tmp_path}", flush=True)

    # 6. Log after successful ingestion
    background_tasks.add_task(
        _log_ingestion,
        stats["source"],
        stats["chunks"],
        stats.get("cached", False),
    )

    # 7. Return response
    return IngestResponse(
        source=stats["source"],
        chunks=stats["chunks"],
        cached=stats.get("cached", False),
        total_chunks=stats.get("total_chunks", 0),
        total_sources=stats.get("total_sources", 0),
    )