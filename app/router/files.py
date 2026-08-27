import mimetypes

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

files = APIRouter()


UPLOAD_DIR = Path(
    "D:/projects/ai/prodcast_rag_model_deployment/app/uploads"
)


@files.get("/files/{filename}")
def get_file(filename: str):

    safe_name = Path(filename).name
    file_path = UPLOAD_DIR / safe_name
    print (f"[request] GET /files/{filename} safe_name={safe_name!r}, file_path={file_path!r}", flush=True)

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )

    media_type, _ = mimetypes.guess_type(safe_name)
    print(f"[request] GET /files/{filename} media_type={media_type!r}", flush=True)

    return FileResponse(
        path=file_path,
        filename=safe_name,
        media_type=media_type,
        content_disposition_type="inline",
    )