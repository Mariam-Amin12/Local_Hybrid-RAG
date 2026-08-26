from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

files = APIRouter()


UPLOAD_DIR = Path(
    "D:/projects/ai/prodcast_rag_model_deployment/uploads"
)


@files.get("/files/{filename}")
def get_file(filename: str):

    safe_name = Path(filename).name
    file_path = UPLOAD_DIR / safe_name

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )

    return FileResponse(
        path=file_path,
        filename=safe_name
    )