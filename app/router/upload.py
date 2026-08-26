from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

upload = APIRouter()

UPLOAD_DIR = Path(
    "D:/projects/ai/prodcast_rag_model_deployment/uploads"
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@upload.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    allowed_extensions = {".pdf", ".txt", ".docx"}

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT and DOCX files are supported"
        )

    safe_name = Path(file.filename).name

    file_path = UPLOAD_DIR / safe_name

    content = await file.read()

    file_path.write_bytes(content)

    return {
        "filename": safe_name,
        "path": str(file_path),
        "message": "File uploaded successfully"
    }