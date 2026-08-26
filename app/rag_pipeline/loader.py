from pathlib import Path
from docx import Document


def load_txt(file_path: str) -> str:
    return Path(file_path).read_text(
        encoding="utf-8",
        errors="replace"
    )


def load_pdf(file_path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(file_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages)


def load_docx(file_path: str) -> str:

    document = Document(file_path)

    paragraphs = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs)


def load_file(file_path: str) -> str:

    path = Path(file_path)

    extension = path.suffix.lower()

    if extension == ".txt":
        return load_txt(file_path)

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension == ".docx":
        return load_docx(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )