
from dotenv import load_dotenv
import os
from typing import List
import re
import unicodedata

from app.rag_pipeline.schema import Document, TextChunk
load_dotenv()

DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 500))
DEFAULT_OVERLAP =int(os.getenv("RAG_OVERLAP", 50))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", 5))
DEFAULT_TOP_N = int(os.getenv("RAG_TOP_N", 20))






def clean_text(text: str) -> str: # more general to handle diff types of files
    print(f"[preprocess] Cleaning text with {len(text)} characters", flush=True)
    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove null characters
    text = text.replace("\x00", "")

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove spaces/tabs at the beginning/end of lines
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces before newlines
    text = re.sub(r" +\n", "\n", text)

    # Limit excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned = text.strip()
    print(f"[preprocess] Cleaned text has {len(cleaned)} characters", flush=True)
    return cleaned



def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[TextChunk]:
    print(
        f"[preprocess] Chunking document={document.document_id!r}, "
        f"words={len(document.text.split())}, size={chunk_size}, overlap={overlap}",
        flush=True,
    )
    chunks = []
    words = document.text.split()

    start_idx = 0
    chunk_index = 0
    char_index = 0

    while start_idx < len(words):

        end_idx = min(start_idx + chunk_size, len(words))

        chunk_words = words[start_idx:end_idx]
        chunk_text = " ".join(chunk_words)

        char_start = char_index
        char_end = char_start + len(chunk_text)

        chunks.append(
            TextChunk(
                chunk_id=f"{document.document_id}_chunk_{chunk_index}",
                text=chunk_text,
                source=document.source,
                metadata=document.metadata.copy(),
                char_start=char_start,
                char_end=char_end,
                token_count=len(chunk_words),
            )
        )

        chunk_index += 1

        if end_idx == len(words):
            break

        start_idx += chunk_size - overlap
        char_index = char_end + 1

    print(f"[preprocess] Created {len(chunks)} chunks", flush=True)
    return chunks