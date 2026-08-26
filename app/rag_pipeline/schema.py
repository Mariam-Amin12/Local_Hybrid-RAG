
from dataclasses import dataclass

@dataclass
class Document:  # awel 7aga al document ka kol
    document_id: str
    text: str
    source: str
    metadata: dict

@dataclass # tany 7aga lel chunk al wa7da 
class TextChunk:
    chunk_id: int
    text: str
    metadata: dict
    char_start: int
    char_end: int
    source: str
    token_count: int