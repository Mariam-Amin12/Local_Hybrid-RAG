

from typing import List, Tuple
from rank_bm25 import BM25Okapi
import re
import numpy as np

from app.rag_pipeline.schema import TextChunk


class BM25Index:
    def __init__(self):
        self.chunks: List[TextChunk] = []
        self.bm25 = None

    def build(self, chunks: List[TextChunk]) -> None:
        print(f"[bm25] Building index from {len(chunks)} chunks", flush=True)
        if not chunks:
            self.chunks = []
            self.bm25 = None
            return

        self.chunks = list(chunks)

        tokenized_corpus = [
            self._tokenize_query(chunk.text)
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        print("[bm25] Index build complete", flush=True)

    def _tokenize_query(self, query: str) -> List[str]:
        return re.sub(
            r"[^\w\s]",
            "",
            query.lower(),
        ).split()

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Tuple[str, dict, float]]:

        if self.bm25 is None or not self.chunks:
            print("[bm25] Search skipped: index is empty", flush=True)
            return []

        if not query.strip():
            print("[bm25] Search skipped: query is empty", flush=True)
            return []

        tokenized_query = self._tokenize_query(query)

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        matches = [
            (
                self.chunks[i].text,
                {
                    **self.chunks[i].metadata,
                    "chunk_id": self.chunks[i].chunk_id,
                    "source": self.chunks[i].source,
                    "char_start": self.chunks[i].char_start,
                    "char_end": self.chunks[i].char_end,
                    "token_count": self.chunks[i].token_count,
                },
                float(scores[i]),
            )
            for i in top_indices
            if scores[i] > 0
        ]
        print(f"[bm25] Search returned {len(matches)} results", flush=True)
        return matches
