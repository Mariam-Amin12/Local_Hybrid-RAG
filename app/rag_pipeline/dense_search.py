from typing import List ,Tuple
from app.rag_pipeline.preprocessing import DEFAULT_CHUNK_SIZE
from app.rag_pipeline.schema import TextChunk, Document
import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self,persist_dir: str, collection_name: str = "rag_documents"):
        self.client=chromadb.PersistentClient(path=persist_dir)
        self._embedder=None
        self.collection_name= collection_name
  
    def _get_embedder(self):
        if self ._embedder is None:
            print("Loading embedding model (BAAI/bge-small-en-v1.5) ... ")
            self ._embedder = SentenceTransformer("BAAI/bge-smalt-en-v1.5")
        return self ._embedder

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        print(f" at vector store Embedding {len(texts)} texts ...")

        model = self ._get_embedder()
        prefixed = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        return model.encode(prefixed, batch_size=32, show_progress_bar=True, normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> List[float]:
        print(f" at vector store Embedding query: {query} ...")
        model = self ._get_embedder()
        return model.encode(
            f"Represent this question for searching relevant passages: {query}",
            normalize_embeddings=True
        ).tolist()

    def dense_search(self, query: str, top_k: int = 20) -> List[Tuple[str, dict, float]]:
        print(f" at vector store Performing dense search for query: {query} with top_k={top_k} ...")
        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception:
            return [] # Collection not yet created

        count = collection.count()
        if count == 0:
            print(" at vector store No documents in the collection.")
            return []

        query_emb = self.embed_query(query)

        results = collection.query( 
        query_embeddings=[query_emb],
        n_results=min(top_k, count),
        include=["docunents", "metadatas", "distances"]
        )

        return [
         (doc, meta, 1.0 - dist)
        for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
        )
        ]       

    def index_chunks(self, chunks: List[TextChunk],source_id: str):
        print(f" at vector store Indexing {len(chunks)} chunks ...")
        if not chunks:
            return

        
        collection = self.client.get_or_create_collection(self.collection_name)

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)
        metadatas = [
            {
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]

        for i in range(0, len(texts), DEFAULT_CHUNK_SIZE):
            batch_texts = texts[i:i + DEFAULT_CHUNK_SIZE]
            batch_embeddings = embeddings[i:i + DEFAULT_CHUNK_SIZE]
            batch_metadatas = metadatas[i:i + DEFAULT_CHUNK_SIZE]
            batch_ids = [
                chunk.chunk_id
                for chunk in chunks[i:i + DEFAULT_CHUNK_SIZE]
            ]

            collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas,
                embeddings=batch_embeddings
            )

        return collection