
from sentence_transformers import CrossEncoder

from typing import List, Tuple, Dict
class Reranker :
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"[reranker] Configured model: {model_name}", flush=True)
        self.model = None 
        self.model_name=model_name


    def _load(self):
        if self.model is None:
            print(f"[reranker] Loading model: {self.model_name}", flush=True)
            self.model = CrossEncoder(self.model_name)
            print("[reranker] Model loaded", flush=True)


    def rerank(self, query: str, results: List[Tuple[str, dict, float]], top_n: int = 5) -> List[Tuple[str, dict, float]]:
        if not results:
            print("[reranker] Skipping rerank: no results", flush=True)
            return []
        
        self._load()
        

        texts = [text for text, _, _ in results]
        query_texts = [query] * len(texts)

        pairs = [
            [query, text]
            for text in texts
        ]
        scores = self.model.predict(pairs)

        scored_results = [(text, metadata, score) for (text, metadata, _), score in zip(results, scores)]
        scored_results.sort(key=lambda x: x[2], reverse=True)

        reranked = scored_results[:top_n]
        print(f"[reranker] Returning {len(reranked)} results", flush=True)
        return reranked
