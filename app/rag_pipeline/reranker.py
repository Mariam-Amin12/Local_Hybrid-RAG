
from sentence_transformers import CrossEncoder

from typing import List, Tuple, Dict
class Reranker :
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading reranker model ({model_name}) ... ")
        self.model = None 
        self.model_name=model_name


    def _load(self):
        if self.model is None:
            print(f"Loading reranker model ({self.model_name}) ... ")
            self.model = CrossEncoder(self.model_name)


    def rerank(self, query: str, results: List[Tuple[str, dict, float]], top_n: int = 5) -> List[Tuple[str, dict, float]]:
        if not results:
            return []
        
        self._load()

        texts = [text for text, _, _ in results]
        query_texts = [query] * len(texts)
        scores = self.model.predict(query_texts, texts)

        scored_results = [(text, metadata, score) for (text, metadata, _), score in zip(results, scores)]
        scored_results.sort(key=lambda x: x[2], reverse=True)

        return scored_results[:top_n]
