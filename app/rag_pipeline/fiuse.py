
from typing import Tuple,List



def reciprocal_rank_fusion(dense_results: List[Tuple[str, dict, float]], bm25_results: List[Tuple[str, dict, float]], k: int = 60,dense_weight:float=0.6,sparse_weight:float=0.4) -> List[Tuple[str, dict, float]]:
    combined_scores = {}
    docs={}

    for rank, (text, metadata, score) in enumerate(dense_results):
        cid=f"{metadata.get('source', '')}_{metadata.get('chunk_id', '')}"
        combined_scores[cid] = combined_scores.get(cid, 0) + dense_weight * (1 / (rank + 1+k))
        docs[cid] = (text, metadata)
        
    for rank, (text, metadata, score) in enumerate(bm25_results):
        cid=f"{metadata.get('source', '')}_{metadata.get('chunk_id', '')}"
        combined_scores[cid] = combined_scores.get(cid  , 0) + sparse_weight * (1 / (rank + 1+k)) 
        docs[cid] = (text, metadata)

    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)


    return [(docs[cid][0], docs[cid][1], score) for cid, score in sorted_results]
