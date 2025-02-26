# retrieval/hybrid_retriever.py

from .faiss_manager import FAISSManager
from .bm25_manager import BM25Manager


class HybridRetriever:
    def __init__(self, faiss_mgr: FAISSManager, bm25_mgr: BM25Manager):
        self.faiss_mgr = faiss_mgr
        self.bm25_mgr = bm25_mgr

    def query(self, query_text: str, top_k: int = 5):
        faiss_results = self.faiss_mgr.query(query_text, top_k)
        bm25_results = self.bm25_mgr.query(query_text, top_k)

        merged_results = {res["text"]: res for res in faiss_results}
        for res in bm25_results:
            if res["text"] in merged_results:
                merged_results[res["text"]]["score"] += res["score"]
            else:
                merged_results[res["text"]] = res

        return sorted(merged_results.values(), key=lambda x: x["score"], reverse=True)
