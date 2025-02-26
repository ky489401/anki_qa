# retrieval/bm25_manager.py

from rank_bm25 import BM25Okapi
from typing import List, Dict


class BM25Manager:
    def __init__(self, chunked_docs: List[Dict]):
        self.docs = chunked_docs
        self.texts = [doc["text"] for doc in chunked_docs]
        self.bm25 = BM25Okapi([text.split() for text in self.texts])

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieves top-k BM25 matches.
        """
        scores = self.bm25.get_scores(query_text.split())
        ranked = sorted(zip(self.texts, scores), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        return [{"text": text, "score": score} for text, score in ranked]
