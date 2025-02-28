"""
tests/test_retrievers.py

Unit tests for the retrieval managers:
 - FAISSManager for vector retrieval.
 - BM25Manager for keyword retrieval.
 - HybridRetriever for merging both results.
"""

import unittest

from retrieval.bm25_manager import BM25Manager
from retrieval.faiss_manager import FAISSManager


class TestRetrievers(unittest.TestCase):
    def setUp(self):
        # Create dummy chunked documents.
        self.docs = [
            {
                "id": "doc1_chunk_0",
                "text": "Supervised learning uses labeled data.",
                "meta": {},
            },
            {
                "id": "doc2_chunk_0",
                "text": "Unsupervised learning does not use labels.",
                "meta": {},
            },
            {
                "id": "doc3_chunk_0",
                "text": "Reinforcement learning involves rewards.",
                "meta": {},
            },
        ]

    def test_faiss_manager(self):
        faiss_mgr = FAISSManager()
        faiss_mgr.build_index(self.docs)
        results = faiss_mgr.anki_query("supervised learning", top_k=2)
        self.assertLessEqual(len(results), 2)
        for res in results:
            self.assertIn("id", res)
            self.assertIn("score", res)

    def test_bm25_manager(self):
        bm25_mgr = BM25Manager(self.docs)
        results = bm25_mgr.anki_query("learning", top_k=2)
        self.assertLessEqual(len(results), 2)
        for res in results:
            self.assertIn("text", res)
            self.assertIn("score", res)

    def test_hybrid_retriever(self):
        from retrieval.faiss_manager import FAISSManager
        from retrieval.bm25_manager import BM25Manager
        from retrieval.hybrid_retriever import HybridRetriever

        faiss_mgr = FAISSManager()
        faiss_mgr.build_index(self.docs)
        bm25_mgr = BM25Manager(self.docs)
        hybrid_retriever = HybridRetriever(faiss_mgr, bm25_mgr)
        results = hybrid_retriever.anki_query("learning", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        for res in results:
            self.assertIn("text", res)
            self.assertIn("score", res)


if __name__ == "__main__":
    unittest.main()
