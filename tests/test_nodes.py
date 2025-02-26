"""
tests/test_nodes.py

Unit tests for the node functions defined in advanced_nodes.py.
These tests verify that each node properly updates the pipeline’s state.
"""

import unittest
from advanced_nodes import (
    user_node,
    clarification_node,
    query_expansion_node,
    bm25_node,
    faiss_node,
    merge_rerank_node,
    summarization_node,
    answer_node,
    loopback_node,
    AdvancedState,
)
from unittest.mock import patch


class TestAdvancedNodes(unittest.TestCase):
    def test_user_node(self):
        state: AdvancedState = {"query": "Hello world"}
        new_state = user_node(state)
        self.assertEqual(new_state["query"], "Hello world")

    def test_clarification_node_short(self):
        state: AdvancedState = {"query": "Hi"}
        new_state = clarification_node(state)
        self.assertTrue("clarifying details" in new_state["query"])

    def test_clarification_node_clear(self):
        state: AdvancedState = {"query": "What is supervised learning?"}
        new_state = clarification_node(state)
        self.assertEqual(new_state["query"], "What is supervised learning?")

    def test_query_expansion_node(self):
        state: AdvancedState = {"query": "What is supervised learning?"}
        new_state = query_expansion_node(state)
        self.assertIn("additional context terms", new_state["expanded_query"])

    def test_bm25_node(self):
        state: AdvancedState = {
            "expanded_query": "What is supervised learning? additional context terms"
        }
        new_state = bm25_node(state)
        self.assertIn("bm25_results", new_state)
        self.assertEqual(len(new_state["bm25_results"]), 3)
        for res in new_state["bm25_results"]:
            self.assertIn("BM25 result for", res["text"])

    def test_faiss_node(self):
        state: AdvancedState = {
            "expanded_query": "What is supervised learning? additional context terms"
        }
        new_state = faiss_node(state)
        self.assertIn("faiss_results", new_state)
        self.assertEqual(len(new_state["faiss_results"]), 3)
        for res in new_state["faiss_results"]:
            self.assertIn("FAISS result for", res["text"])

    def test_merge_rerank_node(self):
        # Create dummy BM25 and FAISS results
        state: AdvancedState = {
            "bm25_results": [
                {"text": "Result A", "score": 0.8},
                {"text": "Result B", "score": 0.7},
            ],
            "faiss_results": [
                {"text": "Result A", "score": 0.9},  # duplicate text: will be merged
                {"text": "Result C", "score": 0.6},
            ],
        }
        new_state = merge_rerank_node(state)
        merged = new_state["merged_results"]
        # Expect three unique results: "Result A", "Result B", "Result C"
        self.assertEqual(len(merged), 3)
        for res in merged:
            if res["text"] == "Result A":
                self.assertAlmostEqual(res["score"], 0.8 + 0.9)

    def test_summarization_node(self):
        state: AdvancedState = {
            "merged_results": [
                {"text": "This is the first result text", "score": 1.0},
                {"text": "This is the second result text", "score": 0.9},
                {"text": "Third result", "score": 0.8},
            ]
        }
        new_state = summarization_node(state)
        summary = new_state["summary"]
        self.assertTrue(len(summary) > 0)
        self.assertTrue(summary.endswith("..."))

    def test_answer_node(self):
        state: AdvancedState = {
            "query": "What is supervised learning?",
            "summary": "Summary text...",
        }
        new_state = answer_node(state)
        self.assertIn(
            "Answer to 'What is supervised learning?'", new_state["final_answer"]
        )

    @patch("advanced_nodes.random.random", return_value=0.2)
    def test_loopback_node_loop(self, mock_random):
        state: AdvancedState = {"query": "Initial query"}
        new_state = loopback_node(state)
        self.assertIn("more details", new_state["query"])

    @patch("advanced_nodes.random.random", return_value=0.4)
    def test_loopback_node_proceed(self, mock_random):
        state: AdvancedState = {"query": "Initial query"}
        new_state = loopback_node(state)
        self.assertEqual(new_state["query"], "Initial query")


if __name__ == "__main__":
    unittest.main()
