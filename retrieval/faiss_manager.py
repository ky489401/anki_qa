# retrieval/faiss_manager.py

from config import client
import faiss
import numpy as np
from typing import List, Dict


class FAISSManager:
    """
    FAISSManager uses OpenAI's embedding model (e.g., text-embedding-ada-002)
    to generate embeddings for a list of documents, builds a FAISS index,
    and performs semantic retrieval.

    Requirements:
      - Set your OpenAI API key in your environment (OPENAI_API_KEY)
      - pip install openai faiss-cpu numpy
    """

    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.model_name = model_name
        self.index = None  # FAISS index
        self.docs = []  # List of document dictionaries


    def get_embeddings(self, texts, model="text-embedding-3-small"):
        embeddings = client.embeddings.create(input=texts, model=model)
        embeddings = np.array([x.embedding for x in embeddings.data])
        return embeddings

    def build_index(self, chunked_docs: List[Dict]) -> None:
        """
        Builds a FAISS index from the provided chunked documents.

        Args:
            chunked_docs (List[Dict]): Each document should have keys:
                - "id": unique identifier.
                - "text": text content.
                - "meta": additional metadata.
        """
        self.docs = chunked_docs
        texts = [doc["text"] for doc in chunked_docs]
        embeddings = self.get_embeddings(texts)

        # Normalize embeddings (L2 normalization) to enable cosine similarity via inner product.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        dim = embeddings.shape[1]
        # Create a FAISS index using inner product similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Given a query, generates its embedding, and returns the top_k similar documents.

        Args:
            query_text (str): The query string.
            top_k (int): Number of top documents to retrieve.

        Returns:
            List[Dict]: List of dictionaries with keys "id", "text", "meta", and "score".
        """
        query_embedding = self.get_embeddings([query_text])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        D, I = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(D[0], I[0]):
            doc_data = self.docs[idx]
            results.append({
                "id": doc_data["id"],
                "text": doc_data["text"],
                "meta": doc_data["meta"],
                "score": float(score)
            })
        return results

if __name__ == '__main__':

    docs = [
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

    faiss_mgr = FAISSManager()
    faiss_mgr.build_index(docs)
    results = faiss_mgr.query("supervised learning", top_k=2)