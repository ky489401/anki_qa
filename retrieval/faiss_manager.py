# retrieval/faiss_manager.py

import os
import pickle
from typing import Dict
from typing import List

import faiss
import numpy as np
from openai import OpenAI

from config import OPENAI_API_KEY

# Set OpenAI API key (Replace 'your-api-key' with your actual key)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# OpenAI client
client = OpenAI()


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

    def get_embeddings(
        self, texts: List[str], model: str = "text-embedding-3-small"
    ) -> np.ndarray:
        """
        Generates embeddings for a list of input texts using OpenAI's embedding model.

        Args:
            texts (List[str]): A list of text strings to generate embeddings for.
            model (str, optional): The name of the OpenAI embedding model to use.
                                   Defaults to "text-embedding-3-small".

        Returns:
            np.ndarray: A NumPy array of shape (len(texts), embedding_dim) containing
                        the generated embeddings.
        """
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

    def save_index(self, index_path: str, metadata_path: str) -> None:
        """
        Saves the FAISS index and associated metadata to files.

        Args:
            index_path (str): Path to save the FAISS index.
            metadata_path (str): Path to save document metadata.
        """
        if self.index is None:
            raise ValueError("No index built. Call build_index() first.")

        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.docs, f)

    def load_index(self, index_path: str, metadata_path: str) -> None:
        """
        Loads the FAISS index and document metadata from files.

        Args:
            index_path (str): Path to the FAISS index file.
            metadata_path (str): Path to the document metadata file.
        """
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.docs = pickle.load(f)

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
        query_embedding = query_embedding / np.linalg.norm(
            query_embedding, axis=1, keepdims=True
        )

        D, I = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(D[0], I[0]):
            doc_data = self.docs[idx]
            results.append(
                {
                    "id": doc_data["id"],
                    "text": doc_data["text"],
                    "meta": doc_data["meta"],
                    "score": float(score),
                }
            )
        return results


if __name__ == "__main__":

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

    # Save index
    faiss_mgr.save_index("artifacts/faiss.index", "artifacts/metadata.pkl")

    # Load index
    new_faiss_mgr = FAISSManager()
    new_faiss_mgr.load_index("artifacts/faiss.index", "artifacts/metadata.pkl")

    # Query
    results = new_faiss_mgr.query("supervised learning", top_k=2)
    for res in results:
        print(res)
