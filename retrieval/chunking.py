# retrieval/chunking.py

from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap  # Slide with overlap

    return chunks


def chunk_document(doc_id: str, full_text: str) -> List[Dict]:
    """
    Splits a document into multiple chunked segments with metadata.
    """
    chunked_texts = chunk_text(full_text)
    return [
        {"id": f"{doc_id}_chunk_{i}", "text": chunk, "meta": {"source_doc_id": doc_id}}
        for i, chunk in enumerate(chunked_texts)
    ]
