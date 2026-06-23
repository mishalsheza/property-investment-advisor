"""RAG Tool.

Vector search over Indian real estate reports, zoning documents, and policy
notes (./data/rag_corpus). Uses ChromaDB with its built-in local embedding
function so no embedding API key is required — the only LLM provider used
anywhere in this project is Groq, and Groq has no embeddings endpoint.
"""

from __future__ import annotations

import os

import chromadb
from chromadb.utils import embedding_functions

_CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "rag_corpus")
_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "chroma_db"
)
_COLLECTION_NAME = "india_real_estate_corpus"

_client = None
_collection = None


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    os.makedirs(_PERSIST_DIR, exist_ok=True)
    _client = chromadb.PersistentClient(path=_PERSIST_DIR)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    existing = {c.name for c in _client.list_collections()}
    needs_build = _COLLECTION_NAME not in existing

    _collection = _client.get_or_create_collection(
        name=_COLLECTION_NAME, embedding_function=embedding_fn
    )

    if needs_build or _collection.count() == 0:
        _build_index(_collection)

    return _collection


def _build_index(collection) -> None:
    ids, documents, metadatas = [], [], []
    for filename in sorted(os.listdir(_CORPUS_DIR)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(_CORPUS_DIR, filename)
        with open(path) as f:
            text = f.read()
        for i, chunk in enumerate(_chunk_text(text)):
            ids.append(f"{filename}::{i}")
            documents.append(chunk)
            metadatas.append({"source": filename})
    if documents:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)


def query_rag(query: str, k: int = 4) -> list[dict[str, str]]:
    """Query the vector store, returning [{text, source}, ...]."""
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return [
        {"text": doc, "source": meta.get("source", "unknown")}
        for doc, meta in zip(documents, metadatas)
    ]
