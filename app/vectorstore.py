"""Thin wrapper around the Chroma vector store used for policy retrieval."""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app import config

_embeddings = None
_store = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _embeddings


def get_store() -> Chroma:
    """Return the persisted Chroma collection, creating the client if needed."""
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(config.CHROMA_DIR),
        )
    return _store


def reset_store() -> None:
    """Drop the cached client so the next get_store() reopens the (possibly rebuilt) persist dir."""
    global _store
    _store = None


def similarity_search(query: str, k: int = config.TOP_K):
    """Return top-k (Document, score) pairs for a query."""
    return get_store().similarity_search_with_relevance_scores(query, k=k)
