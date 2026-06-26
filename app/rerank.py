"""Reranking of retrieved candidate chunks: cross-encoder score blended with embedding similarity.

A pure cross-encoder rerank was tried first (both cross-encoder/ms-marco-MiniLM-L-6-v2 and
BAAI/bge-reranker-base) and, on this small, jargon-dense, structurally uniform policy corpus,
both general-purpose rerankers sometimes buried a section that embedding similarity had ranked
clearly near the top — e.g. asking about corporate KYC requirements got the actually-correct
"§2 Corporate Customer Identification" section ranked 5th-6th out of 12 by the cross-encoder
alone, when embeddings had it essentially tied for #1. Blending keeps embedding similarity as
the dominant signal and lets the cross-encoder only nudge/break ties, rather than fully
overriding it.
"""
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app import config

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANK_MODEL)
    return _reranker


def _normalize(scores: list[float]) -> list[float]:
    """Min-max normalize to [0, 1]; if all scores are equal, treat them as uninformative (1.0)."""
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def rerank(
    query: str, candidates: list[tuple[Document, float]], weight: float = config.RERANK_WEIGHT
) -> list[tuple[Document, float]]:
    """Blend cross-encoder score with the original embedding similarity score.

    candidates: (doc, embedding_score) pairs, e.g. straight from vectorstore.similarity_search.
    Returns (doc, blended_score) pairs sorted descending.
    """
    if not candidates:
        return []

    docs = [doc for doc, _ in candidates]
    embed_scores = [score for _, score in candidates]
    cross_scores = [float(s) for s in get_reranker().predict([(query, doc.page_content) for doc in docs])]

    norm_embed = _normalize(embed_scores)
    norm_cross = _normalize(cross_scores)
    blended = [(1 - weight) * e + weight * c for e, c in zip(norm_embed, norm_cross)]

    scored = list(zip(docs, blended))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
