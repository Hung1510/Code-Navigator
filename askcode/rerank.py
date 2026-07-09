"""
rerank.py — second-stage cross-encoder re-ranking.

First-stage retrieval (vector + BM25, fused) is cheap because it compares
*separately-computed* representations: the query vector never actually meets
the document vector, they're just dotted together. That's fast enough to run
over the whole index, but it leaves quality on the table.

A cross-encoder is the opposite trade-off. It feeds the query and one document
through a transformer *together*, so their tokens attend to each other, and
returns a single relevance score. Much more accurate, far too slow to run over
thousands of chunks. So the standard move is two-stage:

    hybrid retrieve top-N  (cheap, wide)  ->  cross-encoder re-rank  (precise, narrow)

We only re-rank the ~30 candidates the first stage already liked, then keep the
best k. Backed by fastembed's ONNX cross-encoder (no PyTorch), consistent with
the rest of the stack. It's optional: if the model isn't available, retrieval
falls back to the fused order.
"""

from __future__ import annotations

from typing import Protocol

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    def rerank(self, query: str, docs: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """fastembed cross-encoder. Higher score = more relevant."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.model_name = model_name
        self._enc = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        return list(self._enc.rerank(query, docs))


def load_reranker(model_name: str = DEFAULT_RERANK_MODEL, progress=None):
    """Build a reranker, or return None if fastembed's cross-encoder isn't
    available (so callers can degrade gracefully to the fused order)."""
    try:
        return CrossEncoderReranker(model_name)
    except Exception as e:                      # import error or model load failure
        if progress:
            progress(f"(reranker unavailable: {e}; using fused order)")
        return None
