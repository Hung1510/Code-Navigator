"""
query.py — answer a question against a built index.

Two-stage retrieval:
  Stage 1 (fast, wide): vector and/or BM25, fused with Reciprocal Rank Fusion.
  Stage 2 (precise, narrow): an optional cross-encoder re-ranks the top
    candidates from stage 1 and we keep the best k.

Modes ("vector" | "lexical" | "hybrid") select stage 1. A reranker, when
provided, always runs on top of whichever mode produced the candidates.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .embedder import Embedder
from .index import index_dir_for
from .lexical import BM25Index
from .llm import answer as llm_answer
from .rerank import Reranker
from .store import SearchHit, VectorStore

RRF_K = 60            # RRF dampening constant
RERANK_DOC_CHARS = 1600   # cap doc length fed to the cross-encoder (token budget)


def load_store(repo_path: Path) -> VectorStore:
    idx = index_dir_for(repo_path)
    if not (idx / "manifest.json").exists():
        raise SystemExit(f"No index at {idx}. Run `codenav index {repo_path}` first.")
    return VectorStore.load(idx)


def _rrf(rankings: list[list[int]], k_rrf: int = RRF_K) -> list[int]:
    """Fuse ranked lists of doc ids; each contributes 1/(k_rrf+rank)."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] += 1.0 / (k_rrf + rank + 1)
    return sorted(fused, key=lambda d: -fused[d])


def _base_ranking(store: VectorStore, question: str, embedder: Embedder,
                  mode: str, depth: int, bm25: "BM25Index | None" = None) -> list[tuple[int, float]]:
    """Stage 1: return (index, score) candidates for the chosen mode.

    bm25: a prebuilt lexical index. Eval passes one in so it isn't rebuilt for
    every query; normal retrieval leaves it None and builds on demand."""
    vec_ranked: list[tuple[int, float]] = []
    lex_ranked: list[tuple[int, float]] = []
    if mode in ("vector", "hybrid"):
        vec_ranked = store.search_ranked(embedder.embed_query(question), depth)
    if mode in ("lexical", "hybrid"):
        bm25 = bm25 or BM25Index(store.doc_texts())
        lex_ranked = bm25.search(question, depth)

    if mode == "vector":
        return vec_ranked
    if mode == "lexical":
        return lex_ranked
    # hybrid: fuse, then attach the fused RRF score for transparency
    vr = {i: r for r, (i, _) in enumerate(vec_ranked)}
    lr = {i: r for r, (i, _) in enumerate(lex_ranked)}
    order = _rrf([[i for i, _ in vec_ranked], [i for i, _ in lex_ranked]])
    out = []
    for i in order:
        s = 0.0
        if i in vr:
            s += 1.0 / (RRF_K + vr[i] + 1)
        if i in lr:
            s += 1.0 / (RRF_K + lr[i] + 1)
        out.append((i, s))
    return out


def _expand_with_graph(store: VectorStore, repo_path: Path, hits: list[SearchHit],
                       k_expand: int = 3, max_add: int = 6) -> list[SearchHit]:
    """For the top hits, follow call-graph edges to pull in the code they
    actually call. Returns extra SearchHits (annotated, deduped) to append.

    This is the graph⨯retrieval fusion: retrieval finds what a question is
    *about*; the graph adds what that code *does*, which keyword/vector search
    would miss when the callee doesn't resemble the question."""
    from .callgraph import load_graph
    from .index import index_dir_for

    graph = load_graph(index_dir_for(repo_path))
    if graph is None:
        return []

    seen = {(h.path, h.start_line, h.end_line) for h in hits}
    added: list[SearchHit] = []
    for h in hits[:k_expand]:
        sym = graph.symbol_at(h.path, h.start_line, h.end_line)
        if sym is None:
            continue
        callee_ids: list[int] = []
        for e in graph.callees(sym.id):
            callee_ids.extend(e.resolved)
        for cid in dict.fromkeys(callee_ids):        # dedupe, keep order
            if cid == sym.id:                        # skip self-recursion
                continue
            tgt = graph.symbols[cid]
            for ch in store.chunks_in_span(tgt.path, tgt.start_line, tgt.end_line, limit=2):
                key = (ch.path, ch.start_line, ch.end_line)
                if key in seen:
                    continue
                seen.add(key)
                ch.symbol = f"{ch.symbol}  \u2190 called by {sym.qualified}"
                added.append(ch)
                if len(added) >= max_add:
                    return added
    return added


def retrieve(repo_path: Path, question: str, embedder: Embedder, k: int = 8,
             mode: str = "hybrid", reranker: Reranker | None = None,
             pool: int | None = None, rerank_pool: int | None = None,
             graph_expand: bool = False) -> list[SearchHit]:
    """Return the top-k chunks for a question.

    reranker: optional cross-encoder applied to the top `rerank_pool` stage-1
              candidates (the two-stage pattern). None -> return fused order.
    graph_expand: also append code that the top hits *call*, via the call graph.
    """
    store = load_store(repo_path)
    if len(store) == 0:
        return []
    pool = pool or max(k * 4, 20)
    rerank_pool = rerank_pool or max(k * 5, 30)
    depth = max(pool, rerank_pool if reranker else 0)

    ordered = _base_ranking(store, question, embedder, mode, depth)

    if reranker is not None and ordered:
        cand = [i for i, _ in ordered[:rerank_pool]]
        texts = store.doc_texts()
        docs = [texts[i][:RERANK_DOC_CHARS] for i in cand]
        scores = reranker.rerank(question, docs)
        ranked = sorted(zip(cand, scores), key=lambda pair: -pair[1])[:k]
        hits = [store.hit_for_index(i, float(s)) for i, s in ranked]
    else:
        hits = [store.hit_for_index(i, s) for i, s in ordered[:k]]

    if graph_expand and hits:
        hits = hits + _expand_with_graph(store, repo_path, hits)
    return hits


def ask(repo_path: Path, question: str, embedder: Embedder, k: int = 8,
        mode: str = "hybrid", reranker: Reranker | None = None,
        graph_expand: bool = True, retrieve_only: bool = False) -> str:
    hits = retrieve(repo_path, question, embedder, k=k, mode=mode,
                    reranker=reranker, graph_expand=graph_expand)
    if retrieve_only:
        return "\n\n".join(f"[{h.score:.4f}] {h.locator()}\n{h.text}" for h in hits)
    return llm_answer(question, hits)
