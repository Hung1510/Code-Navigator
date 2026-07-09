"""
eval.py — measure retrieval quality instead of asserting it.

Every quality claim about this tool ("hybrid beats vector", "rerank helps") is
only worth what you can measure. This harness turns them into numbers:
recall@1, recall@k, and MRR across every retrieval mode, on your own repo.

The hard part of any eval is ground truth. We get most of it for free:

  name -> code (auto, all languages)
    Turn each function's name into a natural query ("validateRefreshToken" ->
    "validate refresh token"); the gold answer is that function's own chunk.
    A real NL->code retrieval task with zero hand-labeling. Its one caveat: the
    name's words also appear in the code, which mildly favors keyword search —
    so read it as "can the system map a concept phrase to the right function",
    and use a curated set (below) for the cleanest semantic comparison.

  curated (manual, JSONL)
    Realistic questions you write, each with the file/lines that should answer
    it. One object per line:
      {"query": "where is JWT refresh handled?", "path": "svc/auth.py",
       "start_line": 40, "end_line": 58}

A hit counts as correct when it's in the same file and its line range overlaps
the gold span. Metrics are averaged over the dataset, per mode.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from .embedder import Embedder
from .lexical import BM25Index
from .query import _base_ranking, load_store
from .rerank import Reranker
from .store import SearchHit, VectorStore

_SUBWORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

# The mode matrix. (label, base retrieval mode, use cross-encoder rerank).
DEFAULT_SPECS = [
    ("vector", "vector", False),
    ("lexical", "lexical", False),
    ("hybrid", "hybrid", False),
]
RERANK_SPEC = ("hybrid+rerank", "hybrid", True)


@dataclass
class EvalItem:
    query: str
    gold_path: str
    gold_start: int
    gold_end: int
    source: str = "name"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _identifier_words(name: str) -> list[str]:
    parts = _SUBWORD.findall(name)
    return [p.lower() for p in parts] if parts else [name.lower()]


def build_name_dataset(repo_path: Path, max_items: int = 200,
                       min_words: int = 2, seed: int = 0) -> list[EvalItem]:
    """Auto NL->code queries from callable names, using the call graph's
    symbol table as ground truth. Needs no embeddings or API key to build."""
    from .callgraph import build_call_graph

    graph = build_call_graph(repo_path)
    items: list[EvalItem] = []
    for s in graph.symbols:
        if s.kind != "callable":
            continue
        words = _identifier_words(s.name)
        # Skip trivial or uninformative names (get, run, __init__, main, a).
        if len(words) < min_words or sum(len(w) for w in words) < 6:
            continue
        if s.name.lower() in {"__init__", "main", "new", "default", "toString"}:
            continue
        items.append(EvalItem(" ".join(words), s.path, s.start_line, s.end_line, "name"))

    random.Random(seed).shuffle(items)
    return items[:max_items]


def load_curated(path: Path) -> list[EvalItem]:
    """Load a hand-written JSONL eval set."""
    items: list[EvalItem] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        o = json.loads(line)
        items.append(EvalItem(o["query"], o["path"],
                              int(o["start_line"]), int(o["end_line"]), "curated"))
    return items


def build_scaffold(repo_path: Path, max_items: int = 30, min_lines: int = 4,
                   seed: int = 0) -> list[dict]:
    """Emit curated-JSONL template rows for the repo's meatier callables, so you
    can attach real questions instead of starting from a blank file. Prefers
    longer functions (more likely to be worth asking about) and spreads picks
    across files."""
    from .callgraph import build_call_graph

    graph = build_call_graph(repo_path)
    cands = [s for s in graph.symbols
             if s.kind == "callable" and (s.end_line - s.start_line) >= min_lines
             and s.name.lower() not in {"__init__", "main"}]
    # Sort by size desc, then round-robin across files for variety.
    cands.sort(key=lambda s: -(s.end_line - s.start_line))
    by_file: dict[str, list] = {}
    for s in cands:
        by_file.setdefault(s.path, []).append(s)
    rows, files = [], list(by_file)
    seen: set = set()
    i = 0
    while len(rows) < max_items and any(by_file.values()):
        f = files[i % len(files)]
        i += 1
        if by_file[f]:
            s = by_file[f].pop(0)
            key = (s.name, s.start_line, s.end_line)   # collapse built/duplicate copies
            if key in seen:
                continue
            seen.add(key)
            rows.append({"query": f"TODO: ask about {s.qualified}",
                         "path": s.path, "start_line": s.start_line,
                         "end_line": s.end_line, "_symbol": s.qualified})
    return rows


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _matches(hit: SearchHit, item: EvalItem) -> bool:
    return (hit.path == item.gold_path
            and not (hit.end_line < item.gold_start or hit.start_line > item.gold_end))


def _first_match_rank(hits: list[SearchHit], item: EvalItem) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if _matches(h, item):
            return rank
    return None


def _rank_for(store: VectorStore, bm25: BM25Index, item: EvalItem, embedder: Embedder,
              base_mode: str, use_rerank: bool, reranker: Reranker | None,
              k: int, rerank_pool: int) -> int | None:
    depth = max(k * 4, 20, rerank_pool if use_rerank else 0)
    ordered = _base_ranking(store, item.query, embedder, base_mode, depth, bm25=bm25)
    if use_rerank and reranker is not None and ordered:
        cand = [i for i, _ in ordered[:rerank_pool]]
        texts = store.doc_texts()
        scores = reranker.rerank(item.query, [texts[i][:1600] for i in cand])
        cand = [i for i, _ in sorted(zip(cand, scores), key=lambda p: -p[1])]
        top = cand[:k]
    else:
        top = [i for i, _ in ordered[:k]]
    hits = [store.hit_for_index(i) for i in top]
    return _first_match_rank(hits, item)


def evaluate(repo_path: Path, dataset: list[EvalItem], embedder: Embedder,
             k: int = 10, reranker: Reranker | None = None,
             progress=lambda *a: None) -> dict:
    """Run every mode over the dataset; return per-mode recall@1 / recall@k / MRR."""
    store = load_store(repo_path)
    bm25 = BM25Index(store.doc_texts())      # built once, reused for all queries
    specs = list(DEFAULT_SPECS)
    if reranker is not None:
        specs.append(RERANK_SPEC)
    rerank_pool = max(k * 3, 30)

    agg = {label: {"hit1": 0, "hitk": 0, "rr": 0.0} for label, _, _ in specs}
    n = len(dataset)
    for idx, item in enumerate(dataset):
        for label, base_mode, use_rerank in specs:
            rank = _rank_for(store, bm25, item, embedder, base_mode, use_rerank,
                             reranker, k, rerank_pool)
            if rank is not None:
                agg[label]["hitk"] += 1
                agg[label]["rr"] += 1.0 / rank
                if rank == 1:
                    agg[label]["hit1"] += 1
        if (idx + 1) % 25 == 0:
            progress(f"  scored {idx + 1}/{n}")

    report = {"n": n, "k": k, "modes": {}}
    for label, _, _ in specs:
        a = agg[label]
        report["modes"][label] = {
            "recall@1": a["hit1"] / n if n else 0.0,
            f"recall@{k}": a["hitk"] / n if n else 0.0,
            "mrr": a["rr"] / n if n else 0.0,
        }
    return report


def format_report(report: dict) -> str:
    k = report["k"]
    lines = [f"\nEvaluation over {report['n']} queries (k={k})\n",
             f"{'mode':<16}{'recall@1':>10}{f'recall@{k}':>12}{'MRR':>10}",
             "-" * 48]
    for label, m in report["modes"].items():
        lines.append(f"{label:<16}{m['recall@1']:>10.3f}"
                     f"{m[f'recall@{k}']:>12.3f}{m['mrr']:>10.3f}")
    return "\n".join(lines) + "\n"
