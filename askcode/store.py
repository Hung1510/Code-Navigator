"""
store.py — a vector store you can actually see through.

A production tool would drop in Qdrant / LanceDB here. For v1 we hand-roll it,
because the whole point is to understand what a vector DB *is*:

    - a matrix of unit vectors  (the "index")   -> stored as a .npy file
    - a row of metadata per vector              -> stored in SQLite
    - "search" == one matrix-vector dot product, then argsort

Because vectors are L2-normalized (see embedder.py), cosine similarity is just
the dot product. Search over ~50k chunks is a few milliseconds of numpy — you
do not need a vector database until you're well past that.

The class exposes add / search / persist / load. Keep this interface and you
can later replace the body with `import lancedb` without touching index.py or
query.py. That's the seam.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchHit:
    score: float
    path: str
    lang: str
    start_line: int
    end_line: int
    symbol: str
    text: str

    def locator(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line} ({self.symbol})"

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 6),
            "path": self.path, "lang": self.lang,
            "start_line": self.start_line, "end_line": self.end_line,
            "symbol": self.symbol, "locator": self.locator(), "text": self.text,
        }


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self._meta: list[dict] = []

    # -- build ---------------------------------------------------------------

    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> None:
        if vectors.shape[0] != len(metadatas):
            raise ValueError("vectors and metadatas length mismatch")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
        self._meta.extend(metadatas)

    # -- query ---------------------------------------------------------------

    def search(self, query_vec: np.ndarray, k: int = 8) -> list[SearchHit]:
        """Vector-only convenience: cosine top-k as SearchHits."""
        return [self.hit_for_index(i, score) for i, score in self.search_ranked(query_vec, k)]

    def search_ranked(self, query_vec: np.ndarray, k: int = 8) -> list[tuple[int, float]]:
        """Return up to k (meta_index, cosine) pairs, highest first.

        Returning indices (not built hits) lets a caller fuse this ranking with
        another retriever's ranking over the same index space — that's what
        hybrid search needs.
        """
        if len(self._meta) == 0:
            return []
        # cosine == dot product on unit vectors. One matmul over the whole index.
        scores = self._vectors @ query_vec.astype(np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]      # O(n) top-k, unordered
        top = top[np.argsort(-scores[top])]            # sort just those k
        return [(int(i), float(scores[i])) for i in top]

    def hit_for_index(self, i: int, score: float = 0.0) -> SearchHit:
        m = self._meta[i]
        return SearchHit(
            score=score, path=m["path"], lang=m["lang"],
            start_line=m["start_line"], end_line=m["end_line"],
            symbol=m["symbol"], text=m["text"],
        )

    def doc_texts(self) -> list[str]:
        """Per-chunk text for the lexical index, aligned to meta index. Includes
        path + symbol so identifier names in the locator are searchable too."""
        return [f"{m['path']} {m['symbol']}\n{m['text']}" for m in self._meta]

    def chunks_in_span(self, path: str, start_line: int, end_line: int,
                       limit: int = 3) -> list[SearchHit]:
        """Chunks in `path` whose line range overlaps [start_line, end_line].
        Used by graph expansion to fetch a linked symbol's already-indexed code."""
        out = []
        for i, m in enumerate(self._meta):
            if m["path"] != path:
                continue
            if m["end_line"] < start_line or m["start_line"] > end_line:
                continue
            out.append(self.hit_for_index(i))
            if len(out) >= limit:
                break
        return out

    def __len__(self) -> int:
        return len(self._meta)

    def remove_paths(self, paths: set[str]) -> int:
        """Drop every chunk whose source file is in `paths`. Returns count removed.

        This is what makes incremental indexing possible: when a file changes or
        is deleted, we remove its old chunks, then (for a change) add fresh ones.
        Rebuilding the vector matrix by fancy-indexing the surviving rows is a
        few milliseconds even at tens of thousands of chunks.
        """
        if not self._meta or not paths:
            return 0
        keep = [i for i, m in enumerate(self._meta) if m["path"] not in paths]
        removed = len(self._meta) - len(keep)
        if removed:
            self._vectors = (self._vectors[keep] if keep
                             else np.empty((0, self.dim), dtype=np.float32))
            self._meta = [self._meta[i] for i in keep]
        return removed

    def paths(self) -> set[str]:
        """Distinct source paths currently in the index."""
        return {m["path"] for m in self._meta}

    # -- persistence ---------------------------------------------------------

    def persist(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", self._vectors)
        db = sqlite3.connect(index_dir / "meta.db")
        db.execute("DROP TABLE IF EXISTS chunks")
        db.execute(
            "CREATE TABLE chunks (rowid INTEGER PRIMARY KEY, path TEXT, lang TEXT, "
            "start_line INT, end_line INT, symbol TEXT, text TEXT)"
        )
        db.executemany(
            "INSERT INTO chunks (rowid, path, lang, start_line, end_line, symbol, text) "
            "VALUES (?,?,?,?,?,?,?)",
            [(i, m["path"], m["lang"], m["start_line"], m["end_line"], m["symbol"], m["text"])
             for i, m in enumerate(self._meta)],
        )
        db.commit()
        db.close()
        (index_dir / "manifest.json").write_text(
            json.dumps({"dim": self.dim, "count": len(self._meta)}, indent=2)
        )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        manifest = json.loads((index_dir / "manifest.json").read_text())
        store = cls(dim=manifest["dim"])
        store._vectors = np.load(index_dir / "vectors.npy")
        db = sqlite3.connect(index_dir / "meta.db")
        rows = db.execute(
            "SELECT path, lang, start_line, end_line, symbol, text FROM chunks ORDER BY rowid"
        ).fetchall()
        db.close()
        store._meta = [
            {"path": r[0], "lang": r[1], "start_line": r[2],
             "end_line": r[3], "symbol": r[4], "text": r[5]}
            for r in rows
        ]
        return store
