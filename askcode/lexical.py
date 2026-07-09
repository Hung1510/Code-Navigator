"""
lexical.py — keyword search over code, via hand-rolled BM25.

Embeddings are great at "where is auth handled" but they can miss exact
identifier lookups: ask "what calls refreshToken" and a semantic model may
rank a paraphrase above the actual call site. Lexical search nails those,
because it matches tokens directly. Running both and fusing (see query.py) is
strictly better than either alone.

Two pieces:
  - tokenize(): a *code-aware* tokenizer. It splits snake_case and camelCase
    into subwords, so `refreshToken`, `refresh_token`, and the query "refresh
    token" all share tokens. Without this, keyword search on code is nearly
    useless.
  - BM25Index: classic Okapi BM25 over an inverted index. IDF weights rare
    terms higher; the length-normalization term stops long chunks from winning
    just by being long. We only score chunks that share a term with the query
    (that's what the inverted index buys us), so it stays fast.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

# Matches runs of letters/digits; snake_case splits here for free because "_"
# isn't included.
_WORD = re.compile(r"[A-Za-z0-9]+")
# Splits a single alnum run into camelCase / PascalCase / digit subwords:
#   "getHTTPResponse" -> get, HTTP, Response ; "refreshToken" -> refresh, Token
_SUBWORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for run in _WORD.findall(text):
        low = run.lower()
        tokens.append(low)                    # keep the whole token (exact-ish match)
        parts = _SUBWORD.findall(run)
        if len(parts) > 1:                    # add subwords for camelCase runs
            tokens.extend(p.lower() for p in parts)
    return tokens


class BM25Index:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(doc, tf)]
        df: Counter = Counter()

        for doc_id, doc in enumerate(docs):
            toks = tokenize(doc)
            self.doc_len.append(len(toks))
            for term, freq in Counter(toks).items():
                self.postings[term].append((doc_id, freq))
                df[term] += 1

        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        # BM25 IDF with the +0.5 smoothing; the outer (1 + ...) keeps it non-negative.
        self.idf = {
            term: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to k (doc_id, score) pairs, highest score first."""
        if self.N == 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, freq in self.postings[term]:
                dl = self.doc_len[doc_id]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += idf * (freq * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[:k]
