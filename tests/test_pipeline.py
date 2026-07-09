"""
Offline end-to-end test. Uses a deterministic hash-based fake embedder so the
full chunk -> embed -> store -> retrieve loop runs with no model download and
no API key. Run: python -m pytest -q   (or just python tests/test_pipeline.py)
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from codenavigator.chunker import chunk_repo
from codenavigator.index import build_index
from codenavigator.query import retrieve
from codenavigator.store import VectorStore


class FakeEmbedder:
    """Bag-of-words hashed into a fixed-dim vector. Deterministic, offline.
    Good enough that lexically-overlapping text lands near the query."""
    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().replace("_", " ").split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.array([self._vec(t) for t in texts], dtype=np.float32)

    def embed_query(self, text):
        return self._vec(text)


def _make_sample_repo(root: Path):
    (root / "auth").mkdir(parents=True)
    (root / "auth" / "tokens.py").write_text(
        "import time\n\n"
        "def refresh_token(user, session):\n"
        "    '''Rotate an expired JWT refresh token.'''\n"
        "    new = issue_jwt(user)\n"
        "    session.store(new)\n"
        "    return new\n\n"
        "def issue_jwt(user):\n"
        "    return sign({'sub': user.id, 'exp': time.time() + 3600})\n"
    )
    (root / "billing.py").write_text(
        "def charge_card(customer, amount):\n"
        "    gateway.capture(customer.card, amount)\n"
        "    return Receipt(customer, amount)\n"
    )


def test_end_to_end(tmp_path: Path):
    repo = tmp_path / "sample"
    repo.mkdir()
    _make_sample_repo(repo)

    chunks = chunk_repo(repo)
    symbols = " ".join(c.symbol for c in chunks)
    # Backend-agnostic: tree-sitter emits "refresh_token", heuristic "def refresh_token".
    assert "refresh_token" in symbols
    assert "charge_card" in symbols

    emb = FakeEmbedder()
    build_index(repo, emb, progress=lambda *a: None)

    hits = retrieve(repo, "where is the jwt refresh token handled", emb, k=3)
    assert hits, "expected at least one hit"
    top = hits[0]
    # The refresh-token chunk should win over the billing chunk.
    assert "tokens.py" in top.path
    assert top.start_line >= 1

    # Persistence round-trips.
    reloaded = VectorStore.load(repo / ".codenavigator")
    assert len(reloaded) == len(chunks)
    print(f"OK: {len(chunks)} chunks, top hit = {top.locator()}")


def test_treesitter_python_nesting(tmp_path: Path):
    """Tree-sitter path: classes split into header + qualified methods,
    decorators ride with their function. Skipped if grammars aren't installed."""
    from codenavigator.treesitter import ts_supports
    if not ts_supports("python"):
        return  # grammar wheel not installed; heuristic path covers this repo

    src = (
        "import os\n"
        "\n"
        "@route\n"
        "def refresh_token(user):\n"
        "    return issue(user)\n"
        "\n"
        "class SessionManager:\n"
        "    def invalidate(self, sid):\n"
        "        self.db.delete(sid)\n"
    )
    f = tmp_path / "app.py"
    f.write_text(src)

    from codenavigator.chunker import chunk_file
    chunks = chunk_file(f, tmp_path)
    symbols = {c.symbol for c in chunks}
    assert "SessionManager.invalidate" in symbols   # qualified method name
    assert any("refresh_token" in s for s in symbols)
    # The decorator line must be inside the refresh_token chunk.
    rt = next(c for c in chunks if "refresh_token" in c.symbol)
    assert "@route" in rt.text
    print("tree-sitter OK:", sorted(symbols))


class CountingEmbedder(FakeEmbedder):
    """FakeEmbedder that records how many documents it embedded, so a test can
    assert that an incremental re-index only touches changed files."""
    model_name = "fake-counting"

    def __init__(self):
        self.embed_count = 0

    def embed_documents(self, texts):
        self.embed_count += len(texts)
        return super().embed_documents(texts)


def test_incremental_indexing(tmp_path: Path):
    from codenavigator.index import build_index
    from codenavigator.query import retrieve
    from codenavigator.store import VectorStore

    repo = tmp_path / "sample"
    repo.mkdir()
    _make_sample_repo(repo)   # auth/tokens.py + billing.py

    emb = CountingEmbedder()

    # --- first build: everything is embedded ---
    build_index(repo, emb, progress=lambda *a: None)
    first_count = emb.embed_count
    assert first_count > 0
    total_chunks = len(VectorStore.load(repo / ".codenavigator"))
    assert total_chunks == first_count

    # --- re-run with no changes: nothing re-embedded ---
    emb.embed_count = 0
    build_index(repo, emb, progress=lambda *a: None)
    assert emb.embed_count == 0, "unchanged repo should embed nothing"

    # --- modify ONE file: only its chunks re-embedded ---
    emb.embed_count = 0
    (repo / "billing.py").write_text(
        "def charge_card(customer, amount):\n"
        "    gateway.capture(customer.card, amount)\n"
        "    log_charge(customer, amount)\n"          # <- changed body
        "    return Receipt(customer, amount)\n\n"
        "def log_charge(customer, amount):\n"          # <- new function
        "    audit.write(customer.id, amount)\n"
    )
    build_index(repo, emb, progress=lambda *a: None)
    assert 0 < emb.embed_count < first_count, "only the changed file should re-embed"
    # The new function is now retrievable.
    hits = retrieve(repo, "audit log a charge", emb, k=5)
    assert any("log_charge" in h.symbol for h in hits)

    # --- delete a file: its chunks vanish, nothing re-embedded ---
    emb.embed_count = 0
    (repo / "auth" / "tokens.py").unlink()
    build_index(repo, emb, progress=lambda *a: None)
    assert emb.embed_count == 0, "a deletion embeds nothing"
    store = VectorStore.load(repo / ".codenavigator")
    assert not any("tokens.py" in p for p in store.paths()), "deleted file's chunks remain"
    print("incremental OK: first=%d, delta stayed small, deletion clean" % first_count)


def test_tokenizer_splits_identifiers():
    from codenavigator.lexical import tokenize
    assert set(tokenize("refreshToken")) >= {"refresh", "token"}
    assert set(tokenize("refresh_token")) >= {"refresh", "token"}
    assert set(tokenize("getHTTPResponse")) >= {"get", "http", "response"}


def test_bm25_exact_identifier():
    from codenavigator.lexical import BM25Index
    docs = [
        "def issue_jwt(user): return sign(user)",
        "def refreshToken(user): return issue_jwt(user)",
        "def charge_card(customer, amount): gateway.capture(amount)",
    ]
    bm = BM25Index(docs)
    ranked = bm.search("refreshToken", 3)
    assert ranked and ranked[0][0] == 1          # the refreshToken def wins
    ranked2 = bm.search("charge the customer card", 3)
    assert ranked2 and ranked2[0][0] == 2        # billing def wins


def test_rrf_fusion():
    from codenavigator.query import _rrf
    # doc 5 appears near the top of both lists -> should fuse to first.
    order = _rrf([[5, 1, 2], [3, 5, 4]])
    assert order[0] == 5


def test_hybrid_modes(tmp_path: Path):
    from codenavigator.index import build_index
    from codenavigator.query import retrieve

    repo = tmp_path / "sample"
    repo.mkdir()
    _make_sample_repo(repo)
    emb = FakeEmbedder()
    build_index(repo, emb, progress=lambda *a: None)

    for mode in ("vector", "lexical", "hybrid"):
        hits = retrieve(repo, "where is the jwt refresh token handled", emb, k=3, mode=mode)
        assert hits, f"mode {mode} returned nothing"

    # Lexical mode pins an exact identifier lookup.
    hits = retrieve(repo, "issue_jwt", emb, k=3, mode="lexical")
    assert any("tokens.py" in h.path for h in hits)

    # Hybrid surfaces the refresh handler at or near the top.
    hits = retrieve(repo, "rotate an expired refresh token", emb, k=3, mode="hybrid")
    assert any("tokens.py" in h.path for h in hits)
    print("hybrid OK: vector/lexical/hybrid all return hits")


class FakeReranker:
    """Scores by token overlap with the query — deterministic, offline. Enough
    to prove the two-stage plumbing reorders candidates by the reranker."""
    def rerank(self, query: str, docs: list[str]) -> list[float]:
        from codenavigator.lexical import tokenize
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(d)))) for d in docs]


def test_reranker_reorders(tmp_path: Path):
    from codenavigator.index import build_index
    from codenavigator.query import retrieve

    repo = tmp_path / "sample"
    repo.mkdir()
    _make_sample_repo(repo)
    emb = FakeEmbedder()
    build_index(repo, emb, progress=lambda *a: None)

    # Without a reranker: fused order. With the fake reranker: candidates are
    # reordered by query-token overlap. Both should return the JWT chunk on top
    # for this query, proving the reranker path runs end-to-end.
    q = "rotate the jwt refresh token for a user"
    base = retrieve(repo, q, emb, k=5, mode="hybrid")
    reranked = retrieve(repo, q, emb, k=5, mode="hybrid", reranker=FakeReranker())
    assert base and reranked
    assert "tokens.py" in reranked[0].path
    # Reranked scores come from the reranker (token-overlap counts), not RRF.
    assert reranked[0].score >= 1.0
    print("reranker OK: top =", reranked[0].locator())


def test_json_serialization():
    from codenavigator.store import SearchHit
    h = SearchHit(0.5, "auth/tokens.py", "python", 3, 8, "refresh_token", "def refresh_token(): ...")
    d = h.to_dict()
    assert d["path"] == "auth/tokens.py" and d["locator"].startswith("auth/tokens.py:3-8")
    json.dumps(d)  # must be JSON-serializable


def _make_graph_repo(root: Path):
    (root / "svc").mkdir(parents=True)
    (root / "svc" / "tokens.py").write_text(
        "def issue_jwt(user):\n    return sign(user)\n\n"
        "def verify(token):\n    return check(token)\n\n"
        "def sign(user):\n    return str(user)\n"
    )
    (root / "svc" / "auth.py").write_text(
        "from .tokens import issue_jwt, verify\n\n"
        "class AuthService:\n"
        "    def login(self, user):\n"
        "        return issue_jwt(user)\n\n"
        "    def refresh(self, token):\n"
        "        if verify(token):\n"
        "            return issue_jwt(token)\n"
    )
    (root / "svc" / "api.py").write_text(
        "from .auth import AuthService\n\n"
        "def handle_login(req):\n"
        "    return AuthService().login(req.user)\n"
    )


def test_call_graph(tmp_path: Path):
    from codenavigator.callgraph import build_call_graph, load_graph, save_graph

    repo = tmp_path / "repo"
    repo.mkdir()
    _make_graph_repo(repo)
    g = build_call_graph(repo)

    # defs: qualified names present
    quals = {s.qualified for s in g.symbols}
    assert {"issue_jwt", "AuthService", "AuthService.login", "AuthService.refresh"} <= quals

    # callers of issue_jwt: both login and refresh (cross-file), resolved.
    tgt = next(s for s in g.symbols if s.qualified == "issue_jwt")
    caller_names = {g.symbols[cid].qualified for cid, _, _ in g.callers(tgt.id) if cid >= 0}
    assert {"AuthService.login", "AuthService.refresh"} <= caller_names

    # callees of AuthService.login: issue_jwt resolves cross-file.
    login = next(s for s in g.symbols if s.qualified == "AuthService.login")
    callee_targets = {g.symbols[i].qualified for e in g.callees(login.id) for i in e.resolved}
    assert "issue_jwt" in callee_targets

    # callers of AuthService.login: handle_login (method call on a fresh instance)
    caller_of_login = {g.symbols[cid].qualified for cid, _, _ in g.callers(login.id) if cid >= 0}
    assert "handle_login" in caller_of_login

    # persistence round-trips.
    save_graph(g, repo / ".codenavigator")
    g2 = load_graph(repo / ".codenavigator")
    assert len(g2.symbols) == len(g.symbols) and len(g2.edges) == len(g.edges)
    print("call graph OK:", len(g.symbols), "symbols,", len(g.edges), "edges")


def test_call_graph_ambiguity(tmp_path: Path):
    """Two files defining the same name -> a cross-file call reports BOTH
    candidates rather than guessing. Same-file calls stay unambiguous."""
    from codenavigator.callgraph import build_call_graph

    repo = tmp_path / "repo"
    (repo / "a").mkdir(parents=True)
    (repo / "b").mkdir(parents=True)
    (repo / "a" / "x.py").write_text("def save(v):\n    return v\n")
    (repo / "b" / "y.py").write_text("def save(v):\n    return v\n")
    (repo / "caller.py").write_text("def go():\n    return save(1)\n")
    g = build_call_graph(repo)

    go = next(s for s in g.symbols if s.qualified == "go")
    edges = g.callees(go.id)
    save_edge = next(e for e in edges if e.callee == "save")
    # No same-file 'save' -> both cross-file defs are reported as candidates.
    assert len(save_edge.resolved) == 2
    print("ambiguity OK: 'save' -> 2 candidates reported")


def test_graph_expansion(tmp_path: Path):
    """Retrieval finds the login code; graph expansion then pulls in issue_jwt
    (which login calls, in a different file) even though the query doesn't
    resemble issue_jwt at all. That's the graph⨯retrieval win."""
    from codenavigator.index import build_index
    from codenavigator.query import retrieve

    repo = tmp_path / "repo"
    repo.mkdir()
    _make_graph_repo(repo)                 # AuthService.login calls issue_jwt (cross-file)
    emb = FakeEmbedder()
    build_index(repo, emb, progress=lambda *a: None)   # builds vector store + call graph

    query = "AuthService login method"     # matches login, not issue_jwt
    base = retrieve(repo, query, emb, k=3, mode="hybrid")
    base_syms = " ".join(h.symbol for h in base)
    assert "AuthService.login" in base_syms
    assert "issue_jwt" not in base_syms    # not retrieved on its own merits

    expanded = retrieve(repo, query, emb, k=3, mode="hybrid", graph_expand=True)
    exp_syms = " ".join(h.symbol for h in expanded)
    assert "issue_jwt" in exp_syms                       # pulled in via the graph
    assert any("called by" in h.symbol for h in expanded)  # and annotated
    assert len(expanded) > len(base)
    print("graph expansion OK:", [h.symbol for h in expanded if "called by" in h.symbol])


def test_eval_harness(tmp_path: Path):
    from codenavigator.index import build_index
    from codenavigator.eval import build_name_dataset, evaluate

    repo = tmp_path / "repo"
    repo.mkdir()
    _make_graph_repo(repo)      # AuthService.login/refresh, issue_jwt, verify, sign
    emb = FakeEmbedder()
    build_index(repo, emb, progress=lambda *a: None)

    dataset = build_name_dataset(repo, max_items=50, min_words=2)
    assert dataset, "expected some name->code eval items"
    # 'issue_jwt' -> query 'issue jwt', gold = tokens.py definition
    assert any(it.query == "issue jwt" and "tokens.py" in it.gold_path for it in dataset)

    report = evaluate(repo, dataset, emb, k=5)
    assert report["n"] == len(dataset)
    for label in ("vector", "lexical", "hybrid"):
        m = report["modes"][label]
        for key, val in m.items():
            assert 0.0 <= val <= 1.0, f"{label}.{key} out of range: {val}"
    # Lexical should find name-based queries well (words appear in the code).
    assert report["modes"]["lexical"]["recall@5"] >= 0.5
    print("eval harness OK:", {k: round(v["recall@5"], 2)
                               for k, v in report["modes"].items()})


def test_scaffold_and_gate(tmp_path: Path):
    from codenavigator.eval import build_scaffold
    from codenavigator.cli import _parse_thresholds

    repo = tmp_path / "repo"
    repo.mkdir()
    _make_graph_repo(repo)
    rows = build_scaffold(repo, max_items=10, min_lines=1)
    assert rows and all({"query", "path", "start_line", "end_line"} <= set(r) for r in rows)
    assert all(r["query"].startswith("TODO") for r in rows)
    # dedupe: no two rows share the same (symbol, span)
    keys = [(r["_symbol"], r["start_line"], r["end_line"]) for r in rows]
    assert len(keys) == len(set(keys))

    th = _parse_thresholds("recall@10=0.8, mrr=0.5")
    assert th == {"recall@10": 0.8, "mrr": 0.5}


def test_ignore_file(tmp_path: Path):
    from codenavigator.chunker import iter_source_files

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "dist").mkdir()
    (repo / "src" / "app.py").write_text("def real(): pass\n")
    (repo / "dist" / "app.py").write_text("def duplicate(): pass\n")     # built copy
    (repo / "src" / "bundle.min.js").write_text("var x=1")               # minified

    # Without an ignore file, dist/ IS excluded by SKIP_DIRS, but let's use a
    # dir that isn't in SKIP_DIRS to prove the ignore file itself works.
    (repo / "built").mkdir()
    (repo / "built" / "app.py").write_text("def built_dup(): pass\n")
    (repo / ".codenavigatorignore").write_text("built/\n*.min.js\n")

    files = {p.relative_to(repo).as_posix() for p in iter_source_files(repo)}
    assert "src/app.py" in files
    assert "built/app.py" not in files          # excluded by 'built/'
    assert "src/bundle.min.js" not in files      # excluded by '*.min.js'
    print("ignore file OK:", sorted(files))


if __name__ == "__main__":
    import tempfile
    for t in (test_end_to_end, test_treesitter_python_nesting,
              test_incremental_indexing, test_hybrid_modes,
              test_reranker_reorders, test_call_graph, test_call_graph_ambiguity,
              test_graph_expansion, test_eval_harness, test_scaffold_and_gate,
              test_ignore_file):
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
    test_tokenizer_splits_identifiers()
    test_bm25_exact_identifier()
    test_rrf_fusion()
    test_json_serialization()
    print("passed")
