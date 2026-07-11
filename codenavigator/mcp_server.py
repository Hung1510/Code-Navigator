"""
mcp_server.py — CodeNavigator as an MCP context engine.

Exposes the retrieval + call-graph engine to any MCP host (Claude Desktop,
Claude Code, Cursor, Continue, VS Code Copilot Agent...). The host's model asks
a question, this server returns *precise context* — ranked chunks with
file:line citations — and the host's model does the answering.

Two design decisions worth stating, because they're the whole point:

1. WE RETURN CONTEXT, NOT ANSWERS.
   There's already a capable model on the other end of the pipe. Calling our own
   LLM would be a second, redundant inference. So `ask_codebase` hands back the
   retrieved code and lets the host reason over it. Consequence: this server
   needs NO API key, which makes it trivial to adopt.

2. EVERY TOOL IS BUDGET-AWARE.
   The value proposition over a grep-and-read-files agent loop isn't magic, it's
   *precision*: instead of the model burning many tool calls and dumping whole
   files into its context, it gets a handful of relevant, deduplicated,
   already-ranked chunks. That only holds if we stay inside a token budget, so
   every response is capped (MAX_CONTEXT_CHARS) and long chunks are truncated
   with an explicit marker plus the locator, so the model can ask for more.

Stdio transport carries JSON-RPC on stdout, so nothing here may ever print() to
stdout. All diagnostics go to stderr.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# stdout is the JSON-RPC channel — logs MUST go to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(levelname)s codenavigator-mcp: %(message)s")
log = logging.getLogger(__name__)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from .callgraph import load_graph  # noqa: E402
from .chunker import IGNORE_FILE  # noqa: E402
from .index import index_dir_for  # noqa: E402

# --- budget -----------------------------------------------------------------
MAX_CONTEXT_CHARS = int(os.environ.get("CODENAVIGATOR_MAX_CHARS", "6000"))
MAX_CHUNK_CHARS = int(os.environ.get("CODENAVIGATOR_MAX_CHUNK_CHARS", "1800"))

# The repo this server serves. Set by the host config; defaults to cwd.
REPO = Path(os.environ.get("CODENAVIGATOR_REPO", ".")).resolve()

mcp = FastMCP(
    "codenavigator",
    instructions=(
        "Local code-intelligence context engine for the repository at "
        f"{REPO}. Use search_code or ask_codebase to find relevant code by "
        "meaning (instead of reading many files); use get_definition, "
        "find_callers, and find_callees for structural questions about "
        "symbols. All results cite path:start-end so you can read further if "
        "needed. Prefer these tools over scanning the filesystem."
    ),
)

# Lazily built, then reused across calls (model load is the expensive part).
_embedder = None
_reranker = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        # Test-only hook: lets the MCP end-to-end test run without downloading
        # a ~130MB model. Never set this in real use — retrieval quality is bad.
        if os.environ.get("CODENAVIGATOR_FAKE_EMBED") == "1":
            import hashlib
            import numpy as np

            class _FakeEmbedder:
                dim = 256
                model_name = "fake-testing-only"

                def _v(self, t: str):
                    v = np.zeros(self.dim, dtype=np.float32)
                    for tok in t.lower().replace("_", " ").split():
                        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                        v[h % self.dim] += 1.0
                    n = np.linalg.norm(v)
                    return v / n if n else v

                def embed_documents(self, texts):
                    return np.array([self._v(t) for t in texts], dtype=np.float32)

                def embed_query(self, t):
                    return self._v(t)

            log.warning("using FAKE embedder (testing only)")
            _embedder = _FakeEmbedder()
            return _embedder

        from .embedder import FastEmbedder
        log.info("loading embedding model (first call only)...")
        _embedder = FastEmbedder()
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None and os.environ.get("CODENAVIGATOR_RERANK", "1") != "0":
        from .rerank import load_reranker
        _reranker = load_reranker(progress=log.info)
    return _reranker


def _ensure_index() -> str | None:
    """Auto-index on first use, and incrementally refresh when files changed.

    Incremental indexing hashes files and re-embeds only what changed, so this
    is cheap to call on every query — the user never has to remember to reindex.
    Returns an error string, or None on success.
    """
    from .index import build_index
    try:
        build_index(REPO, _get_embedder(), progress=log.info)
        return None
    except SystemExit as e:                     # e.g. no source files found
        return str(e)
    except Exception as e:
        log.exception("indexing failed")
        return f"Indexing failed: {e}"


def _fmt_hits(hits, header: str) -> str:
    """Render hits inside the char budget, with locators and truncation marks."""
    if not hits:
        return (f"{header}\nNo matching code found. Try different wording, or a "
                f"symbol name for get_definition/find_callers.")
    out = [header]
    used = len(header)
    for h in hits:
        text = h.text
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS] + f"\n... [truncated — read {h.locator()} for the rest]"
        block = f"\n\n--- {h.path}:{h.start_line}-{h.end_line}  ({h.symbol}) ---\n{text}"
        if used + len(block) > MAX_CONTEXT_CHARS:
            out.append(f"\n\n[context budget reached — {len(hits)} hits found, "
                       f"showing the top {len(out) - 1}]")
            break
        out.append(block)
        used += len(block)
    return "".join(out)


# --- tools ------------------------------------------------------------------

@mcp.tool()
def search_code(query: str, k: int = 6) -> str:
    """Search this repository for code relevant to a natural-language query.

    Use this INSTEAD of reading many files or grepping: it returns the most
    relevant code chunks, ranked, each cited as path:start-end. Good for
    "where is X handled", "how does Y work", "find the code that does Z".

    Args:
        query: What you're looking for, in plain language or as an identifier.
        k: How many chunks to return (1-15, default 6).
    """
    err = _ensure_index()
    if err:
        return err
    from .query import retrieve
    k = max(1, min(int(k), 15))
    hits = retrieve(REPO, query, _get_embedder(), k=k, mode="hybrid",
                    reranker=_get_reranker())
    return _fmt_hits(hits, f"Top {len(hits)} results for: {query!r}")


@mcp.tool()
def ask_codebase(question: str, k: int = 6) -> str:
    """Gather the context needed to answer a question about this repository.

    Like search_code, but also pulls in code that the top matches actually CALL
    (via the call graph), marked "<- called by X". Use this for "how does X
    work" questions where the implementation spans several functions or files.
    Returns context for YOU to answer from — cite the path:line locators.

    Args:
        question: The question about the codebase.
        k: How many primary chunks to retrieve before expansion (1-15, default 6).
    """
    err = _ensure_index()
    if err:
        return err
    from .query import retrieve
    k = max(1, min(int(k), 15))
    hits = retrieve(REPO, question, _get_embedder(), k=k, mode="hybrid",
                    reranker=_get_reranker(), graph_expand=True)
    return _fmt_hits(
        hits,
        f"Context for: {question!r}\n"
        f"(Chunks marked '<- called by X' were pulled in via the call graph "
        f"because X calls them. Answer from these chunks and cite path:line.)"
    )


@mcp.tool()
def get_definition(symbol: str) -> str:
    """Find where a symbol (function, method, class) is DEFINED in this repo.

    Args:
        symbol: A name like "login" or a qualified name like "AuthService.login".
    """
    err = _ensure_index()
    if err:
        return err
    graph = load_graph(index_dir_for(REPO))
    if graph is None:
        return "No call graph available. Re-index the repository."
    matches = graph.find(symbol)
    if not matches:
        return f"No definition named {symbol!r} found. Try search_code instead."
    lines = [f"{len(matches)} definition(s) of {symbol!r}:"]
    for m in matches:
        lines.append(f"  {m.qualified}  {m.locator()}  ({m.kind})")
    return "\n".join(lines)


@mcp.tool()
def find_callers(symbol: str) -> str:
    """Find every place that CALLS a symbol. Use for impact analysis:
    "what breaks if I change X", "who uses this function".

    Cross-file resolution is name-based, so if several files define the same
    name, call sites for all of them are reported.

    Args:
        symbol: A name like "issue_jwt" or "AuthService.login".
    """
    err = _ensure_index()
    if err:
        return err
    graph = load_graph(index_dir_for(REPO))
    if graph is None:
        return "No call graph available. Re-index the repository."
    matches = graph.find(symbol)
    if not matches:
        return f"No definition named {symbol!r} found."
    rows, seen = [], set()
    for tgt in matches:
        for caller_id, path, line in graph.callers(tgt.id):
            key = (caller_id, path, line)
            if key in seen:
                continue
            seen.add(key)
            who = graph.symbols[caller_id].qualified if caller_id >= 0 else "<module>"
            rows.append(f"  {who}  at {path}:{line}")
    if not rows:
        return f"{symbol!r} is defined but has no call sites in this repo."
    return f"{len(rows)} call site(s) reaching {symbol!r}:\n" + "\n".join(sorted(rows))


@mcp.tool()
def find_callees(symbol: str) -> str:
    """Find what a symbol CALLS — the functions it depends on. Use to understand
    an implementation without reading whole files.

    Args:
        symbol: A name like "login" or "AuthService.login".
    """
    err = _ensure_index()
    if err:
        return err
    graph = load_graph(index_dir_for(REPO))
    if graph is None:
        return "No call graph available. Re-index the repository."
    matches = graph.find(symbol)
    if not matches:
        return f"No definition named {symbol!r} found."
    internal, external = [], set()
    for tgt in matches:
        for e in graph.callees(tgt.id):
            if e.resolved:
                for rid in e.resolved:
                    s = graph.symbols[rid]
                    internal.append(f"  -> {s.qualified}  ({s.locator()})")
            else:
                external.add(e.callee)
    out = [f"{symbol!r} calls {len(internal)} internal symbol(s):"]
    out.extend(sorted(set(internal)))
    if external:
        out.append(f"  external/stdlib: {', '.join(sorted(external))}")
    return "\n".join(out)


def main() -> None:
    if not REPO.exists():
        log.error("CODENAVIGATOR_REPO does not exist: %s", REPO)
        sys.exit(1)
    log.info("serving repo: %s", REPO)
    log.info("context budget: %d chars (chunk cap %d)", MAX_CONTEXT_CHARS, MAX_CHUNK_CHARS)
    if not (REPO / IGNORE_FILE).exists():
        log.info("tip: add a %s to skip built/vendored copies", IGNORE_FILE)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
