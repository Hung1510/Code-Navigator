"""
callgraph.py — a lightweight symbol + call graph over the repo.

This is the step from *search* to *code intelligence*. Text retrieval answers
"where is JWT handled"; a call graph answers structural questions that keywords
can't: who calls this, what does this call, where is this defined.

We reuse the tree-sitter parse (no embeddings involved) to extract, per file:
  - definitions  (functions, methods, classes) with qualified names + spans
  - call sites    (foo(), obj.method(), pkg.Fn(), new Widget(), macros)

Then we resolve each call's callee name against the global definition table.

Honest boundaries — resolution is name-based, not type-aware:
  - Same-file calls resolve exactly (a definition in the caller's file wins).
  - Cross-file calls resolve by name; if several files define that name, we
    report *all* candidates rather than guessing one. That's the ambiguity a
    real type resolver (LSP) would settle — we surface it instead of hiding it.
  - Calls with no matching definition (stdlib, third-party) are marked external.
This is deliberately a heuristic graph: useful and honest, not a compiler.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import LANG_BY_EXT, iter_source_files
from .treesitter import _NODES, _node_name, _parser, ts_supports

# Which node types are calls, per language.
CALL_NODES = {
    "python": {"call"},
    "javascript": {"call_expression", "new_expression"},
    "typescript": {"call_expression", "new_expression"},
    "tsx": {"call_expression", "new_expression"},
    "rust": {"call_expression", "macro_invocation"},
    "java": {"method_invocation", "object_creation_expression"},
    "csharp": {"invocation_expression", "object_creation_expression"},
    "cpp": {"call_expression"},
    "go": {"call_expression"},
}

# For member/attribute expressions, which field holds the final called name.
_MEMBER_FIELD = {
    "attribute": "attribute",             # python  obj.method
    "member_expression": "property",      # js/ts   obj.method
    "selector_expression": "field",       # go      pkg.Fn
    "member_access_expression": "name",   # c#      obj.Method
    "field_expression": "field",          # rust/cpp obj.method
}
_SCOPED = {"scoped_identifier", "qualified_identifier", "scoped_type_identifier"}
_IDENT = {"identifier", "type_identifier", "field_identifier", "property_identifier"}


@dataclass
class Symbol:
    id: int
    name: str          # unqualified, e.g. "invalidate"
    qualified: str     # e.g. "SessionManager.invalidate"
    kind: str          # "callable" or "container"
    path: str
    start_line: int
    end_line: int

    def locator(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "qualified": self.qualified,
                "kind": self.kind, "path": self.path,
                "start_line": self.start_line, "end_line": self.end_line}


@dataclass
class CallEdge:
    caller_id: int         # symbol id of enclosing def, or -1 for module scope
    callee: str            # unqualified callee name as written
    path: str              # file where the call appears
    line: int
    resolved: list[int]    # symbol ids this callee resolves to (0, 1, or many)

    def to_dict(self) -> dict:
        return {"caller_id": self.caller_id, "callee": self.callee,
                "path": self.path, "line": self.line, "resolved": self.resolved}


# ---------------------------------------------------------------------------
# Test detection
#
# There is no language-agnostic way to know what a test is, so we use each
# ecosystem's own naming convention — the same signal the test runners use.
# It's a heuristic and it will miss things (a test suite in an oddly-named
# directory, a Rust `#[cfg(test)] mod tests` inside a production file is caught
# by the qualified-name rule, but an integration test driven over HTTP is not).
# ---------------------------------------------------------------------------

_TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs", "testing", "e2e"}
_TEST_SUFFIXES = (
    "_test.py", "_test.go", "_test.rs", "_test.java", "_test.ts", "_test.js",
    ".test.ts", ".test.tsx", ".test.js", ".test.jsx",
    ".spec.ts", ".spec.tsx", ".spec.js", ".spec.jsx",
    "test.java", "tests.cs", "test.cs", "_spec.rb",
)


def is_test_path(path: str) -> bool:
    """True if this file is, by its ecosystem's convention, a test file."""
    p = path.replace("\\", "/").lower()
    parts = p.split("/")
    if any(seg in _TEST_DIRS for seg in parts[:-1]):
        return True
    fname = parts[-1]
    if fname.startswith("test_") or fname.startswith("test-"):
        return True
    return fname.endswith(_TEST_SUFFIXES)


def _looks_like_test(qualified: str) -> bool:
    """Catches in-file test modules (Rust `mod tests`) and test-named callables
    that live in a production file."""
    q = qualified.lower()
    return (q.startswith("test") or ".test" in q
            or "tests." in q or q.endswith(".tests"))


@dataclass
class ImpactNode:
    key: tuple             # ("sym", id) or ("mod", path) — identity in the BFS
    symbol_id: int         # -1 for a module-scope call site
    qualified: str
    path: str
    call_line: int         # where the call into the previous hop appears
    depth: int             # hops from the changed symbol
    uncertain: bool        # reached via at least one ambiguous name resolution
    is_test: bool

    def to_dict(self) -> dict:
        return {"symbol_id": self.symbol_id, "qualified": self.qualified,
                "path": self.path, "line": self.call_line, "depth": self.depth,
                "uncertain": self.uncertain, "is_test": self.is_test}


@dataclass
class ImpactResult:
    root: Symbol
    nodes: list[ImpactNode]       # the nodes we report (may be filtered)
    parent: dict                  # node key -> parent key (BFS tree, for chains)
    max_depth: int
    truncated: bool               # stopped at max_depth with frontier still growing
    all_nodes: list[ImpactNode] = field(default_factory=list)
    # ^ every node the BFS reached. `nodes` may be a filtered view of this (see
    #   tests_for), but chains still have to walk through the nodes we filtered
    #   OUT — a test reaches the symbol *through* production code, and that
    #   intermediate hop is the entire explanation. Losing it loses the "why".

    def __post_init__(self):
        if not self.all_nodes:
            self.all_nodes = self.nodes

    def chain(self, node: ImpactNode) -> list[str]:
        """The call path from `node` down to the changed symbol. This is the
        'why' — the reason a test or caller is implicated at all."""
        names: list[str] = [node.qualified]
        key = self.parent.get(node.key)
        index = {n.key: n for n in self.all_nodes}
        while key is not None:
            if key[0] == "sym" and key[1] == self.root.id:
                names.append(self.root.qualified)
                break
            n = index.get(key)
            if n is None:
                break
            names.append(n.qualified)
            key = self.parent.get(key)
        return names

    @property
    def uncertain_count(self) -> int:
        return sum(1 for n in self.nodes if n.uncertain)

    def to_dict(self) -> dict:
        return {
            "symbol": self.root.to_dict(),
            "max_depth": self.max_depth,
            "truncated": self.truncated,
            "uncertain_count": self.uncertain_count,
            "impacted": [dict(n.to_dict(), chain=self.chain(n)) for n in self.nodes],
        }


@dataclass
class CallGraph:
    symbols: list[Symbol] = field(default_factory=list)
    edges: list[CallEdge] = field(default_factory=list)

    def by_name(self) -> dict[str, list[int]]:
        d: dict[str, list[int]] = defaultdict(list)
        for s in self.symbols:
            d[s.name].append(s.id)
        return d

    # -- queries -----------------------------------------------------------

    def find(self, query: str) -> list[Symbol]:
        """Resolve a user-supplied symbol name to matching definitions.
        Accepts a qualified name (SessionManager.invalidate) or a bare name."""
        exact = [s for s in self.symbols if s.qualified == query]
        if exact:
            return exact
        return [s for s in self.symbols
                if s.name == query or s.qualified.endswith("." + query)]

    def callers(self, sid: int) -> list[tuple[int, str, int]]:
        """(caller_symbol_id or -1, path, line) for each site calling sid."""
        return [(e.caller_id, e.path, e.line) for e in self.edges if sid in e.resolved]

    def callees(self, sid: int) -> list[CallEdge]:
        """Call edges originating inside symbol sid."""
        return [e for e in self.edges if e.caller_id == sid]

    # -- reverse reachability ----------------------------------------------

    def _reverse(self) -> dict[int, list[CallEdge]]:
        """callee symbol id -> the edges that call it."""
        rev: dict[int, list[CallEdge]] = defaultdict(list)
        for e in self.edges:
            for sid in e.resolved:
                rev[sid].append(e)
        return rev

    def impact(self, sid: int, max_depth: int = 3) -> "ImpactResult":
        """Everything that transitively reaches `sid` — i.e. what could break
        if you change its signature or behaviour.

        Breadth-first over reversed call edges. Each frontier is one hop further
        from the change site, so depth is a proxy for blast radius.

        Two honesty mechanics, because this is where a name-based graph is most
        tempted to lie:

          * An edge whose callee name matched several definitions is `ambiguous`
            — we followed it, but a type-aware resolver might not have. Ambiguity
            is *inherited*: a node reached through any ambiguous hop is itself
            marked uncertain, because its presence in this set is conditional on
            a guess. Error compounds with depth; we make that visible instead of
            flattening it into a confident-looking tree.

          * Module-level call sites (caller_id == -1) are real impact — a script
            calling the function at import time is affected too — but they have
            no enclosing symbol to recurse into, so they're leaves.
        """
        rev = self._reverse()
        root = self.symbols[sid]

        # Node keys: ("sym", id) for definitions, ("mod", path) for module scope.
        start: tuple = ("sym", sid)
        seen: set[tuple] = {start}
        nodes: list[ImpactNode] = []
        parent: dict[tuple, tuple | None] = {start: None}

        index: dict[tuple, ImpactNode] = {}
        frontier: list[tuple] = [start]
        for depth in range(1, max_depth + 1):
            nxt: list[tuple] = []
            for key in frontier:
                if key[0] != "sym":
                    continue                      # module-scope nodes are leaves
                for e in rev.get(key[1], []):
                    ambiguous = len(e.resolved) > 1
                    # Inherit uncertainty from the path we arrived by.
                    parent_node = index.get(key)
                    if parent_node is not None and parent_node.uncertain:
                        ambiguous = True

                    if e.caller_id >= 0:
                        nkey: tuple = ("sym", e.caller_id)
                        sym = self.symbols[e.caller_id]
                        qualified, npath = sym.qualified, sym.path
                    else:
                        nkey = ("mod", e.path)
                        qualified, npath = f"<module {e.path}>", e.path

                    if nkey in seen:
                        continue
                    seen.add(nkey)
                    parent[nkey] = key
                    node = ImpactNode(
                        key=nkey,
                        symbol_id=e.caller_id,
                        qualified=qualified,
                        path=npath,
                        call_line=e.line,
                        depth=depth,
                        uncertain=ambiguous,
                        is_test=is_test_path(npath) or _looks_like_test(qualified),
                    )
                    nodes.append(node)
                    index[nkey] = node
                    nxt.append(nkey)
            frontier = nxt
            if not frontier:
                break

        # Did we stop because we ran out of graph, or because we hit max_depth?
        truncated = bool(frontier)
        return ImpactResult(root=root, nodes=nodes, parent=parent,
                            max_depth=max_depth, truncated=truncated,
                            all_nodes=nodes)

    def tests_for(self, sid: int, max_depth: int = 4) -> "ImpactResult":
        """Which tests exercise this symbol.

        This is impact analysis with a filter: a test covers `sid` if some test
        function transitively calls it. Deeper default depth than `impact`,
        because tests usually reach production code through a layer or two of
        setup helpers — but the same ambiguity caveats apply, more so.

        Absence of a result is weak evidence. A test that drives the symbol via
        HTTP, a DI container, reflection, or a mock will not appear here — no
        static call edge exists. Read this as 'tests that demonstrably call it',
        never as 'the complete set of tests that cover it'.
        """
        full = self.impact(sid, max_depth=max_depth)
        tests = [n for n in full.nodes if n.is_test]
        return ImpactResult(root=full.root, nodes=tests, parent=full.parent,
                            max_depth=max_depth, truncated=full.truncated,
                            all_nodes=full.nodes)

    def symbol_at(self, path: str, start_line: int, end_line: int) -> "Symbol | None":
        """The innermost symbol in `path` overlapping [start_line, end_line].
        Maps a retrieved chunk back to its definition so we can follow edges."""
        best, best_size = None, None
        for s in self.symbols:
            if s.path != path:
                continue
            if s.end_line < start_line or s.start_line > end_line:
                continue
            size = s.end_line - s.start_line
            if best_size is None or size < best_size:
                best, best_size = s, size
        return best


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _callee_name(node) -> str | None:
    if node is None:
        return None
    t = node.type
    if t in _IDENT:
        return node.text.decode("utf-8", "ignore")
    if t in _MEMBER_FIELD:
        f = node.child_by_field_name(_MEMBER_FIELD[t])
        if f is not None:
            return f.text.decode("utf-8", "ignore")
    # scoped (a::b::c) or anything else: take the last identifier descendant.
    last = None
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in _IDENT or n.type.endswith("identifier"):
            last = n
        stack.extend(n.named_children)
    return last.text.decode("utf-8", "ignore") if last else None


def _call_target(call_node):
    t = call_node.type
    if t == "method_invocation":            # java
        return call_node.child_by_field_name("name")
    if t == "macro_invocation":             # rust
        return call_node.child_by_field_name("macro")
    if t == "object_creation_expression":   # java new
        return call_node.child_by_field_name("type")
    if t == "new_expression":               # js new
        return call_node.child_by_field_name("constructor")
    return call_node.child_by_field_name("function")


def _iter_defs(root, cfg):
    """Yield (qualified, kind, start_line0, end_line0, start_byte, end_byte)."""
    out = []

    def visit(node, scope):
        for child in node.named_children:
            t = child.type
            if t in cfg.wrappers:
                visit(child, scope)
                continue
            if t in cfg.containers or t in cfg.callables:
                name = _node_name(child) or t
                qualified = ".".join(scope + [name])
                kind = "container" if t in cfg.containers else "callable"
                out.append((qualified, kind, child.start_point[0], child.end_point[0],
                            child.start_byte, child.end_byte))
                visit(child, scope + [name])
            else:
                visit(child, scope)

    visit(root, [])
    return out


def _iter_calls(root, call_types):
    """Yield (callee_name, line0, call_start_byte)."""
    out = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in call_types:
            name = _callee_name(_call_target(n))
            if name:
                out.append((name, n.start_point[0], n.start_byte))
        stack.extend(n.named_children)
    return out


def _enclosing(entries: list[tuple[int, int, int]], call_byte: int) -> int:
    """Smallest def span containing the call byte; -1 if module-level."""
    best_id, best_size = -1, None
    for sid, sb, eb in entries:
        if sb <= call_byte < eb:
            size = eb - sb
            if best_size is None or size < best_size:
                best_id, best_size = sid, size
    return best_id


def build_call_graph(repo_path: Path, progress=lambda *a: None) -> CallGraph:
    repo = repo_path.resolve()
    graph = CallGraph()
    file_entries: dict[str, list[tuple[int, int, int]]] = {}   # path -> (sid, sb, eb)
    file_calls: dict[str, list[tuple[str, int, int]]] = {}

    files = [p for p in iter_source_files(repo)
             if ts_supports(LANG_BY_EXT.get(p.suffix.lower(), ""))]
    progress(f"Building call graph from {len(files)} files ...")

    # Pass 1: parse each file once; collect definitions (symbols) and raw calls.
    for p in files:
        lang = LANG_BY_EXT[p.suffix.lower()]
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(p.relative_to(repo))
        tree = _parser(lang).parse(raw.encode("utf-8", "ignore"))
        entries = []
        for qualified, kind, s0, e0, sb, eb in _iter_defs(tree.root_node, _NODES[lang]):
            sid = len(graph.symbols)
            graph.symbols.append(Symbol(sid, qualified.split(".")[-1], qualified,
                                        kind, rel, s0 + 1, e0 + 1))
            entries.append((sid, sb, eb))
        file_entries[rel] = entries
        file_calls[rel] = _iter_calls(tree.root_node, CALL_NODES[lang])

    # Pass 2: resolve callee names against the global symbol table.
    by_name = graph.by_name()
    for rel, calls in file_calls.items():
        entries = file_entries[rel]
        for callee, line0, cb in calls:
            caller_id = _enclosing(entries, cb)
            ids = by_name.get(callee, [])
            same = [i for i in ids if graph.symbols[i].path == rel]
            resolved = same if same else ids       # prefer same-file, else all candidates
            graph.edges.append(CallEdge(caller_id, callee, rel, line0 + 1, resolved))

    progress(f"Call graph: {len(graph.symbols)} symbols, {len(graph.edges)} call sites.")
    return graph


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

GRAPH_FILE = "graph.json"


def save_graph(graph: CallGraph, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / GRAPH_FILE).write_text(json.dumps({
        "symbols": [s.to_dict() for s in graph.symbols],
        "edges": [e.to_dict() for e in graph.edges],
    }))


def load_graph(index_dir: Path) -> CallGraph | None:
    fp = index_dir / GRAPH_FILE
    if not fp.exists():
        return None
    data = json.loads(fp.read_text())
    g = CallGraph()
    g.symbols = [Symbol(**s) for s in data["symbols"]]
    g.edges = [CallEdge(**e) for e in data["edges"]]
    return g
