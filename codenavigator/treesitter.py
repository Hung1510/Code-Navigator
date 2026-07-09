"""
treesitter.py — structure-aware chunking backed by real parse trees.

This is the upgrade over the regex heuristic in chunker.py. Instead of guessing
where a definition starts from the shape of a line, we parse the file into a
concrete syntax tree and walk it. That fixes the things regex gets wrong:
nested defs, multi-line signatures, decorators/`export` prefixes, and blocks
that share a line with their braces.

Chunking strategy — "smallest meaningful unit, with a breadcrumb":
  - A *callable* (function / method) is emitted whole. Nested closures stay
    with their parent, because they're part of that function's meaning.
  - A *container* (class / impl / struct / namespace ...) is emitted as a
    header chunk (its declaration + docstring + fields, up to the first nested
    definition), and each method inside becomes its own chunk. So a hit is
    "SessionManager.invalidate", not 400 lines of class.
  - Enclosing scope names ride along as a dotted breadcrumb in the symbol, so
    retrieval and the LLM both know where a chunk lives.

Grammars ship as precompiled PyPI wheels (tree-sitter-python, -rust, ...), so
there's no runtime download and no C compiler needed. If a grammar for a file's
language isn't installed, we return None and chunker.py falls back to regex.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Grammar loading. Each entry: lang label -> (module name, factory attribute).
# The module is imported lazily so CodeNavigator works even if a grammar isn't present.
# ---------------------------------------------------------------------------

_GRAMMARS = {
    "python":     ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx":        ("tree_sitter_typescript", "language_tsx"),
    "rust":       ("tree_sitter_rust", "language"),
    "java":       ("tree_sitter_java", "language"),
    "csharp":     ("tree_sitter_c_sharp", "language"),
    "cpp":        ("tree_sitter_cpp", "language"),
    "go":         ("tree_sitter_go", "language"),
}


# Per-language node classification.
#   containers -> emit a header + recurse for members
#   callables  -> emit whole, do not recurse
#   wrappers   -> transparent nodes whose span should extend the child's start
#                 (decorators, `export`), recursed through but never emitted
class _Cfg:
    __slots__ = ("containers", "callables", "wrappers")

    def __init__(self, containers, callables, wrappers=()):
        self.containers = set(containers)
        self.callables = set(callables)
        self.wrappers = set(wrappers)


_NODES = {
    "python": _Cfg(
        containers={"class_definition"},
        callables={"function_definition"},
        wrappers={"decorated_definition"},
    ),
    "javascript": _Cfg(
        containers={"class_declaration"},
        callables={"function_declaration", "method_definition", "generator_function_declaration"},
        wrappers={"export_statement"},
    ),
    "rust": _Cfg(
        # struct/enum/union have no nested callables, so they're emitted whole.
        containers={"impl_item", "trait_item", "mod_item",
                    "struct_item", "enum_item", "union_item"},
        callables={"function_item"},
        wrappers={},
    ),
    "java": _Cfg(
        containers={"class_declaration", "interface_declaration", "enum_declaration"},
        callables={"method_declaration", "constructor_declaration"},
        wrappers={},
    ),
    "csharp": _Cfg(
        containers={"namespace_declaration", "class_declaration", "struct_declaration",
                    "interface_declaration", "enum_declaration"},
        callables={"method_declaration", "constructor_declaration"},
        wrappers={},
    ),
    "cpp": _Cfg(
        containers={"class_specifier", "struct_specifier", "namespace_definition",
                    "enum_specifier"},
        callables={"function_definition"},
        wrappers={},
    ),
    "go": _Cfg(
        containers={},
        callables={"function_declaration", "method_declaration"},
        wrappers={},
    ),
}
# typescript / tsx reuse the javascript config plus interfaces/enums.
_NODES["typescript"] = _Cfg(
    containers={"class_declaration", "interface_declaration", "enum_declaration",
                "internal_module", "module"},
    callables={"function_declaration", "method_definition", "abstract_method_signature"},
    wrappers={"export_statement"},
)
_NODES["tsx"] = _NODES["typescript"]


def ts_supports(lang: str) -> bool:
    """True if we have a grammar mapping and the wheel is importable."""
    if lang not in _GRAMMARS:
        return False
    try:
        _load_language(lang)
        return True
    except Exception:
        return False


@lru_cache(maxsize=None)
def _load_language(lang: str):
    from tree_sitter import Language
    mod_name, attr = _GRAMMARS[lang]
    mod = __import__(mod_name)
    return Language(getattr(mod, attr)())


@lru_cache(maxsize=None)
def _parser(lang: str):
    from tree_sitter import Parser
    return Parser(_load_language(lang))


# ---------------------------------------------------------------------------
# Name extraction
# ---------------------------------------------------------------------------

_ID_TYPES = {"identifier", "type_identifier", "property_identifier",
             "field_identifier", "name"}


def _node_name(node) -> str | None:
    # Most grammars expose a "name" field.
    n = node.child_by_field_name("name")
    if n is not None:
        return n.text.decode("utf-8", "ignore")
    # Go: `type T struct {...}` -> name lives on the inner type_spec.
    if node.type == "type_declaration":
        for c in node.named_children:
            if c.type == "type_spec":
                nm = c.child_by_field_name("name")
                if nm is not None:
                    return nm.text.decode("utf-8", "ignore")
    # Rust impl / others: first identifier-ish child.
    for c in node.named_children:
        if c.type in _ID_TYPES:
            return c.text.decode("utf-8", "ignore")
    return None


def _arrow_name(node) -> str | None:
    """JS/TS: `const foo = () => {}` -> return 'foo' if this lexical_declaration
    binds an arrow function or function expression."""
    for decl in node.named_children:
        if decl.type != "variable_declarator":
            continue
        value = decl.child_by_field_name("value")
        if value is not None and value.type in ("arrow_function", "function_expression"):
            nm = decl.child_by_field_name("name")
            return nm.text.decode("utf-8", "ignore") if nm else "<anon>"
    return None


# ---------------------------------------------------------------------------
# Walk + emit
# ---------------------------------------------------------------------------

def ts_chunk_file(path: Path, root: Path, raw: str, lang: str):
    """Return list[Chunk] for this file, or None to signal 'use the fallback'."""
    from .chunker import Chunk, MAX_CHUNK_LINES, _window  # local import avoids cycle

    if not ts_supports(lang):
        return None

    cfg = _NODES[lang]
    src = raw.encode("utf-8", "ignore")
    lines = raw.splitlines()
    rel = str(path.relative_to(root))
    tree = _parser(lang).parse(src)

    emitted: list[Chunk] = []

    def classify(node) -> str | None:
        t = node.type
        if t in cfg.wrappers:
            return "wrapper"
        if t in cfg.containers:
            return "container"
        if t in cfg.callables:
            return "callable"
        if t == "lexical_declaration" and _arrow_name(node):
            return "callable"
        if t == "type_declaration" and lang == "go":
            return "container"  # struct/interface decl; emitted whole (no members inside)
        return None

    def start_with_wrappers(node) -> int:
        n = node
        while n.parent is not None and n.parent.type in cfg.wrappers:
            n = n.parent
        return n.start_point[0]  # 0-indexed line

    def first_inner_boundary_line(node) -> int | None:
        best = None
        stack = list(node.named_children)
        while stack:
            c = stack.pop()
            if classify(c) in ("container", "callable"):
                ln = c.start_point[0]
                best = ln if best is None else min(best, ln)
            else:
                stack.extend(c.named_children)
        return best

    def make_chunk(start0: int, end0: int, symbol: str) -> None:
        """start0/end0 are 0-indexed inclusive line numbers."""
        block = lines[start0:end0 + 1]
        if not "".join(block).strip():
            return
        # Re-window anything too long to embed as a single unit.
        if end0 - start0 + 1 > MAX_CHUNK_LINES:
            for ws, we, _ in _window(block, 80, 15):
                sub = block[ws:we]
                if not "".join(sub).strip():
                    continue
                emitted.append(Chunk(rel, lang, start0 + ws + 1, start0 + we,
                                     f"{symbol} (part)", "\n".join(sub)))
        else:
            emitted.append(Chunk(rel, lang, start0 + 1, end0 + 1, symbol,
                                 "\n".join(block)))

    def name_of(node) -> str:
        if node.type == "lexical_declaration":
            return _arrow_name(node) or "<fn>"
        return _node_name(node) or node.type

    def visit(node, scope: list[str]) -> None:
        for child in node.named_children:
            kind = classify(child)
            if kind == "wrapper":
                visit(child, scope)
            elif kind == "container":
                nm = name_of(child)
                qualified = ".".join(scope + [nm])
                s = start_with_wrappers(child)
                inner = first_inner_boundary_line(child)
                header_end = (inner - 1) if inner is not None else child.end_point[0]
                make_chunk(s, header_end, f"{qualified} (decl)")
                if inner is not None:
                    visit(child, scope + [nm])
            elif kind == "callable":
                nm = name_of(child)
                qualified = ".".join(scope + [nm])
                s = start_with_wrappers(child)
                make_chunk(s, child.end_point[0], qualified)
            else:
                visit(child, scope)

    root = tree.root_node
    visit(root, [])

    if not emitted:
        # No definitions found (script / data / unusual file) -> window fallback.
        return None

    # Capture top-level "loose" code that isn't inside any definition: imports,
    # module constants, config dicts, script bodies. We group contiguous
    # non-boundary direct children of the root into module-level chunks, so
    # nothing at file scope is silently dropped.
    def wraps_boundary(child) -> bool:
        return any(classify(gc) in ("container", "callable")
                   for gc in child.named_children)

    # Each maximal span of consecutive non-boundary top-level nodes becomes one
    # module-level chunk (make_chunk windows it if it's very long).
    run: tuple[int, int] | None = None
    for child in root.named_children:
        k = classify(child)
        is_boundary = k in ("container", "callable") or (k == "wrapper" and wraps_boundary(child))
        if is_boundary:
            if run is not None:
                make_chunk(run[0], run[1], "<module>")
                run = None
        else:
            s, e = child.start_point[0], child.end_point[0]
            run = (run[0], e) if run is not None else (s, e)
    if run is not None:
        make_chunk(run[0], run[1], "<module>")

    emitted.sort(key=lambda c: (c.start_line, c.end_line))
    return emitted
