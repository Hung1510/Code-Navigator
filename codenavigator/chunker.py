"""
chunker.py — turn a repo into retrieval-friendly chunks.

The single most important decision in a code-RAG tool is *how you split*.
Split too coarse (whole file) and retrieval returns 500 irrelevant lines.
Split too fine (every line) and you lose the context that makes an answer useful.

Strategy here: **structure-aware chunking**. We split on top-level
definitions (functions / classes / methods) so each chunk is a semantically
whole unit — "the JWT refresh handler", not "lines 40-60". When a file has no
recognizable structure (config, markdown, plain text) we fall back to a
sliding line-window with overlap so nothing is lost at boundaries.

This is a heuristic (regex + brace/indent tracking), deliberately dependency
-free so v1 runs today. The natural upgrade is tree-sitter, which gives you a
real parse tree per language — swap `iter_definitions` for a tree-sitter walk
and everything downstream is unchanged.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

IGNORE_FILE = ".codenavigatorignore"

# ---------------------------------------------------------------------------
# What we index, and what we skip.
# ---------------------------------------------------------------------------

# Map file extension -> language label (used for prompts + brace vs indent mode)
LANG_BY_EXT = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".rs": "rust",
    ".java": "java",
    ".cs": "csharp",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".h": "cpp",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".md": "markdown", ".mdx": "markdown",
    ".txt": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".sql": "sql",
}

# Indent-delimited languages use a different definition detector than
# brace-delimited ones.
INDENT_LANGS = {"python"}

# Directories that never contain source worth indexing.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "target", "dist", "build",
    "__pycache__", ".venv", "venv", "env", ".idea", ".vscode", ".mypy_cache",
    ".pytest_cache", ".next", ".turbo", "coverage", "vendor", "bin", "obj",
    ".codenavigator",  # our own index directory — never index it
}

# Definition-line patterns per family. Kept intentionally loose — we only need
# to find *boundaries*, not perfectly parse the language.
_DEF_PATTERNS = {
    "python": re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+\w+"),
    "brace": re.compile(
        r"^\s*(?:"
        r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+\w+"      # JS/TS function
        r"|(?:export\s+)?(?:abstract\s+)?class\s+\w+"                     # class
        r"|(?:pub\s+)?(?:async\s+)?fn\s+\w+"                              # Rust fn
        r"|(?:pub\s+)?(?:struct|enum|trait|impl)\s+\w+"                   # Rust types
        r"|(?:public|private|protected|internal|static|\s)+[\w<>\[\]]+\s+\w+\s*\("  # Java/C# method
        r"|(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"     # JS arrow fn
        r"|func\s+\w+"                                                     # Go func
        r")"
    ),
}

MAX_CHUNK_LINES = 120      # a chunk longer than this gets window-split
WINDOW_LINES = 80          # fallback window size
WINDOW_OVERLAP = 15        # overlap so we don't cut a thought in half


@dataclass
class Chunk:
    """One retrievable unit of code/text."""
    path: str              # repo-relative path
    lang: str
    start_line: int        # 1-indexed, inclusive
    end_line: int
    symbol: str            # best-guess name, e.g. "def refresh_token" or "<window>"
    text: str

    def header(self) -> str:
        """Human/LLM-readable locator prepended to the embedded text."""
        return f"{self.path}:{self.start_line}-{self.end_line} ({self.symbol})"

    def to_row(self) -> dict:
        return {
            "path": self.path, "lang": self.lang,
            "start_line": self.start_line, "end_line": self.end_line,
            "symbol": self.symbol, "text": self.text,
        }


def load_ignore(root: Path) -> list[str]:
    """Read `.codenavigatorignore` (gitignore-flavored) from the repo root.

    Keeps built/vendored/duplicate copies out of the index. Patterns:
      - `dist/` or `node_modules`  -> a directory of that name at any depth
      - `public/js/`               -> that path anchored to the repo root
      - `*.min.js`                 -> filename glob at any depth
      - `**/vendor/**`             -> loose glob (** matches across slashes)
    Lines starting with `#` and blank lines are ignored.
    """
    fp = root / IGNORE_FILE
    if not fp.exists():
        return []
    out = []
    for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _match_ignore(rel_posix: str, pattern: str) -> bool:
    dir_only = pattern.endswith("/")
    p = pattern.rstrip("/")
    if not p:
        return False
    if "**" in p:                                   # loose ** (crosses slashes)
        if fnmatch.fnmatch(rel_posix, p.replace("**", "*")):
            return True
    if "/" in p:                                    # anchored to repo root
        if rel_posix == p or rel_posix.startswith(p + "/"):
            return True
        if fnmatch.fnmatch(rel_posix, p) or fnmatch.fnmatch(rel_posix, p + "/*"):
            return True
        return False
    parts = rel_posix.split("/")
    if any(fnmatch.fnmatch(seg, p) for seg in parts[:-1]):   # dir component anywhere
        return True
    if not dir_only and fnmatch.fnmatch(parts[-1], p):        # filename glob
        return True
    return False


def is_ignored(rel_posix: str, patterns: list[str]) -> bool:
    return any(_match_ignore(rel_posix, pat) for pat in patterns)


def iter_source_files(root: Path) -> list[Path]:
    """Walk the repo, skipping junk dirs, binaries, huge files, and anything
    matched by a `.codenavigatorignore` at the repo root."""
    patterns = load_ignore(root)
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in LANG_BY_EXT:
            continue
        if patterns and is_ignored(p.relative_to(root).as_posix(), patterns):
            continue
        try:
            if p.stat().st_size > 1_000_000:   # skip >1MB (minified bundles, data)
                continue
        except OSError:
            continue
        out.append(p)
    return out


def _symbol_name(line: str) -> str:
    """Extract a readable symbol label from a definition line."""
    m = re.search(r"(def|class|fn|function|struct|enum|trait|impl|func)\s+(\w+)", line)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    # Java/C# method: grab the identifier before '('
    m = re.search(r"(\w+)\s*\(", line)
    return m.group(1) if m else "<block>"


def _split_by_definitions(lines: list[str], lang: str) -> list[tuple[int, int, str]]:
    """Return [(start_idx, end_idx, symbol)] spans, 0-indexed, end exclusive."""
    pat = _DEF_PATTERNS["python" if lang in INDENT_LANGS else "brace"]
    starts = [i for i, ln in enumerate(lines) if pat.match(ln)]
    if not starts:
        return []
    spans: list[tuple[int, int, str]] = []
    # Preamble before the first definition (imports, module docstring) is its own chunk.
    if starts[0] > 0:
        spans.append((0, starts[0], "<module-preamble>"))
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        spans.append((s, e, _symbol_name(lines[s])))
    return spans


def _window(lines: list[str], size: int, overlap: int) -> list[tuple[int, int, str]]:
    """Sliding line-window fallback, 0-indexed, end exclusive."""
    spans, i, n = [], 0, len(lines)
    step = max(1, size - overlap)
    while i < n:
        spans.append((i, min(i + size, n), "<window>"))
        i += step
    return spans


def chunk_file(path: Path, root: Path, prefer_treesitter: bool = True) -> list[Chunk]:
    """Chunk one file into structure-aware Chunks.

    Tries the tree-sitter backend first (accurate parse tree). If the grammar
    for this language isn't installed, or the file has no parseable structure,
    falls back to the regex heuristic below. Both paths return identical Chunk
    objects, so nothing downstream cares which ran.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    if not raw.strip():
        return []

    lang = LANG_BY_EXT.get(path.suffix.lower(), "text")

    if prefer_treesitter:
        try:
            from .treesitter import ts_chunk_file, ts_supports
            if ts_supports(lang):
                chunks = ts_chunk_file(path, root, raw, lang)
                if chunks:            # None or [] -> fall through to heuristic
                    return chunks
        except ImportError:
            pass                      # tree-sitter not installed; use heuristic

    return _heuristic_chunk_file(path, root, raw, lang)


def _heuristic_chunk_file(path: Path, root: Path, raw: str, lang: str) -> list[Chunk]:
    """Regex/indent-based fallback chunker (no dependencies)."""
    lines = raw.splitlines()
    rel = str(path.relative_to(root))

    spans = _split_by_definitions(lines, lang)
    if not spans:                      # config/markdown/text -> window it
        spans = _window(lines, WINDOW_LINES, WINDOW_OVERLAP)

    chunks: list[Chunk] = []
    for start, end, symbol in spans:
        # Re-window any definition that's too long to embed usefully.
        if end - start > MAX_CHUNK_LINES:
            for ws, we, _ in _window(lines[start:end], WINDOW_LINES, WINDOW_OVERLAP):
                block = lines[start + ws:start + we]
                if not "".join(block).strip():
                    continue
                chunks.append(Chunk(
                    rel, lang, start + ws + 1, start + we,
                    f"{symbol} (part)", "\n".join(block),
                ))
        else:
            block = lines[start:end]
            if not "".join(block).strip():
                continue
            chunks.append(Chunk(
                rel, lang, start + 1, end, symbol, "\n".join(block),
            ))
    return chunks


def chunk_repo(root: Path) -> list[Chunk]:
    """Chunk every source file under root."""
    root = root.resolve()
    all_chunks: list[Chunk] = []
    for f in iter_source_files(root):
        all_chunks.extend(chunk_file(f, root))
    return all_chunks
