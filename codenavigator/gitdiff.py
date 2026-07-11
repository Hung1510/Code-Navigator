"""
gitdiff.py — turn "what I just changed" into "what I just put at risk".

The call graph can already answer *what breaks if I change `issue_jwt`*. But
nobody actually thinks that way. They think: **I touched five files — what did
I just break, and which tests should I run?** That question is the one you ask
before every commit, and answering it is what turns a query tool into a
workflow tool.

Mechanically this is a small bridge:

    git diff  ->  changed line ranges  ->  enclosing symbols  ->  impact()

The subtlety is line numbers. A diff's line numbers describe the WORKING TREE.
A symbol's span comes from the INDEX, which was built at some earlier moment.
If you edited a file after indexing, every span in that file has shifted, and
mapping "line 47" to a symbol will silently point at the wrong function — the
kind of bug that produces confident, plausible, wrong answers. So we hash the
files and refuse to guess when the index is stale. See `stale_files`.

What this cannot see (and neither can any purely static tool):
  - Behavioural changes with no signature change still ripple through callers.
    We report structural reachability, not semantic equivalence.
  - Deleted code is gone from the working tree, so a deletion-only hunk has no
    symbol in the current index to map to. We count those and say so.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .callgraph import CallGraph, Symbol
from .index import INDEX_DIRNAME

HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


class GitError(RuntimeError):
    pass


@dataclass
class FileChange:
    path: str                       # repo-relative, forward slashes
    ranges: list[tuple[int, int]]   # 1-based, inclusive, NEW-side line ranges
    deletion_only_hunks: int        # hunks that only removed lines


def _git(repo: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError as e:
        raise GitError("git is not on PATH.") from e
    if proc.returncode != 0:
        raise GitError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout


def is_git_repo(repo: Path) -> bool:
    try:
        _git(repo, "rev-parse", "--git-dir")
        return True
    except GitError:
        return False


def diff_changes(repo: Path, rev: str | None = None,
                 staged: bool = False) -> list[FileChange]:
    """Parse `git diff` into per-file changed line ranges on the NEW side.

    rev=None, staged=False -> working tree vs HEAD (what you're about to commit)
    rev=None, staged=True  -> the index vs HEAD (what you have `git add`ed)
    rev="HEAD~1"           -> that revision vs the working tree
    rev="a..b"             -> between two revisions
    """
    args = ["diff", "--unified=0", "--no-color", "--no-ext-diff",
            "--find-renames", "--diff-filter=ACMR"]
    if staged:
        args.append("--cached")
    args.append(rev if rev else "HEAD")

    out = _git(repo, *args)

    changes: dict[str, FileChange] = {}
    current: FileChange | None = None
    for line in out.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            if target == "/dev/null":
                current = None
                continue
            # strip the b/ prefix git adds
            rel = target[2:] if target.startswith(("b/", "a/")) else target
            rel = rel.replace("\\", "/")
            current = changes.setdefault(rel, FileChange(rel, [], 0))
            continue

        if current is None or not line.startswith("@@"):
            continue

        m = HUNK_RE.match(line)
        if not m:
            continue
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) is not None else 1
        if count == 0:
            # Pure deletion: nothing exists on the new side at this point.
            # The removed code is not in the index, so we cannot resolve it.
            current.deletion_only_hunks += 1
            continue
        current.ranges.append((start, start + count - 1))

    # `git diff` does not report untracked files, so a brand-new file you have
    # not `git add`ed yet would be invisible — exactly the file you most want
    # analyzed. Treat each untracked file as wholly new. (Only meaningful for a
    # working-tree diff; comparing against a revision has no "untracked" side.)
    #
    # Two filters, both learned the hard way:
    #   - The index directory is untracked. Without excluding it, CodeNavigator
    #     reports its own index as a change and then declares the index stale
    #     because of it. Every re-index re-triggers it. Exclude it explicitly.
    #   - Untracked junk (logs, scratch files, downloads) is not code. For
    #     TRACKED changes we still report a non-source file as "changed but the
    #     graph is silent on it", because you deliberately committed it. For
    #     untracked ones that would just be noise, so we keep only source files.
    if rev is None and not staged:
        from .chunker import LANG_BY_EXT
        for rel in _untracked(repo):
            if rel.startswith(INDEX_DIRNAME + "/"):
                continue
            if Path(rel).suffix.lower() not in LANG_BY_EXT:
                continue
            fc = changes.setdefault(rel, FileChange(rel, [], 0))
            if not fc.ranges:
                try:
                    n = len((repo / rel).read_text(encoding="utf-8",
                                                   errors="ignore").splitlines())
                except OSError:
                    continue
                if n:
                    fc.ranges.append((1, n))

    return [c for c in changes.values()
            if (c.ranges or c.deletion_only_hunks)
            and not c.path.startswith(INDEX_DIRNAME + "/")]


def _untracked(repo: Path) -> list[str]:
    """Untracked files, honoring .gitignore (--exclude-standard)."""
    out = _git(repo, "ls-files", "--others", "--exclude-standard")
    return [ln.strip().replace("\\", "/") for ln in out.splitlines() if ln.strip()]


def stale_files(repo: Path, index_hashes: dict[str, str],
                changed: list[FileChange]) -> list[str]:
    """Changed files whose on-disk content no longer matches what was indexed.

    This is the guardrail. A stale span turns "line 47" into the wrong function,
    and the resulting impact set looks perfectly reasonable while being wrong.
    Better to stop and say "re-index" than to answer confidently from bad spans.
    """
    from .index import _hash_file  # same hash the manifest is built from

    stale: list[str] = []
    for c in changed:
        p = repo / c.path
        if not p.exists():
            continue
        recorded = index_hashes.get(c.path)
        if recorded is None:
            stale.append(c.path)          # never indexed
            continue
        try:
            if _hash_file(p) != recorded:
                stale.append(c.path)      # edited since indexing
        except OSError:
            stale.append(c.path)
    return stale


def changed_symbols(graph: CallGraph,
                    changes: list[FileChange]) -> tuple[list[Symbol], list[str]]:
    """Map changed line ranges to the innermost symbol enclosing each one.

    Returns (symbols, unmapped_paths). A path lands in `unmapped_paths` when it
    changed but no symbol covers the changed lines — a new file with no defs
    yet, an edit to imports or module-level config, a language with no grammar.
    That is information, not an error: it means "changed, but the graph has
    nothing to say about it", which is different from "safe".
    """
    by_path: dict[str, list[Symbol]] = {}
    for s in graph.symbols:
        by_path.setdefault(s.path.replace("\\", "/"), []).append(s)

    found: dict[int, Symbol] = {}
    unmapped: list[str] = []

    for c in changes:
        syms = by_path.get(c.path)
        if not syms:
            unmapped.append(c.path)
            continue
        hit = False
        for start, end in c.ranges:
            # Innermost symbol overlapping this hunk: smallest span wins, so an
            # edit inside a method maps to the method, not its enclosing class.
            best, best_size = None, None
            for s in syms:
                if s.end_line < start or s.start_line > end:
                    continue
                size = s.end_line - s.start_line
                if best_size is None or size < best_size:
                    best, best_size = s, size
            if best is not None:
                found[best.id] = best
                hit = True
        if not hit and c.ranges:
            unmapped.append(c.path)

    return list(found.values()), unmapped
