"""
index.py — build the index incrementally: chunk -> embed -> store.

The first `index` run embeds every chunk. Every run after that should be
near-instant, because embedding (model inference) is the only expensive step
and most files don't change between runs.

How the incremental path works:
  1. Hash the content of every source file on disk (fast — a few ms of I/O).
  2. Compare against the `{path: hash}` manifest saved by the last run.
       - unchanged files -> keep their existing chunks + vectors, skip entirely
       - changed / new    -> re-chunk + re-embed, replacing any old chunks
       - deleted          -> drop their chunks from the store
  3. Persist the updated store + the new file manifest.

If the embedding model changed since the last run, mixing old and new vectors
would silently corrupt retrieval, so we detect that and rebuild from scratch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .chunker import Chunk, chunk_file, iter_source_files
from .embedder import Embedder
from .store import VectorStore

INDEX_DIRNAME = ".codenavigator"
FILES_MANIFEST = "files.json"


def index_dir_for(repo_path: Path) -> Path:
    return repo_path.resolve() / INDEX_DIRNAME


# ---------------------------------------------------------------------------
# File hashing + manifest
# ---------------------------------------------------------------------------

def _hash_file(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _scan(repo_path: Path) -> dict[str, tuple[Path, str]]:
    """Map repo-relative path -> (absolute path, content hash)."""
    out: dict[str, tuple[Path, str]] = {}
    for p in iter_source_files(repo_path):
        rel = str(p.relative_to(repo_path))
        out[rel] = (p, _hash_file(p))
    return out


def _load_manifest(index_dir: Path) -> dict:
    fp = index_dir / FILES_MANIFEST
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_manifest(index_dir: Path, embed_model: str | None, files: dict[str, str]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / FILES_MANIFEST).write_text(
        json.dumps({"embed_model": embed_model, "files": files}, indent=2)
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_and_add(store: VectorStore, chunks: list[Chunk], embedder: Embedder,
                   batch_size: int, progress) -> None:
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        # Embed header + text so the file/symbol locator is part of the signal.
        texts = [f"{c.header()}\n{c.text}" for c in batch]
        vecs = embedder.embed_documents(texts)
        store.add(vecs, [c.to_row() for c in batch])
        progress(f"  embedded {min(i + batch_size, len(chunks))}/{len(chunks)}")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_index(repo_path: Path, embedder: Embedder, batch_size: int = 64,
                force: bool = False, build_graph: bool = True, progress=print) -> VectorStore:
    """Build or incrementally update the index for a repo.

    force=True rebuilds from scratch, ignoring any existing index.
    build_graph=True also (re)builds the call graph (cheap; always full).
    """
    repo_path = repo_path.resolve()
    index_dir = index_dir_for(repo_path)
    model_name = getattr(embedder, "model_name", None)

    progress(f"Scanning {repo_path} ...")
    current = _scan(repo_path)              # rel -> (abs, hash)
    if not current:
        raise SystemExit("No source files found to index.")
    cur_hashes = {rel: h for rel, (_, h) in current.items()}

    # Decide between incremental update and full rebuild.
    manifest = {} if force else _load_manifest(index_dir)
    prev_hashes: dict[str, str] = manifest.get("files", {})
    prev_model = manifest.get("embed_model")
    has_existing = bool(prev_hashes) and (index_dir / "manifest.json").exists()

    if has_existing and prev_model not in (None, model_name):
        progress(f"Embedding model changed ({prev_model} -> {model_name}); rebuilding fully.")
        has_existing = False

    if has_existing:
        try:
            store = VectorStore.load(index_dir)
            if store.dim != embedder.dim:      # dimension mismatch -> rebuild
                raise ValueError("dim mismatch")
        except Exception:
            has_existing = False

    if not has_existing:
        store = VectorStore(dim=embedder.dim)
        changed = set(cur_hashes)              # everything is "new"
        deleted: set[str] = set()
        progress(f"Full build: {len(changed)} files.")
    else:
        changed = {rel for rel, h in cur_hashes.items() if prev_hashes.get(rel) != h}
        deleted = set(prev_hashes) - set(cur_hashes)
        unchanged = len(cur_hashes) - len(changed)
        if not changed and not deleted:
            progress(f"Index up to date — {unchanged} files unchanged. Nothing to do.")
            return store
        progress(f"Incremental: {len(changed)} changed/new, {len(deleted)} deleted, "
                 f"{unchanged} unchanged.")

    # 1) Remove stale chunks (changed files get their old chunks dropped too).
    removed = store.remove_paths(changed | deleted)
    if removed:
        progress(f"  dropped {removed} stale chunks.")

    # 2) Re-chunk + embed the changed/new files.
    new_chunks: list[Chunk] = []
    for rel in sorted(changed):
        abs_path, _ = current[rel]
        new_chunks.extend(chunk_file(abs_path, repo_path))
    if new_chunks:
        progress(f"Chunked {len(new_chunks)} units from {len(changed)} files. Embedding ...")
        _embed_and_add(store, new_chunks, embedder, batch_size, progress)

    # 3) Persist store + file manifest.
    store.persist(index_dir)
    _save_manifest(index_dir, model_name, cur_hashes)
    progress(f"Index written to {index_dir} ({len(store)} chunks, {len(cur_hashes)} files).")

    # 4) Call graph. Rebuilt in full each run (parsing is cheap; only embedding
    #    needs to be incremental). Resolution is global, so a partial graph
    #    would go stale as soon as any cross-file caller changes.
    if build_graph:
        try:
            from .callgraph import build_call_graph, save_graph
            graph = build_call_graph(repo_path, progress=progress)
            save_graph(graph, index_dir)
        except Exception as e:
            progress(f"(call graph skipped: {e})")

    return store
