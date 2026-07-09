"""
embedder.py — turn text into vectors.

We wrap fastembed behind a tiny `Embedder` protocol. Two reasons:
  1. Tests can inject a fake embedder (no model download, no network).
  2. Swapping BGE-small for a bigger model, or for fastembed-rs later, is a
     one-class change — nothing downstream cares how vectors are produced.

fastembed uses ONNX (no PyTorch), so the install is light and it runs on CPU.
Default model BAAI/bge-small-en-v1.5 -> 384-dim vectors, which punch well
above their size for code/semantic search.

BGE models are trained with query/passage prefixes. We follow that convention:
documents get "passage: ", the user's question gets "query: ". This measurably
improves retrieval — it's not cosmetic.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    dim: int
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    """Real embedder backed by fastembed/ONNX."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from fastembed import TextEmbedding  # imported lazily so tests stay offline
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        # Probe the output dimension once.
        self.dim = int(next(iter(self._model.embed(["probe"]))).shape[0])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"passage: {t}" for t in texts]
        vecs = np.array(list(self._model.embed(prefixed)), dtype=np.float32)
        return _l2_normalize(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        vec = np.array(next(iter(self._model.embed([f"query: {text}"]))), dtype=np.float32)
        return _l2_normalize(vec[None, :])[0]


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length so cosine similarity == dot product."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms
