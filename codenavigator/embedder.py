"""
embedder.py — turn text into vectors.

We wrap fastembed behind a tiny `Embedder` protocol. Two reasons:
  1. Tests can inject a fake embedder (no model download, no network).
  2. Swapping BGE-small for a bigger model, or for fastembed-rs later, is a
     one-class change — nothing downstream cares how vectors are produced.

fastembed uses ONNX (no PyTorch), so the install is light and it runs on CPU.
Default model BAAI/bge-small-en-v1.5 -> 384-dim vectors, which punch well
above their size for code/semantic search.

Instruction prefixes
--------------------
Retrieval-trained embedding models are sensitive to the exact prefix strings
they were trained with, and the convention is NOT shared across model families:

  BGE (BAAI/bge-*-v1.5)
      query   -> "Represent this sentence for searching relevant passages: "
      passage -> no prefix at all
      BGE v1.5 was tuned to retrieve well *without* the query instruction, so
      this is a small optional gain, not a requirement.

  E5 (intfloat/e5-*, multilingual-e5-*)
      query   -> "query: "
      passage -> "passage: "

fastembed does NOT insert any prefix for dense text models — its `query_embed`
and `passage_embed` are pass-throughs to `embed`. Whatever we pass is what the
model sees, so the prefix is our responsibility. Getting it wrong means feeding
the model out-of-distribution tokens on every single chunk.

Rather than assert which scheme wins, we make it a knob and let eval.py settle
it. The prefix is part of the embedder's fingerprint, so changing it forces a
full re-index — you can't accidentally score fresh query vectors against stale
passage vectors.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# (query_prefix, passage_prefix) per convention.
PREFIX_SCHEMES: dict[str, tuple[str, str]] = {
    "bge": ("Represent this sentence for searching relevant passages: ", ""),
    "e5": ("query: ", "passage: "),
    "none": ("", ""),
}
DEFAULT_PREFIX = "bge"


class Embedder(Protocol):
    dim: int
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    """Real embedder backed by fastembed/ONNX."""

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 prefix: str = DEFAULT_PREFIX):
        from fastembed import TextEmbedding  # imported lazily so tests stay offline

        if prefix not in PREFIX_SCHEMES:
            raise ValueError(f"unknown prefix scheme {prefix!r}; "
                             f"choose from {sorted(PREFIX_SCHEMES)}")
        self.model_name = model_name
        self.prefix = prefix
        self._q_prefix, self._p_prefix = PREFIX_SCHEMES[prefix]
        self._model = TextEmbedding(model_name=model_name)
        # Probe the output dimension once.
        self.dim = int(next(iter(self._model.embed(["probe"]))).shape[0])

    @property
    def fingerprint(self) -> str:
        """Identifies the vector space. The index rebuilds when this changes —
        a different prefix scheme yields incompatible passage vectors."""
        return f"{self.model_name}|prefix={self.prefix}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"{self._p_prefix}{t}" for t in texts] if self._p_prefix else texts
        vecs = np.array(list(self._model.embed(prefixed)), dtype=np.float32)
        return _l2_normalize(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        vec = np.array(next(iter(self._model.embed([f"{self._q_prefix}{text}"]))),
                       dtype=np.float32)
        return _l2_normalize(vec[None, :])[0]


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length so cosine similarity == dot product."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms
