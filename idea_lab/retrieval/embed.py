"""Embedders: text -> unit-length vectors, so cosine similarity is a dot product.

- BgeSmall: BAAI/bge-small-en-v1.5 (384 dims) run locally through fastembed (ONNX, no PyTorch, no API
  key). Small, fast on a laptop CPU, and strong on short English text like titles and pitches.
- HashEmbedder: a deterministic stand-in for tests and CI. It hashes words and word pairs into a fixed
  number of buckets ("feature hashing"), so it captures word overlap but no meaning. Never downloads.

Both return L2-normalised float32 arrays of shape (n, dim).
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .. import config


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.where(n == 0, 1, n)).astype(np.float32)


class BgeSmall:
    name = "bge-small-en-v1.5"
    dim = 384

    def __init__(self):
        self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        if self._model is None:
            from fastembed import TextEmbedding  # imported lazily: only needed when embedding
            self._model = TextEmbedding("BAAI/bge-small-en-v1.5")
        return _normalise(np.array(list(self._model.embed(texts, batch_size=64))))


class HashEmbedder:
    name = "hash-512"
    dim = 512

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for row, text in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", text.lower())
            for f in words + [a + " " + b for a, b in zip(words, words[1:])]:
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0  # signed hashing reduces collision bias
        return _normalise(out)


_CACHE: dict[str, Embedder] = {}


def get_embedder() -> Embedder | None:
    """The configured embedder, or None if it can't be loaded (search then falls back to BM25 only)."""
    name = config.EMBEDDER
    if name in _CACHE:
        return _CACHE[name]
    if name == "hash":
        emb: Embedder | None = HashEmbedder()
    elif name == "bge-small":
        try:
            import fastembed  # noqa: F401
            emb = BgeSmall()
        except ImportError:
            emb = None
    else:
        emb = None
    if emb is not None:
        _CACHE[name] = emb
    return emb
