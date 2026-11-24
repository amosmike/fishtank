"""Rerankers: re-score the top candidates from hybrid search, reading query and idea together.

Retrieval is two stages. The first stage (BM25 + vectors, fused with RRF) is cheap and has to scan
every idea, so it compares fixed vectors and word counts that were computed for each text alone. A
cross-encoder takes one (query, idea) pair at a time through a transformer, so every word of the
query can attend to every word of the idea. That is far more accurate per pair and far too slow to run
over the whole lab, so it only re-orders the first stage's top RERANK_DEPTH.

- CrossEncoder: a fastembed cross-encoder (ONNX, local, no API key). Which model is the default came
  out of the retrieval eval; see config.RERANKER and README "Retrieval".
- OverlapReranker: a deterministic stand-in for tests and CI (word overlap, no download).

score(query, docs) returns one relevance score per doc; higher is better. Scales differ by model,
so only the order is used.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .. import config
from .bm25 import tokens

MODELS = {  # short name -> fastembed model id
    "minilm-l6": "Xenova/ms-marco-MiniLM-L-6-v2",
    "minilm-l12": "Xenova/ms-marco-MiniLM-L-12-v2",
    "bge-base": "BAAI/bge-reranker-base",
    "jina-tiny": "jinaai/jina-reranker-v1-tiny-en",
    "jina-turbo": "jinaai/jina-reranker-v1-turbo-en",
}


class Reranker(Protocol):
    name: str

    def score(self, query: str, docs: list[str]) -> np.ndarray: ...


class CrossEncoder:
    def __init__(self, short: str):
        self.name, self.model_id = short, MODELS[short]
        self._model = None

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.zeros(0, np.float32)
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder  # lazily: only needed when reranking
            self._model = TextCrossEncoder(self.model_id)
        return np.array(list(self._model.rerank(query, docs, batch_size=32)), np.float32)


class OverlapReranker:
    """Share of the query's words found in the doc. Deterministic and instant; for tests only."""
    name = "overlap"

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        q = set(tokens(query))
        return np.array([len(q & set(tokens(d))) / (len(q) or 1) for d in docs], np.float32)


_CACHE: dict[str, Reranker] = {}


def get_reranker(name: str | None = None) -> Reranker | None:
    """The configured reranker, or None when it's off or can't load (search then stays hybrid)."""
    name = name or config.RERANKER
    if name in _CACHE:
        return _CACHE[name]
    if name == "overlap":
        r: Reranker | None = OverlapReranker()
    elif name in MODELS:
        try:
            import fastembed  # noqa: F401
            r = CrossEncoder(name)
        except ImportError:
            r = None
    else:  # "off"
        r = None
    if r is not None:
        _CACHE[name] = r
    return r
