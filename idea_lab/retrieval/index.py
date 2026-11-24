"""The idea index: one vector and one BM25 document per idea, searched three ways.

- "vector": cosine similarity over normalised embeddings, served by a vector store (vectorstore.py):
  Qdrant by default, with each idea's stage and metadata as payload so filters run in the query, and
  exact NumPy search as the reference the eval checks it against.
- "bm25": keyword ranking (see bm25.py). Good at exact names and rare words.
- "hybrid": both rankings fused with reciprocal rank fusion (RRF): score = sum 1 / (k + rank).
  RRF only uses ranks, so it needs no score calibration between cosine and BM25, which live on
  different scales.
- "rerank": hybrid, then a cross-encoder re-scores the top RERANK_DEPTH (see rerank.py).

Incremental: each idea's text is hashed; on update only new or changed ideas are re-embedded, and
vectors are cached on disk (data/cache/, gitignored) keyed by embedder, so a run re-embeds a handful
of ideas, not all of them. Switching embedders builds a separate cache.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from .. import config
from ..store import Store
from .bm25 import BM25, tokens
from .embed import Embedder, get_embedder
from .rerank import get_reranker
from .vectorstore import ExactStore, VectorStore, make_store, payload_of

RRF_K = 60  # the constant from the original RRF paper (Cormack et al., 2009); damps the top ranks


def doc_text(idea: dict) -> str:
    """What gets indexed for an idea: name, what it does, and its place on the idea map (category and
    core mechanic). Short, so no chunking is needed. The category/mechanic suffix came out of the
    retrieval eval: scouts judge "nearest idea" largely by mechanic, and adding it lifted hybrid
    recall@5 on that benchmark from 0.34 to 0.52."""
    text = f"{idea.get('title') or ''}. {idea.get('pitch') or ''}".strip()
    tags = "; ".join(x for x in (idea.get("category"), idea.get("mechanic")) if x)
    return f"{text} ({tags})" if tags else text


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: -kv[1])


class Index:
    def __init__(self, embedder: Embedder | None, cache_dir: Path | None = None):
        self.embedder = embedder
        self.cache_dir = cache_dir
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.hashes: list[str] = []
        self.vecs = np.zeros((0, embedder.dim if embedder else 0), np.float32)
        self.payloads: list[dict] = []
        self.store: VectorStore | None = None
        if embedder:
            where = hashlib.sha1(str(cache_dir).encode()).hexdigest()[:8]  # one collection per data folder
            self.store = make_store(embedder.dim, f"fishtank-ideas-{embedder.name}-{where}")
        self.bm25 = BM25([])
        self.embedded_last_update = 0
        self._lock = threading.Lock()

    # ---- building -----------------------------------------------------------------------------
    def _cache_paths(self) -> tuple[Path, Path] | None:
        if not (self.cache_dir and self.embedder):
            return None
        base = self.cache_dir / f"index-{self.embedder.name}"
        return base.with_suffix(".npy"), base.with_suffix(".json")

    def _load_cache(self) -> dict[str, tuple[str, np.ndarray]]:
        paths = self._cache_paths()
        if not paths or not paths[0].exists() or not paths[1].exists():
            return {}
        try:
            meta = json.loads(paths[1].read_text())
            vecs = np.load(paths[0])
            return {i: (h, vecs[n]) for n, (i, h) in enumerate(zip(meta["ids"], meta["hashes"]))}
        except (ValueError, KeyError, OSError):
            return {}

    def _save_cache(self) -> None:
        paths = self._cache_paths()
        if not paths:
            return
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        np.save(paths[0], self.vecs)
        paths[1].write_text(json.dumps({"ids": self.ids, "hashes": self.hashes}))

    def update(self, ideas: dict[str, dict]) -> "Index":
        """Bring the index in line with the ideas; embed only what's new or changed."""
        with self._lock:
            ids = sorted(ideas)
            texts = [doc_text(ideas[i]) for i in ids]
            hashes = [hashlib.sha1(t.encode()).hexdigest()[:16] for t in texts]
            payloads = [payload_of(i, ideas[i]) for i in ids]
            if ids == self.ids and hashes == self.hashes and payloads == self.payloads:
                self.embedded_last_update = 0
                return self
            cached = {i: (h, v) for i, h, v in zip(self.ids, self.hashes, self.vecs)} if len(self.ids) else self._load_cache()
            todo = [n for n, (i, h) in enumerate(zip(ids, hashes)) if cached.get(i, ("", None))[0] != h]
            if self.embedder:
                vecs = np.zeros((len(ids), self.embedder.dim), np.float32)
                for n, i in enumerate(ids):
                    if n not in todo:
                        vecs[n] = cached[i][1]
                if todo:
                    vecs[todo] = self.embedder.embed([texts[n] for n in todo])
                self.vecs = vecs
                self.store.sync(ids, vecs, payloads, {ids[n] for n in todo})
            self.ids, self.texts, self.hashes, self.payloads = ids, texts, hashes, payloads
            self.bm25 = BM25([tokens(t) for t in texts])
            self.embedded_last_update = len(todo) if self.embedder else 0
            if todo:
                self._save_cache()
            return self

    # ---- searching ----------------------------------------------------------------------------
    def _vector_ranking(self, query: str, limit: int, allow=None, exclude=None, stages=None, exact: bool = False):
        """Nearest ideas by cosine, filtered in the store (Qdrant payload filters) when one is set."""
        if not self.embedder or not self.ids:
            return []
        q = self.embedder.embed([query])[0]
        store = self.store
        if exact or store is None:
            store = ExactStore()
            store.sync(self.ids, self.vecs, self.payloads, set())
        return store.search(q, limit, allow, exclude, stages)

    def _bm25_ranking(self, query: str) -> list[tuple[str, float]]:
        s = self.bm25.scores(tokens(query))
        order = np.argsort(-s)
        return [(self.ids[n], float(s[n])) for n in order if s[n] > 0]

    def search(self, query: str, k: int = 10, mode: str = "hybrid", exclude: set[str] | None = None,
               allow: set[str] | None = None, depth: int = 50, stages: set[str] | None = None) -> list[tuple[str, float]]:
        """Top k (id, score). Scores are cosine (vector), BM25 (bm25), RRF (hybrid) or the reranker's
        (rerank: hybrid, then a cross-encoder re-orders the top; see rerank.py)."""
        exclude = exclude or set()
        stage_of = dict(zip(self.ids, (p.get("stage") for p in self.payloads)))
        keep = (lambda i: i not in exclude and (allow is None or i in allow) and (stages is None or stage_of.get(i) in stages))
        if mode == "rerank":
            return self._reranked(query, k, exclude, allow, depth, stages)
        if mode == "vector" or (mode == "hybrid" and not self.bm25.tf):
            return self._vector_ranking(query, k, allow, exclude, stages)  # filtered inside the store
        if mode == "bm25" or not self.embedder:
            return [(i, s) for i, s in self._bm25_ranking(query) if keep(i)][:k]
        v = [i for i, _ in self._vector_ranking(query, depth, allow, exclude, stages)]
        b = [i for i, _ in self._bm25_ranking(query) if keep(i)][:depth]
        return rrf([v, b])[:k]

    def _reranked(self, query: str, k: int, exclude: set[str], allow: set[str] | None, depth: int,
                  stages: set[str] | None = None) -> list[tuple[str, float]]:
        """Hybrid search, then the reranker re-orders its top RERANK_DEPTH. Anything below that depth
        keeps its hybrid order after the reranked block (so recall at large k is unchanged)."""
        hybrid = self.search(query, max(k, depth), "hybrid", exclude, allow, depth, stages)
        rr = get_reranker()
        if rr is None:
            return hybrid[:k]
        head, tail = hybrid[:config.RERANK_DEPTH], hybrid[config.RERANK_DEPTH:]
        pos = {i: n for n, i in enumerate(self.ids)}
        scores = rr.score(query, [self.texts[pos[i]] for i, _ in head])
        order = sorted(range(len(head)), key=lambda n: -scores[n])
        return ([(head[n][0], float(scores[n])) for n in order] + tail)[:k]

    def similar(self, idea_id: str, k: int = 5, mode: str = "hybrid", allow: set[str] | None = None,
                stages: set[str] | None = None) -> list[tuple[str, float]]:
        if idea_id not in self.ids:
            return []
        return self.search(self.texts[self.ids.index(idea_id)], k, mode, exclude={idea_id}, allow=allow, stages=stages)


# One index per data folder, shared by the threads of a run and by the dashboard.
_INDEXES: dict[str, Index] = {}
_GUARD = threading.Lock()


def index_for(store: Store) -> Index:
    """The index for this data folder, updated to the current ideas (cheap when nothing changed)."""
    key = str(Path(store.root).resolve())
    with _GUARD:
        idx = _INDEXES.get(key)
        emb = get_embedder()
        if idx is None or (idx.embedder and emb and idx.embedder.name != emb.name):
            idx = _INDEXES[key] = Index(emb, Path(store.root) / "cache")
    return idx.update(with_map(store))


def with_map(store: Store) -> dict[str, dict]:
    """Ideas with category/mechanic filled in from the idea map where the idea itself has none."""
    classes = (store.config("idea_map") or {}).get("ideas", {})
    out = {}
    for i, d in store.ideas().items():
        c = classes.get(i) or {}
        out[i] = {**d, "category": d.get("category") or c.get("category"), "mechanic": d.get("mechanic") or c.get("mechanic")}
    return out
