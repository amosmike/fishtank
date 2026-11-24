"""Vector stores: where the idea vectors live and how they're searched, with metadata filters.

- QdrantStore (default): a Qdrant collection, one point per idea. Each point carries its idea's
  metadata as payload (stage, source, kind, lane, created_at), so filters run inside the query.
  The default is Qdrant's local mode in memory, loaded from the on-disk embedding cache at start-up
  (160 points take milliseconds). In memory rather than on disk because local mode locks its folder
  to one process, and the dashboard and a run are two processes. Set IDEA_LAB_QDRANT_URL to use a
  Qdrant server instead: the same code then gets a persistent, shared collection with an HNSW index.
- ExactStore: a matrix-vector product over the normalised vectors. The reference: the retrieval eval
  checks the Qdrant results against it, and it's the fallback when qdrant-client isn't installed.

At a few hundred ideas exact search is microseconds and never wrong, so Qdrant here is about the
interface (payload filters, the same client against a server) rather than speed. Local mode searches
exhaustively, so it agrees with ExactStore; a server's HNSW index is approximate, and the eval's
parity check is what would show the difference.

Both take normalised float32 vectors and return [(idea_id, cosine)] best first.
"""
from __future__ import annotations

import threading
import uuid
from typing import Protocol

import numpy as np

from .. import config

PAYLOAD_KEYS = ("stage", "source", "kind", "lane", "created_at")


def payload_of(idea_id: str, idea: dict) -> dict:
    return {"idea_id": idea_id, **{k: idea.get(k) for k in PAYLOAD_KEYS if idea.get(k) is not None}}


class VectorStore(Protocol):
    name: str

    def sync(self, ids: list[str], vecs: np.ndarray, payloads: list[dict], changed: set[str]) -> None: ...

    def search(self, q: np.ndarray, limit: int, allow: set[str] | None = None, exclude: set[str] | None = None,
               stages: set[str] | None = None) -> list[tuple[str, float]]: ...


class ExactStore:
    name = "exact"

    def __init__(self):
        self.ids: list[str] = []
        self.vecs = np.zeros((0, 0), np.float32)
        self.payloads: list[dict] = []

    def sync(self, ids, vecs, payloads, changed):
        self.ids, self.vecs, self.payloads = list(ids), vecs, list(payloads)

    def search(self, q, limit, allow=None, exclude=None, stages=None):
        if not self.ids:
            return []
        sims = self.vecs @ q
        out = []
        for n in np.argsort(-sims, kind="stable"):
            i = self.ids[n]
            if (exclude and i in exclude) or (allow is not None and i not in allow) or \
                    (stages is not None and self.payloads[n].get("stage") not in stages):
                continue
            out.append((i, float(sims[n])))
            if len(out) >= limit:
                break
        return out


def _point_id(idea_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"fishtank:idea:{idea_id}"))  # Qdrant ids are ints or UUIDs


class QdrantStore:
    def __init__(self, collection: str, dim: int, url: str | None = None):
        from qdrant_client import QdrantClient, models  # imported lazily: optional dependency
        self.m = models
        self.client = QdrantClient(url=url) if url else QdrantClient(location=":memory:")
        self.name = f"qdrant ({'server' if url else 'local, in memory'})"
        self.collection = collection
        self._lock = threading.Lock()
        self._known: dict[str, dict] = {}  # idea id -> payload last written
        if not self.client.collection_exists(collection):
            self.client.create_collection(collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
            for key in ("idea_id", "stage"):  # indexed payload fields make filtered queries fast on a server
                self.client.create_payload_index(collection, key, models.PayloadSchemaType.KEYWORD)

    def sync(self, ids, vecs, payloads, changed):
        """Upsert points whose vector or payload changed; delete ideas that are gone."""
        m = self.m
        with self._lock:
            if not self._known and self.client.count(self.collection).count:  # a server already holds points
                self._known = {p.payload["idea_id"]: p.payload for p in self._scroll_all()}
            todo = [n for n, i in enumerate(ids) if i in changed or self._known.get(i) != payloads[n]]
            for start in range(0, len(todo), 256):
                batch = todo[start:start + 256]
                self.client.upsert(self.collection, points=[
                    m.PointStruct(id=_point_id(ids[n]), vector=vecs[n].tolist(), payload=payloads[n]) for n in batch])
            gone = set(self._known) - set(ids)
            if gone:
                self.client.delete(self.collection, points_selector=m.PointIdsList(points=[_point_id(i) for i in gone]))
            self._known = {i: payloads[n] for n, i in enumerate(ids)}

    def _scroll_all(self):
        out, offset = [], None
        while True:
            pts, offset = self.client.scroll(self.collection, limit=512, offset=offset, with_payload=True)
            out += pts
            if offset is None:
                return out

    def search(self, q, limit, allow=None, exclude=None, stages=None):
        m = self.m
        must, must_not = [], []
        if allow is not None:
            if not allow:
                return []
            must.append(m.FieldCondition(key="idea_id", match=m.MatchAny(any=sorted(allow))))
        if stages is not None:
            must.append(m.FieldCondition(key="stage", match=m.MatchAny(any=sorted(stages))))
        if exclude:
            must_not.append(m.FieldCondition(key="idea_id", match=m.MatchAny(any=sorted(exclude))))
        flt = m.Filter(must=must or None, must_not=must_not or None) if (must or must_not) else None
        with self._lock:
            res = self.client.query_points(self.collection, query=q.tolist(), limit=limit, query_filter=flt, with_payload=["idea_id"])
        return [(p.payload["idea_id"], float(p.score)) for p in res.points]


def make_store(dim: int, collection: str) -> VectorStore:
    """The configured store; exact search if Qdrant is off or not installed."""
    if config.VECTOR_STORE == "qdrant":
        try:
            return QdrantStore(collection, dim, config.QDRANT_URL)
        except ImportError:
            pass
    return ExactStore()
