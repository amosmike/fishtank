"""Okapi BM25: the standard keyword-ranking function (what Elasticsearch/Lucene use by default).

score(doc, query) = sum over query terms t of
    idf(t) * tf(t, doc) * (k1 + 1) / (tf(t, doc) + k1 * (1 - b + b * len(doc) / avg_len))

- idf: rare terms count more (a match on "hydrograph" says more than a match on "daily").
- k1 (1.5): term-frequency saturation; the 5th "game" adds far less than the 1st.
- b (0.75): length normalisation; long pitches don't win just by having more words.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

STOP = set("""a an and are as at be by for from has have in into is it its of on or that the this to with your you
it's their they them was were will who what which when where how than then so not no can our we us i my
one two each every day daily get""".split())


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP or len(w) < 2:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # crude plural folding: "counties" stays, "games" -> "game"
        out.append(w)
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf), dtype=np.float32)
        if not self.tf:
            return s
        norm = self.k1 * (1 - self.b + self.b * self.len / (self.avg or 1.0))
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            f = np.array([tf.get(t, 0) for tf in self.tf], dtype=np.float32)
            s += idf * f * (self.k1 + 1) / (f + norm)
        return s
