"""Retrieval eval: how well do BM25, vector and hybrid search find the right idea? No model calls.

Two benchmarks built from labels already in the data:

1. "nearest" — every scout names the closest idea already in the lab (`nearest_existing`). Query with
   the new idea; is the idea the scout named among the results? Candidates are limited to ideas that
   existed when the scout wrote it. The label is one scout's judgement, so it's noisy: a miss isn't
   always wrong, which is why recall@5/@10 matter more here than recall@1.
2. "revision" — revised ideas keep their original pitch. Query with the original pitch only (no
   title, so the name can't give it away); is the revised idea found? Revisions often reword the
   whole pitch, so this measures matching on meaning rather than shared words.

Metrics: recall@k (target in the top k), MRR (mean of 1/rank of the target, 0 if not in the top 50).
"""
from __future__ import annotations

import json
import re
import statistics
from datetime import datetime, timezone
from difflib import SequenceMatcher

from ..store import Store
from .index import index_for, with_map

MODES = ("bm25", "vector", "hybrid")
KS = (1, 5, 10)
DEPTH = 50


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(t).lower()).strip()


def resolve(claim: str, ideas: dict[str, dict]) -> str | None:
    """Map a scout's free-text nearest_existing claim to an idea id (id, title, or a close title)."""
    c = str(claim or "").strip()
    if not c or c.lower().startswith("none"):
        return None
    if c in ideas:
        return c
    m = re.search(r"\(([a-z0-9-]+)\)", c)
    if m and m.group(1) in ideas:
        return m.group(1)
    nc = _norm(re.split(r"[:(/]", c)[0])
    best, score = None, 0.0
    for i, d in ideas.items():
        nt = _norm(re.split(r"[:(]", d.get("title") or "")[0])
        if not nt:
            continue
        s = 1.0 if nt == nc else SequenceMatcher(None, nt, nc).ratio()
        if s > score:
            best, score = i, s
    return best if score >= 0.85 else None


def cases(ideas: dict[str, dict]) -> dict[str, list[dict]]:
    nearest, revision = [], []
    for i, d in sorted(ideas.items()):
        t = resolve(d.get("nearest_existing"), ideas)
        if t and t != i:
            made = d.get("created_at") or ""
            allow = {j for j, e in ideas.items() if j != i and (not made or (e.get("created_at") or "") <= made)} | {t}
            nearest.append({"query_id": i, "target": t, "allow": allow})
        op = (d.get("original_pitch") or "").strip()
        if op and op != (d.get("pitch") or "").strip():
            revision.append({"query_text": op, "target": i})
    return {"nearest": nearest, "revision": revision}


def _score(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {f"recall@{k}": round(sum(1 for r in ranks if r is not None and r <= k) / n, 3) if n else None for k in KS}
    out["mrr"] = round(statistics.fmean(1 / r if r else 0 for r in ranks), 3) if n else None
    out["n"] = n
    return out


def run(store: Store) -> dict:
    ideas = with_map(store)
    idx = index_for(store)
    cs = cases(ideas)
    results: dict[str, dict] = {}
    for bench, items in cs.items():
        results[bench] = {}
        for mode in MODES:
            ranks = []
            for c in items:
                if bench == "nearest":
                    hits = idx.similar(c["query_id"], DEPTH, mode, allow=c["allow"])
                else:
                    hits = idx.search(c["query_text"], DEPTH, mode)
                ids = [h for h, _ in hits]
                ranks.append(ids.index(c["target"]) + 1 if c["target"] in ids else None)
            results[bench][mode] = _score(ranks)
    return {"suite": "retrieval", "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "embedder": idx.embedder.name if idx.embedder else "none (bm25 only)", "ideas": len(ideas),
            "results": results}


def table(report: dict) -> str:
    lines = [f"Retrieval eval · {report['ideas']} ideas · embedder {report['embedder']}"]
    for bench, modes in report["results"].items():
        n = next(iter(modes.values()))["n"]
        lines.append(f"\n{bench} (n={n})" + "".join(f"{m:>10}" for m in ("R@1", "R@5", "R@10", "MRR")))
        for mode, s in modes.items():
            lines.append(f"  {mode:<12}" + "".join(f"{(s[k] if s[k] is not None else float('nan')):>10.3f}"
                                                  for k in ("recall@1", "recall@5", "recall@10", "mrr")))
    return "\n".join(lines)


def save(store: Store, report: dict) -> str:
    d = store.root / "evals"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"retrieval-{report['at'].replace(':', '').replace('-', '')}.json"
    p.write_text(json.dumps(report, indent=2) + "\n")
    return str(p)
