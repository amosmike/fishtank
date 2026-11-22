"""Code half of dedupe: drop exact and near-identical titles before the cheap model pass."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

STOP = {"the", "a", "an", "uk", "daily", "game", "my", "your", "of", "and", "for", "online"}


def norm_title(t: str) -> str:
    words = re.sub(r"[^a-z0-9 ]+", " ", str(t).lower()).split()
    return " ".join(w for w in words if w not in STOP)


def similar(a: str, b: str) -> float:
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def code_dedupe(candidates: list[dict], existing: dict[str, dict], threshold: float = 0.88):
    """Returns (kept, dropped). dropped items carry `duplicate_of` and `reason`."""
    kept: list[dict] = []
    dropped: list[dict] = []
    titles = {i: d.get("title") or i for i, d in existing.items()}
    for c in candidates:
        t = c.get("title", "")
        hit = next((i for i, et in titles.items() if similar(t, et) >= threshold), None)
        if hit:
            dropped.append({**c, "duplicate_of": hit, "reason": "same or near-identical title"})
            continue
        twin = next((k for k in kept if similar(t, k.get("title", "")) >= threshold), None)
        if twin:
            dropped.append({**c, "duplicate_of": twin.get("title"), "reason": "same-batch title"})
            continue
        kept.append(c)
    return kept, dropped


def cap_batch(ideas: list[dict], max_total: int, max_big: int) -> tuple[list[dict], list[dict]]:
    """Keep at most max_total ideas, at most max_big of them lane "big", in scout order."""
    kept, over = [], []
    big = 0
    for c in ideas:
        is_big = c.get("lane") == "big"
        if len(kept) >= max_total or (is_big and big >= max_big):
            over.append(c)
            continue
        big += is_big
        kept.append(c)
    return kept, over
