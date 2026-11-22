"""Does each cited URL resolve? Results are cached for config.RIVAL_CACHE_DAYS."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from . import Cache, http_get

URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")


def urls_in(obj, limit: int = 40) -> list[str]:
    """Every distinct URL anywhere in a JSON-like object, in order, capped."""
    seen: list[str] = []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            for u in URL_RE.findall(o):
                u = u.rstrip(".,;:")
                if u not in seen:
                    seen.append(u)

    walk(obj)
    return seen[:limit]


def check(url: str, cache: Cache | None = None) -> int:
    if cache and (hit := cache.get(url)) is not None:
        return hit
    status, _, _ = http_get(url, timeout=10, method="HEAD")
    if status in (0, 403, 405, 400, 501):  # many sites refuse HEAD or bots; try a small GET
        status, _, _ = http_get(url, timeout=12, max_bytes=2_000)
    if cache:
        cache.set(url, status)
    return status


def check_all(urls: list[str], cache: Cache | None = None) -> dict[str, int]:
    with ThreadPoolExecutor(max_workers=8) as ex:
        return dict(zip(urls, ex.map(lambda u: check(u, cache), urls)))
