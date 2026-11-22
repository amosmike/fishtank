"""Fetch the data-source / licence pages an idea names, as plain text, for the logistics model to read."""
from __future__ import annotations

import html
import re

from . import Cache, http_get
from .url_check import URL_RE

LICENCE_HINTS = ("licen", "license", "terms", "copyright", "attribution", "open-government", "creativecommons",
                 "api", "data", "download", "pricing")


def page_text(raw: str, limit: int = 3_000) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", html.unescape(raw)).strip()
    # Keep the parts that mention licensing first, then the start of the page.
    hits = [m.start() for m in re.finditer(r"(?i)licen[cs]e|terms of use|copyright|attribution|commercial", text)]
    if hits:
        start = max(0, hits[0] - 300)
        return text[start:start + limit]
    return text[:limit]


def source_urls(idea: dict, limit: int = 4) -> list[str]:
    blob = " ".join(str(idea.get(f) or "") for f in ("data_sources", "build_plan"))
    lg = idea.get("logistics") or {}
    if isinstance(lg.get("data"), dict):
        blob += " " + str(lg["data"].get("url") or "")
    urls = [u.rstrip(".,;:") for u in URL_RE.findall(blob)]
    urls.sort(key=lambda u: not any(h in u.lower() for h in LICENCE_HINTS))
    out: list[str] = []
    for u in urls:
        if u not in out:
            out.append(u)
    return out[:limit]


def fetch(url: str, cache: Cache | None = None) -> dict:
    if cache and (hit := cache.get(url)):
        return hit
    status, final, body = http_get(url, timeout=15)
    out = {"url": url, "status": status, "final_url": final, "text": page_text(body) if status == 200 else ""}
    if cache and status == 200:
        cache.set(url, out)
    return out
