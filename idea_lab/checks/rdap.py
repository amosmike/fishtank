"""Domain availability via RDAP: Verisign for .com, Nominet for .uk / .co.uk.

RDAP answers 200 for a registered domain and 404 for one that isn't. A 404 means "likely free",
not "free": premium/reserved names and registry quirks exist, so check the registrar before buying.
"""
from __future__ import annotations

import re

from . import Cache, http_get

ENDPOINTS = {
    "com": "https://rdap.verisign.com/com/v1/domain/{d}",
    "uk": "https://rdap.nominet.uk/uk/domain/{d}",
}


def label_for(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower().replace("&", "and"))[:63]


def check_domain(domain: str, cache: Cache | None = None) -> str:
    if cache and (hit := cache.get(domain)):
        return hit
    tld = "com" if domain.endswith(".com") else "uk" if domain.endswith(".uk") else None
    if not tld:
        return "unknown (no RDAP endpoint configured)"
    status, _, _ = http_get(ENDPOINTS[tld].format(d=domain), timeout=10, max_bytes=20_000)
    result = ("registered" if status == 200 else "not found (likely free)" if status == 404
              else f"unknown (RDAP status {status})")
    if cache and not result.startswith("unknown"):
        cache.set(domain, result)
    return result


def check_name(title: str, cache: Cache | None = None) -> dict[str, str]:
    label = label_for(title)
    if not label:
        return {}
    return {d: check_domain(d, cache) for d in (f"{label}.com", f"{label}.uk", f"{label}.co.uk")}
