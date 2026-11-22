"""Pure-code checks the logistics node runs before any model sees the idea."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .. import config

UA = "idea-lab-checks/0.1 (+static site research)"


def http_get(url: str, timeout: float = 12, max_bytes: int = 400_000, method: str = "GET") -> tuple[int, str, str]:
    """(status, final_url, body text). status 0 = network error."""
    req = urllib.request.Request(url, method=method, headers={"User-Agent": UA, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(max_bytes) if method == "GET" else b""
            charset = r.headers.get_content_charset() or "utf-8"
            return r.status, r.geturl(), body.decode(charset, "replace")
    except urllib.error.HTTPError as e:
        return e.code, url, ""
    except Exception as e:  # noqa: BLE001 - DNS, TLS, timeouts: all "didn't resolve"
        return 0, url, str(e)[:200]


class Cache:
    """Tiny JSON cache with a TTL, under data/cache/ (gitignored)."""

    def __init__(self, name: str, ttl_days: float):
        self.path = Path(config.CACHE) / f"{name}.json"
        self.ttl = ttl_days * 86400
        try:
            self.data = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {}

    def get(self, key: str):
        v = self.data.get(key)
        if v and time.time() - v.get("_t", 0) < self.ttl:
            return v.get("v")
        return None

    def set(self, key: str, value) -> None:
        self.data[key] = {"_t": time.time(), "v": value}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))
