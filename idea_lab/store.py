"""Git-tracked JSON store: one file per document under data/<collection>/<id>.json.

Every write to an idea goes through `Store.update_idea`, which enforces Michael's rule: an idea in a
stage he decided (selected, parked, rejected, building, hosted) only ever gets history appended.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import config
from .rules import PROTECTED_STAGES, is_protected


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower().replace("'", "")).strip("-")
    return s[:60] or "idea"


def dump(doc: dict) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


class ProtectedIdea(Exception):
    pass


class Store:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or config.DATA)

    # ---- generic documents ---------------------------------------------------------------
    def path(self, collection: str, doc_id: str) -> Path:
        return self.root / collection / f"{doc_id}.json"

    def get(self, collection: str, doc_id: str, default: Any = None) -> Any:
        p = self.path(collection, doc_id)
        if not p.exists():
            return default
        return json.loads(p.read_text())

    def put(self, collection: str, doc_id: str, doc: dict) -> None:
        p = self.path(collection, doc_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(dump(doc))
        tmp.replace(p)

    def ids(self, collection: str) -> list[str]:
        d = self.root / collection
        return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []

    def all(self, collection: str) -> dict[str, dict]:
        return {i: self.get(collection, i) for i in self.ids(collection)}

    # ---- ideas ------------------------------------------------------------------------------
    def ideas(self) -> dict[str, dict]:
        return self.all("ideas")

    def idea(self, idea_id: str) -> dict:
        d = self.get("ideas", idea_id)
        if d is None:
            raise KeyError(idea_id)
        return d

    def new_idea_id(self, title: str) -> str:
        base = slugify(title)
        taken = set(self.ids("ideas"))
        if base not in taken:
            return base
        n = 2
        while f"{base}-{n}" in taken:
            n += 1
        return f"{base}-{n}"

    def create_idea(self, idea_id: str, doc: dict, by: str, note: str = "") -> dict:
        if self.path("ideas", idea_id).exists():
            raise FileExistsError(idea_id)
        t = now_iso()
        doc = {**doc, "created_at": doc.get("created_at", t), "updated_at": t}
        doc.setdefault("history", []).append({"at": t, "by": by, "stage": doc.get("stage"), "note": note})
        self.put("ideas", idea_id, doc)
        return doc

    def update_idea(self, idea_id: str, patch: dict, by: str, note: str = "",
                    allow_protected: bool = False) -> dict:
        """Merge `patch` into the idea and append one history line.

        On a protected idea (Michael decided) only the history line is written, unless
        `allow_protected` is set by the one path the playbook allows (applying Michael's own
        decisions, and the parked-revival exception he approved).
        """
        doc = self.idea(idea_id)
        t = now_iso()
        if is_protected(doc) and not allow_protected:
            refused = sorted(k for k in patch if k != "history")
            doc.setdefault("history", []).append({
                "at": t, "by": by, "stage": doc.get("stage"),
                "note": (note + (f" (not applied: Michael's decision; fields {', '.join(refused)})" if refused else "")).strip(),
            })
            self.put("ideas", idea_id, doc)
            return doc
        doc.update(patch)
        doc["updated_at"] = t
        doc.setdefault("history", []).append({"at": t, "by": by, "stage": doc.get("stage"), "note": note})
        self.put("ideas", idea_id, doc)
        return doc

    def append_history(self, idea_id: str, by: str, note: str) -> None:
        self.update_idea(idea_id, {}, by, note)

    # ---- runs ---------------------------------------------------------------------------------
    def runs(self) -> list[dict]:
        """All runs, oldest first, each with its id under `_id`."""
        out = []
        for rid in self.ids("runs"):
            r = self.get("runs", rid)
            r["_id"] = rid
            out.append(r)
        out.sort(key=lambda r: r.get("started_at") or r["_id"])
        return out

    def run(self, run_id: str) -> dict:
        return self.get("runs", run_id) or {}

    def update_run(self, run_id: str, patch: dict | None = None, log: Iterable[dict] = ()) -> dict:
        r = self.run(run_id)
        if patch:
            r.update(patch)
        for entry in log:
            r.setdefault("log", []).append({"t": now_iso(), **entry})
        self.put("runs", run_id, r)
        return r

    # ---- config -------------------------------------------------------------------------------
    def config(self, name: str, default: Any = None) -> Any:
        return self.get("config", name, default)


__all__ = ["Store", "ProtectedIdea", "PROTECTED_STAGES", "now_iso", "parse_ts", "slugify"]
